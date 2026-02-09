"""
radar_processor.py — Core radar signal processing for FMCW range-Doppler.

Handles chirp extraction from raw IQ buffers, windowing, 2D FFT,
and MTI clutter filtering.
"""

import numpy as np


def extract_chirps(raw_iq, num_chirps, samples_per_frame, good_samples, start_offset):
    """Extract individual chirp segments from a continuous IQ buffer.

    Args:
        raw_iq: Complex IQ data from one receive buffer (sum or per-channel).
        num_chirps: Number of chirps in the burst.
        samples_per_frame: Samples per PRI frame (PRI_s * sample_rate).
        good_samples: Number of usable samples per chirp (after PLL settling).
        start_offset: Sample offset into each frame where good data begins.

    Returns:
        2D array of shape (num_chirps, good_samples), complex.
    """
    # Vectorized: compute all start indices at once, use fancy indexing
    starts = start_offset + np.arange(num_chirps) * samples_per_frame
    offsets = np.arange(good_samples)
    indices = starts[:, np.newaxis] + offsets[np.newaxis, :]

    valid = indices[:, -1] < len(raw_iq)
    chirps = np.zeros((num_chirps, good_samples), dtype=complex)
    if np.any(valid):
        chirps[valid] = raw_iq[indices[valid]]
    return chirps


def apply_window_2d(chirps, range_window='hann', doppler_window='hann'):
    """Apply window functions along range (fast-time) and Doppler (slow-time).

    Args:
        chirps: 2D array (num_chirps, num_samples), complex.
        range_window: Window type for fast-time dimension ('hann', 'blackmanharris', 'hamming', 'none').
        doppler_window: Window type for slow-time dimension ('hann', 'blackmanharris', 'hamming', 'none').

    Returns:
        Windowed chirp array, same shape.
    """
    num_chirps, num_samples = chirps.shape
    windowed = chirps.copy()

    # Range window (applied along columns — each chirp independently)
    if range_window != 'none':
        rwin = _get_window(range_window, num_samples)
        windowed *= rwin[np.newaxis, :]

    # Doppler window (applied along rows — across chirps at each range bin)
    if doppler_window != 'none':
        dwin = _get_window(doppler_window, num_chirps)
        windowed *= dwin[:, np.newaxis]

    return windowed


def _get_window(name, length):
    """Return a numpy window of the given type and length."""
    windows = {
        'hann': np.hanning,
        'hamming': np.hamming,
        'blackman': np.blackman,
        'blackmanharris': np.blackman,  # numpy doesn't have blackman-harris, use blackman
    }
    if name in windows:
        return windows[name](length)
    raise ValueError(f"Unknown window type: {name}")


def range_doppler_fft(chirps, fft_size_range=None, fft_size_doppler=None):
    """Compute the 2D FFT to produce a range-Doppler map.

    Args:
        chirps: 2D array (num_chirps, num_samples), complex.
        fft_size_range: FFT size for range dimension (zero-padded if > num_samples). None = num_samples.
        fft_size_doppler: FFT size for Doppler dimension (zero-padded). None = num_chirps.

    Returns:
        2D array (fft_size_range, fft_size_doppler) of magnitude in dB (log10 scale).
        Also returns the complex FFT result for phase-based processing.
    """
    num_chirps, num_samples = chirps.shape
    if fft_size_range is None:
        fft_size_range = num_samples
    if fft_size_doppler is None:
        fft_size_doppler = num_chirps

    # Range FFT along fast-time (axis=1), then Doppler FFT along slow-time (axis=0)
    rd_complex = np.fft.fftshift(np.fft.fft2(chirps, s=(fft_size_doppler, fft_size_range)))

    rd_magnitude = np.abs(rd_complex)
    # Avoid log(0)
    rd_magnitude[rd_magnitude == 0] = 1e-12
    rd_db = 20 * np.log10(rd_magnitude)

    return rd_db, rd_complex


def mti_2pulse(chirps):
    """2-pulse canceller MTI filter (original method from Range_Doppler_Plot.py).

    Subtracts consecutive chirps with phase correction to remove static clutter.
    Vectorized — no Python loop over chirps.

    Args:
        chirps: 2D array (num_chirps, num_samples), complex.

    Returns:
        Filtered chirps array (num_chirps, num_samples). Last chirp is a copy of the previous result.
    """
    num_chirps, num_samples = chirps.shape

    # Batch dot products: sum(c0 * conj(c1)) for all consecutive pairs
    dots = np.sum(chirps[:-1] * np.conj(chirps[1:]), axis=1)
    phases = np.angle(dots)

    # Phase-corrected subtraction, all pairs at once
    filtered = np.zeros_like(chirps)
    filtered[:-1] = chirps[1:] - chirps[:-1] * np.exp(-1j * phases[:, np.newaxis])
    filtered[-1] = filtered[-2] if num_chirps > 1 else chirps[-1]
    return filtered


def mti_3pulse(chirps):
    """3-pulse canceller MTI filter — deeper clutter notch than 2-pulse.

    Applies y[n] = x[n] - 2*x[n+1] + x[n+2] with phase correction.
    This produces a sin^2 frequency response (vs sin for 2-pulse),
    giving a wider and deeper null at zero Doppler.
    Vectorized — no Python loop over chirps.

    Args:
        chirps: 2D array (num_chirps, num_samples), complex.

    Returns:
        Filtered chirps array (num_chirps, num_samples).
    """
    num_chirps, num_samples = chirps.shape

    # Batch phase corrections: dot products for all consecutive pairs
    dots_01 = np.sum(chirps[:-2] * np.conj(chirps[1:-1]), axis=1)
    phases_01 = np.angle(dots_01)

    dots_12 = np.sum(chirps[1:-1] * np.conj(chirps[2:]), axis=1)
    phases_12 = np.angle(dots_12)

    # Align and combine, all triplets at once
    c0_aligned = chirps[:-2] * np.exp(-1j * phases_01[:, np.newaxis])
    c2_aligned = chirps[2:] * np.exp(1j * phases_12[:, np.newaxis])

    filtered = np.zeros_like(chirps)
    filtered[:num_chirps - 2] = c0_aligned - 2 * chirps[1:-1] + c2_aligned

    if num_chirps > 2:
        filtered[-2] = filtered[-3]
        filtered[-1] = filtered[-3]
    return filtered


def mti_highpass(chirps, cutoff_normalized=0.05, order=2):
    """High-pass IIR MTI filter along slow-time (Butterworth).

    Filters each range bin independently across chirps to remove
    low-Doppler clutter.

    Args:
        chirps: 2D array (num_chirps, num_samples), complex.
        cutoff_normalized: Cutoff frequency as fraction of PRF (0 to 0.5).
        order: Filter order.

    Returns:
        Filtered chirps array.
    """
    from scipy.signal import butter, sosfilt

    sos = butter(order, cutoff_normalized, btype='high', output='sos')

    # sosfilt supports axis= parameter, no Python loop needed
    filtered_real = sosfilt(sos, chirps.real, axis=0)
    filtered_imag = sosfilt(sos, chirps.imag, axis=0)
    return filtered_real + 1j * filtered_imag


def compute_axes(sample_rate, output_freq, signal_freq, chirp_bw, ramp_time_s,
                 num_chirps, pri_s, samples_per_frame):
    """Compute range and velocity axis arrays and resolution values.

    Returns:
        dict with keys: range_axis, velocity_axis, range_res, velocity_res,
                        max_range, max_velocity, slope
    """
    c = 3e8
    wavelength = c / output_freq
    slope = chirp_bw / ramp_time_s

    # Range axis (from beat frequency)
    freq = np.linspace(-sample_rate / 2, sample_rate / 2, samples_per_frame)
    range_axis = (freq - signal_freq) * c / (2 * slope)

    # Velocity axis
    prf = 1 / pri_s
    max_doppler_freq = prf / 2
    max_velocity = max_doppler_freq * wavelength / 2
    velocity_axis = np.linspace(-max_velocity, max_velocity, num_chirps)

    # Resolutions
    range_res = c / (2 * chirp_bw)
    velocity_res = wavelength / (2 * num_chirps * pri_s)

    return {
        'range_axis': range_axis,
        'velocity_axis': velocity_axis,
        'range_res': range_res,
        'velocity_res': velocity_res,
        'max_range': range_axis.max(),
        'max_velocity': max_velocity,
        'slope': slope,
        'wavelength': wavelength,
    }
