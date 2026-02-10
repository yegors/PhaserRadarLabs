#!/usr/bin/env python3
"""
calibrate.py — Per-element gain and phase calibration for CN0566 Phaser.

Uses the onboard TX path as a CW signal source (no HB100 needed).
Measures and compensates for manufacturing variation in the ADAR1000
beamformer channels and antenna elements.

Calibration steps:
  1. Channel cal — compensates gain mismatch between the two ADAR1000
     sub-arrays and SDR Rx channels
  2. Gain cal — measures per-element gain variation and computes
     correction factors so all elements contribute equally
  3. Phase cal — sweeps adjacent element pairs to find their phase
     offsets and computes cumulative corrections for beam coherence

Results are saved to .pkl files that start_radar.py loads automatically
on next startup (via phaser.load_gain_cal() / load_phase_cal()).

Setup:
  - Point the array at open space or a flat wall >2m distance
  - Run from Playground/ directory:  python calibrate.py
"""

import socket
import time
import sys

import numpy as np
from scipy.signal import windows

import adi

# ── Configuration ──────────────────────────────────────────────────────
SIGNAL_FREQ = 10.25e9       # RF frequency for CW calibration (Hz)
RX_LO = int(2.1e9)          # SDR Rx/Tx LO frequency (Hz)
SAMPLE_RATE = 30_000_000    # 30 MSPS (wide bandwidth for clean peak)
BUFFER_SIZE = 1024           # Small buffer for fast measurements
RX_GAIN = 6                 # Low Rx gain to avoid saturation during cal
TX_GAIN = -3                # TX DDS gain (dB)
DDS_FREQ = 2_000_000        # DDS tone frequency (Hz)
NUM_AVERAGES = 4             # Averages per measurement
PHASE_STEP = 2.8125          # ADAR1000 phase step resolution (degrees)
SETTLE_TIME = 0.5            # Wait after ADAR1000 config change (s)
PEAK_WIDTH = 10              # FFT bins around peak to search


def connect():
    """Connect to Phaser and PlutoSDR, auto-detecting local vs remote."""
    if "phaser" in socket.gethostname():
        rpi_ip = "ip:localhost"
        sdr_ip = "ip:192.168.2.1"
    else:
        rpi_ip = "ip:phaser.local"
        sdr_ip = "ip:phaser.local:50901"

    print(f"  SDR:    {sdr_ip}")
    print(f"  Phaser: {rpi_ip}")
    sdr = adi.ad9361(uri=sdr_ip)
    phaser = adi.CN0566(uri=rpi_ip, sdr=sdr)
    print(f"  pyadi-iio: {adi.__version__}")
    return sdr, phaser


def setup_cw(sdr, phaser):
    """Configure hardware for CW calibration mode (no chirp)."""
    phaser.configure(device_mode="rx")
    phaser.element_spacing = 0.014

    # SDR Rx
    sdr.sample_rate = SAMPLE_RATE
    sdr.rx_lo = RX_LO
    sdr.rx_enabled_channels = [0, 1]
    sdr.rx_buffer_size = BUFFER_SIZE
    sdr.rx_rf_bandwidth = int(10e6)
    sdr.gain_control_mode_chan0 = 'manual'
    sdr.gain_control_mode_chan1 = 'manual'
    sdr.rx_hardwaregain_chan0 = RX_GAIN
    sdr.rx_hardwaregain_chan1 = RX_GAIN

    # ADF4159 PLL in CW mode (frequency ramp disabled)
    pll_freq = (int(SIGNAL_FREQ) + RX_LO) // 4
    phaser.frequency = pll_freq
    phaser.freq_dev_step = 5690
    phaser.freq_dev_range = 0
    phaser.freq_dev_time = 0
    phaser.powerdown = 0
    phaser.ramp_mode = "disabled"

    # SDR TX: DDS tone on channel 1
    sdr.tx_lo = RX_LO
    sdr.tx_enabled_channels = [0, 1]
    sdr.tx_hardwaregain_chan0 = -88
    sdr.tx_hardwaregain_chan1 = TX_GAIN
    sdr.dds_single_tone(DDS_FREQ, 0.9, 1)

    # All elements at max gain, boresight beam
    phaser.set_all_gain(127)
    phaser.set_beam_phase_diff(0.0)
    phaser.Averages = NUM_AVERAGES

    time.sleep(1.0)
    print("  CW mode configured\n")


def find_peak_bin(sdr, phaser):
    """Find the FFT bin with the strongest signal (all elements enabled)."""
    win = np.blackman(BUFFER_SIZE)
    phaser.set_all_gain(127)
    phaser.set_beam_phase_diff(0.0)
    time.sleep(0.2)
    data = sdr.rx()
    y = (data[0] + data[1]) * win
    spectrum = np.fft.fftshift(np.absolute(np.fft.fft(y)))
    peak = int(np.argmax(spectrum))
    noise = np.median(spectrum)
    snr = 20 * np.log10(spectrum[peak] / noise + 1e-20)
    return peak, snr


def measure_at_peak(sdr, peak_bin):
    """Measure signal amplitude at the peak bin (averaged, flattop window)."""
    win = windows.flattop(BUFFER_SIZE)
    win /= np.average(np.abs(win))
    total = 0
    for _ in range(NUM_AVERAGES):
        sdr.rx()  # flush stale buffer
        data = sdr.rx()
        y = (data[0] + data[1]) * win
        spectrum = np.fft.fftshift(np.absolute(np.fft.fft(y)))
        lo = max(0, peak_bin - PEAK_WIDTH)
        hi = min(BUFFER_SIZE, peak_bin + PEAK_WIDTH)
        total += np.max(spectrum[lo:hi])
    return total / (NUM_AVERAGES * BUFFER_SIZE)


def run_channel_cal(sdr, phaser, peak_bin):
    """Measure and compensate gain mismatch between the two sub-arrays."""
    print("Step 1/3: Channel Calibration")
    print("  Measuring sub-array gain mismatch...")

    win = windows.flattop(BUFFER_SIZE)
    win /= np.average(np.abs(win))

    channel_levels = []
    for ch in range(2):
        phaser.set_all_gain(0, apply_cal=False)
        # Enable all 4 elements of this sub-array
        # Note: (1 - ch) because of CN0566 channel mapping
        for el in range(4):
            phaser.set_chan_gain((1 - ch) * 4 + el, 127, apply_cal=False)
        time.sleep(SETTLE_TIME)

        total = 0
        for _ in range(NUM_AVERAGES):
            sdr.rx()
            data = sdr.rx()
            y = (data[0] + data[1]) * win
            spectrum = np.fft.fftshift(np.absolute(np.fft.fft(y)))
            lo = max(0, peak_bin - PEAK_WIDTH)
            hi = min(BUFFER_SIZE, peak_bin + PEAK_WIDTH)
            total += np.max(spectrum[lo:hi])
        level = total / (NUM_AVERAGES * BUFFER_SIZE)
        channel_levels.append(level)
        print(f"  Sub-array {ch} (elements {(1-ch)*4}-{(1-ch)*4+3}): {level:.6f}")

    mismatch_db = 20.0 * np.log10(channel_levels[0] / channel_levels[1])
    if mismatch_db > 0:
        phaser.ccal = [0.0, mismatch_db]
    else:
        phaser.ccal = [-mismatch_db, 0.0]

    print(f"  Mismatch: {mismatch_db:+.1f} dB")
    print(f"  ccal = [{phaser.ccal[0]:.1f}, {phaser.ccal[1]:.1f}]")

    if abs(mismatch_db) > 10:
        print("  WARNING: mismatch exceeds 10 dB — check antenna connections!")
    print()
    return mismatch_db


def run_gain_cal(sdr, phaser, peak_bin):
    """Measure per-element gain and compute correction factors."""
    print("Step 2/3: Gain Calibration")
    print("  Measuring per-element gain (one at a time)...")

    element_levels = []
    for el in range(8):
        phaser.set_all_gain(0, apply_cal=False)
        phaser.set_chan_gain(el, 127, apply_cal=False)
        time.sleep(SETTLE_TIME)

        level = measure_at_peak(sdr, peak_bin)
        element_levels.append(level)
        bar = "#" * max(1, int(level / max(element_levels) * 30)) if element_levels else ""
        print(f"  Element {el}: {level:.6f}  {bar}")

    # Normalize: weakest element is the reference (all factors <= 1.0)
    min_level = min(element_levels)
    max_level = max(element_levels)
    for k in range(8):
        phaser.gcal[k] = min_level / element_levels[k]

    spread_db = 20 * np.log10(max_level / min_level)
    print(f"  Gain spread: {spread_db:.1f} dB")
    print(f"  gcal = [{', '.join(f'{g:.3f}' for g in phaser.gcal)}]")

    weak = [i for i, g in enumerate(phaser.gcal) if g < 0.5]
    if weak:
        print(f"  WARNING: elements {weak} below 50% — may be damaged")
    print()
    return element_levels


def run_phase_cal(sdr, phaser, peak_bin):
    """Measure inter-element phase offsets by sweeping adjacent pairs."""
    print("Step 3/3: Phase Calibration")
    print("  Sweeping adjacent element pairs to find null...")

    win = windows.flattop(BUFFER_SIZE)
    win /= np.average(np.abs(win))
    sweep_phases = np.arange(-180, 180, PHASE_STEP)

    phaser.pcal = [0.0] * 8
    ph_deltas = []

    for pair in range(7):
        ref_el = pair
        cal_el = pair + 1

        # Enable only this pair (with gain cal applied)
        phaser.set_all_gain(0)
        phaser.set_chan_gain(ref_el, 127, apply_cal=True)
        phaser.set_chan_gain(cal_el, 127, apply_cal=True)
        time.sleep(SETTLE_TIME)

        phaser.set_chan_phase(ref_el, 0.0, apply_cal=False)

        # Sweep phase of cal element
        gains = []
        for phase in sweep_phases:
            phaser.set_chan_phase(cal_el, phase, apply_cal=False)
            total = 0
            for _ in range(NUM_AVERAGES):
                data = sdr.rx()
                data = sdr.rx()
                y = (data[0] + data[1]) * win
                spectrum = np.fft.fftshift(np.absolute(np.fft.fft(y)))
                total += np.max(spectrum)
            gains.append(total / (NUM_AVERAGES * BUFFER_SIZE))

        # Null = where two elements cancel (180 degrees out of phase)
        null_idx = np.argmin(gains)
        null_phase = sweep_phases[null_idx]
        ph_delta = (180 - null_phase) % 360
        if ph_delta > 180:
            ph_delta -= 360

        ph_deltas.append(ph_delta)

        # Cumulative correction
        phaser.pcal[cal_el] = (phaser.pcal[ref_el] - ph_delta) % 360
        if phaser.pcal[cal_el] > 180:
            phaser.pcal[cal_el] -= 360

        null_depth = 20 * np.log10(min(gains) / max(gains) + 1e-20)
        print(f"  Pair {ref_el}-{cal_el}: null at {null_phase:+6.1f} deg, "
              f"delta = {ph_delta:+6.1f} deg, depth = {null_depth:.0f} dB")

    print(f"  pcal = [{', '.join(f'{p:+.1f}' for p in phaser.pcal)}]")

    # Sanity check: adjacent element phase deltas should be in a reasonable range
    large = [i for i, d in enumerate(ph_deltas) if abs(d) > 120]
    if large:
        print(f"  WARNING: pairs {large} have phase delta > 120 deg")
    print()
    return ph_deltas


def main():
    print()
    print("=" * 60)
    print("  CN0566 Phaser — Antenna Array Calibration")
    print("=" * 60)
    print()
    print("Point the array at open space or a flat wall >2m away.")
    print("Ensure no strong moving reflectors in the beam.")
    print()

    # Connect
    print("Connecting to hardware...")
    sdr, phaser = connect()
    setup_cw(sdr, phaser)

    # Check signal quality
    print("Checking signal quality...")
    peak_bin, snr = find_peak_bin(sdr, phaser)
    print(f"  Peak bin: {peak_bin}/{BUFFER_SIZE}, SNR: {snr:.0f} dB")
    if snr < 20:
        print(f"  WARNING: SNR is only {snr:.0f} dB — calibration may be inaccurate.")
        print(f"  Check TX path and antenna connections.")
        resp = input("  Continue anyway? (y/n): ")
        if resp.lower() != 'y':
            sdr.tx_destroy_buffer()
            sys.exit(1)
    print()

    t0 = time.time()

    # Run calibrations in order: channel -> gain -> phase
    mismatch = run_channel_cal(sdr, phaser, peak_bin)
    levels = run_gain_cal(sdr, phaser, peak_bin)
    deltas = run_phase_cal(sdr, phaser, peak_bin)

    elapsed = time.time() - t0

    # Save calibration files
    print("Saving calibration files...")
    phaser.save_channel_cal()
    phaser.save_gain_cal()
    phaser.save_phase_cal()
    print("  channel_cal_val.pkl")
    print("  gain_cal_val.pkl")
    print("  phase_cal_val.pkl")

    # Summary
    min_level = min(levels)
    max_level = max(levels)
    spread_db = 20 * np.log10(max_level / min_level)

    print()
    print("=" * 60)
    print("  CALIBRATION COMPLETE")
    print("=" * 60)
    print(f"  Time:              {elapsed:.0f}s")
    print(f"  Signal SNR:        {snr:.0f} dB")
    print(f"  Channel mismatch:  {mismatch:+.1f} dB")
    print(f"  Gain spread:       {spread_db:.1f} dB (before correction)")
    print(f"  Gain corrections:  [{', '.join(f'{g:.3f}' for g in phaser.gcal)}]")
    print(f"  Phase corrections: [{', '.join(f'{p:+.1f}' for p in phaser.pcal)}] deg")
    print()
    print("Restart start_radar.py to use the new calibration values.")
    print()

    # Cleanup
    sdr.tx_destroy_buffer()


if __name__ == '__main__':
    main()
