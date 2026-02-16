"""
pipeline.py — Frame processing pipeline.

Extracts chirps, applies MTI filtering, windowing, 2D FFT,
CFAR detection, clustering, monopulse angle estimation, and tracking.
"""

import numpy as np

from lib.radar_processor import (
    extract_chirps, apply_window_2d, range_doppler_fft,
    mti_2pulse, mti_3pulse, mti_highpass,
)
from lib.cfar_2d import cfar_2d, cluster_detections


def _suppress_sidelobes(targets, range_tol=1.0, power_tol=15.0):
    """Remove targets that are velocity sidelobes of a stronger target.

    Sidelobes appear at the same range but different velocity, and are
    significantly weaker than the main target. Walk the list strongest-first;
    any weaker target within range_tol meters and power_tol dB below a
    stronger one is discarded.
    """
    if len(targets) <= 1:
        return targets
    # targets are already sorted by power (strongest first)
    keep = [True] * len(targets)
    for i in range(len(targets)):
        if not keep[i]:
            continue
        for j in range(i + 1, len(targets)):
            if not keep[j]:
                continue
            range_diff = abs(targets[i]['range_m'] - targets[j]['range_m'])
            power_diff = targets[i]['power_db'] - targets[j]['power_db']
            if range_diff < range_tol and power_diff > power_tol:
                keep[j] = False
    return [t for t, k in zip(targets, keep) if k]


def process_frame(chan0, chan1, cfg, hw):
    """Full processing pipeline for one frame.

    Processes only the sum channel through MTI/window/FFT.
    The diff channel (for monopulse angle) is only computed when
    CFAR finds real targets — saves ~40% CPU on quiet frames.

    Args:
        chan0: Complex IQ from Rx channel 0.
        chan1: Complex IQ from Rx channel 1.
        cfg: RadarConfig instance.
        hw: HardwareContext (for hw.monopulse, hw.tracker).

    Returns:
        rd_display: Range-Doppler map for display (threshold-relative).
        targets: List of detected targets with range, velocity, angle.
        confirmed_tracks: List of confirmed Kalman tracks.
        diag: Dict with diagnostic info (cfar_cells, raw_clusters).
    """
    # Extract chirps from each channel
    chirps_ch0 = extract_chirps(
        chan0, cfg.num_chirps, cfg.N_frame,
        cfg.good_ramp_samples, cfg.start_offset_samples,
    )
    chirps_ch1 = extract_chirps(
        chan1, cfg.num_chirps, cfg.N_frame,
        cfg.good_ramp_samples, cfg.start_offset_samples,
    )

    # Form sum and difference beams
    chirps_sum = chirps_ch0 + chirps_ch1
    chirps_diff = chirps_ch0 - chirps_ch1

    # MTI clutter filtering — sum channel only
    if cfg.mti_mode == '2pulse':
        chirps_sum_filt = mti_2pulse(chirps_sum)
    elif cfg.mti_mode == '3pulse':
        chirps_sum_filt = mti_3pulse(chirps_sum)
    elif cfg.mti_mode == 'highpass':
        chirps_sum_filt = mti_highpass(chirps_sum, cutoff_normalized=cfg.mti_highpass_cutoff)
    else:
        chirps_sum_filt = chirps_sum

    # Window + FFT — sum channel only
    chirps_sum_win = apply_window_2d(chirps_sum_filt, cfg.range_window, cfg.doppler_window)
    rd_db_sum, rd_complex_sum = range_doppler_fft(chirps_sum_win)

    # Zero-Doppler suppression: replace center 3 Doppler bins with the local
    # noise floor (per-range-bin median) to kill residual stationary clutter
    # that survives the MTI filter. Using the median instead of an extreme value
    # like -200 dB is critical — extreme values corrupt the CFAR reference cell
    # averages for all cells within ±(guard+ref) bins, creating false threshold
    # dips and phantom detection bands.
    center_dop = rd_db_sum.shape[0] // 2
    noise_fill = np.median(rd_db_sum, axis=0, keepdims=True)  # (1, n_range)
    rd_db_sum[center_dop - 1:center_dop + 2, :] = noise_fill
    rd_complex_sum[center_dop - 1:center_dop + 2, :] = 0

    # Crop to [0, max_range]
    rd_db_crop = rd_db_sum[:, cfg.range_crop_idx]
    rd_complex_sum_crop = rd_complex_sum[:, cfg.range_crop_idx]

    # Noise floor estimate (median of RD map after MTI)
    noise_floor_db = float(np.median(rd_db_crop))

    # Per-Doppler-bin normalization: flatten vertical band artifacts caused by
    # fixed-frequency spurs (clock/mixer harmonics) that appear as bright
    # columns spanning all range bins at symmetric Doppler frequencies.
    # The column median captures only noise+spur power (real targets occupy
    # only a few range cells and don't shift the median).
    doppler_median = np.median(rd_db_crop, axis=1, keepdims=True)
    rd_db_crop = rd_db_crop - doppler_median + np.median(doppler_median)

    # Velocity axis sized to actual Doppler dimension
    n_doppler = rd_db_crop.shape[0]
    if len(cfg.axes['velocity_axis']) == n_doppler:
        vel_axis = cfg.axes['velocity_axis']
    else:
        vel_axis = np.linspace(-cfg.axes['max_velocity'], cfg.axes['max_velocity'], n_doppler)

    targets = []
    confirmed_tracks = []

    if cfg.cfar_enabled:
        # 2D CFAR on cropped data only
        detections_mask, threshold = cfar_2d(
            rd_db_crop, cfg.cfar_guard_range, cfg.cfar_guard_doppler,
            cfg.cfar_ref_range, cfg.cfar_ref_doppler, cfg.cfar_bias_db, cfg.cfar_method,
        )

        cfar_cell_count = int(np.sum(detections_mask))

        # Threshold-relative display
        rd_above = rd_db_crop - threshold
        rd_display = np.clip(rd_above, 0, cfg.display_range_db).T

        # Zero out bins below min_range (TX leakage blob)
        min_range_mask = cfg.range_axis_cropped < cfg.min_range
        rd_display[min_range_mask, :] = 0

        # Cluster detections into targets
        all_clusters = cluster_detections(
            detections_mask, rd_db_crop,
            cfg.range_axis_cropped, vel_axis,
        )

        # Filter near-zero ghosts and small noise clusters
        targets = [t for t in all_clusters
                   if t['range_m'] >= cfg.min_range and t['pixel_count'] >= cfg.cfar_min_cluster]

        # Sidelobe blanking: remove weaker targets at similar range that are
        # likely Hann window sidelobes (-31 dB) of a stronger target.
        # Two real targets at the same range will have similar power and survive.
        targets = _suppress_sidelobes(targets)

        # Monopulse: only process diff channel if we have real detections
        if targets:
            if cfg.mti_mode == '2pulse':
                chirps_diff_filt = mti_2pulse(chirps_diff)
            elif cfg.mti_mode == '3pulse':
                chirps_diff_filt = mti_3pulse(chirps_diff)
            elif cfg.mti_mode == 'highpass':
                chirps_diff_filt = mti_highpass(chirps_diff, cutoff_normalized=cfg.mti_highpass_cutoff)
            else:
                chirps_diff_filt = chirps_diff

            chirps_diff_win = apply_window_2d(chirps_diff_filt, cfg.range_window, cfg.doppler_window)
            _, rd_complex_diff = range_doppler_fft(chirps_diff_win)
            rd_complex_diff_crop = rd_complex_diff[:, cfg.range_crop_idx]
            targets = hw.monopulse.estimate_targets(
                rd_complex_sum_crop, rd_complex_diff_crop, targets,
            )

        if cfg.tracking_enabled:
            tracker_targets = targets
            min_conf = getattr(cfg, 'min_angle_confidence', 0.0)
            if min_conf > 0:
                tracker_targets = [
                    t for t in targets
                    if t.get('angle_confidence', 1.0) >= min_conf
                ]
            confirmed_tracks = hw.tracker.update(tracker_targets)
    else:
        cfar_cell_count = 0
        all_clusters = []
        rd_display = np.clip(rd_db_crop, 0, 100).T
        min_range_mask = cfg.range_axis_cropped < cfg.min_range
        rd_display[min_range_mask, :] = 0

    diag = {'cfar_cells': cfar_cell_count, 'raw_clusters': len(all_clusters),
            'noise_floor_db': noise_floor_db}
    return rd_display, targets, confirmed_tracks, diag
