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

    # Crop to [0, max_range]
    rd_db_crop = rd_db_sum[:, cfg.range_crop_idx]
    rd_complex_sum_crop = rd_complex_sum[:, cfg.range_crop_idx]

    # Velocity axis sized to actual Doppler dimension
    n_doppler = rd_db_crop.shape[0]
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

        # Cluster detections into targets
        all_clusters = cluster_detections(
            detections_mask, rd_db_crop,
            cfg.range_axis_cropped, vel_axis,
        )

        # Filter near-zero ghosts and small noise clusters
        targets = [t for t in all_clusters
                   if t['range_m'] >= cfg.min_range and t['pixel_count'] >= cfg.cfar_min_cluster]

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
            confirmed_tracks = hw.tracker.update(targets)
    else:
        cfar_cell_count = 0
        all_clusters = []
        rd_display = np.clip(rd_db_crop, 0, 100).T

    diag = {'cfar_cells': cfar_cell_count, 'raw_clusters': len(all_clusters)}
    return rd_display, targets, confirmed_tracks, diag
