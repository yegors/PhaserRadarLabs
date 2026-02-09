"""
config.py — Radar system configuration.

All base parameters and derived timing/axis values in one place.
Replaces the 40+ module-level globals from the monolithic script.
"""

import numpy as np
from lib.radar_processor import compute_axes


class RadarConfig:
    """Mutable configuration object shared across all modules."""

    def __init__(self):
        # ── RF Parameters ──
        self.sample_rate = 4e6
        self.center_freq = 2.1e9
        self.signal_freq = 100e3
        self.rx_gain = 60
        self.tx_gain = 0
        self.output_freq = 9.9e9
        self.chirp_BW = 500e6
        self.ramp_time = 300          # us
        self.num_chirps = 64
        self.max_range = 20.0         # m (display limit)
        self.min_range = 1.0          # m (TX leakage filter)

        # ── Processing ──
        self.mti_mode = '3pulse'
        self.mti_highpass_cutoff = 0.05
        self.range_window = 'hann'
        self.doppler_window = 'hann'

        # ── CFAR ──
        self.cfar_enabled = True
        self.cfar_guard_range = 3
        self.cfar_guard_doppler = 3
        self.cfar_ref_range = 6
        self.cfar_ref_doppler = 6
        self.cfar_bias_db = 20.0
        self.cfar_method = 'average'
        self.cfar_min_cluster = 10

        # ── Tracking ──
        self.tracking_enabled = True
        self.track_confirm_m = 3
        self.track_confirm_n = 5
        self.track_max_misses = 5
        self.track_gate_distance = 10.0

        # ── Display ──
        self.display_range_db = 30.0
        self.save_data = False
        self.save_file = "enhanced_radar_data.npy"
        self.log_file_path = "radar_log.jsonl"

        # ── Derived values (set by recompute_derived) ──
        self.ramp_time_s = 0.0
        self.PRI_ms = 0.0
        self.PRI_s = 0.0
        self.N_frame = 0
        self.good_ramp_samples = 0
        self.start_offset_samples = 0
        self.fft_size = 0
        self.buffer_size = 0
        self.axes = {}
        self.range_crop_idx = np.array([], dtype=int)
        self.range_axis_cropped = np.array([])
        self.frame_time = 0.0

    def recompute_derived(self, ramp_time_actual_us, tdd_frame_length_ms, tdd_ch0_on_ms):
        """Recompute all derived timing/axis values from base parameters.

        Args:
            ramp_time_actual_us: Actual ramp time from hardware (us).
            tdd_frame_length_ms: TDD frame length (ms) — PRI.
            tdd_ch0_on_ms: TDD channel 0 on time (ms).
        """
        self.ramp_time_s = ramp_time_actual_us / 1e6
        self.PRI_ms = tdd_frame_length_ms
        self.PRI_s = self.PRI_ms / 1e3
        self.N_frame = int(self.PRI_s * self.sample_rate)

        begin_offset_time = 0.1 * self.ramp_time_s
        self.good_ramp_samples = int((self.ramp_time_s - begin_offset_time) * self.sample_rate)

        start_offset_time = tdd_ch0_on_ms / 1e3 + begin_offset_time
        self.start_offset_samples = int(start_offset_time * self.sample_rate)

        # FFT size for range
        power = 8
        self.fft_size = int(2**power)
        num_samples_frame = int(tdd_frame_length_ms / 1000 * self.sample_rate)
        while num_samples_frame > self.fft_size:
            power += 1
            self.fft_size = int(2**power)
            if power == 18:
                break

        # Buffer sizing
        total_time = tdd_frame_length_ms * self.num_chirps
        power = 12
        self.buffer_size = 0
        buffer_time = 0
        while total_time > buffer_time:
            power += 1
            self.buffer_size = int(2**power)
            buffer_time = self.buffer_size / self.sample_rate * 1000
            if power == 23:
                break

        # Axes
        self.axes = compute_axes(
            sample_rate=self.sample_rate,
            output_freq=self.output_freq,
            signal_freq=self.signal_freq,
            chirp_bw=self.chirp_BW,
            ramp_time_s=self.ramp_time_s,
            num_chirps=self.num_chirps,
            pri_s=self.PRI_s,
            samples_per_frame=self.good_ramp_samples,
        )

        # Range crop indices
        range_crop_mask = (
            (self.axes['range_axis'] >= 0) &
            (self.axes['range_axis'] <= self.max_range)
        )
        self.range_crop_idx = np.where(range_crop_mask)[0]
        self.range_axis_cropped = self.axes['range_axis'][self.range_crop_idx]

        # Frame time
        self.frame_time = total_time / 1000  # seconds
