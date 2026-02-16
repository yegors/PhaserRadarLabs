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
        self.sample_rate = 4e6  # ADC sampling rate in samples per second.
        self.center_freq = 2.1e9  # RF center frequency for transceiver tuning (Hz).
        self.signal_freq = 100e3  # IF/baseband tone used in FMCW processing (Hz).
        self.rx_gain = 60  # Receiver gain level.
        self.tx_gain = 0  # Transmit gain level.
        self.output_freq = 9.9e9  # Effective radar output carrier frequency (Hz).
        self.chirp_BW = 500e6  # Chirp sweep bandwidth (Hz).
        self.ramp_time = 300  # Nominal chirp ramp duration (us).
        self.num_chirps = 64  # Number of chirps per frame/CPI.
        self.max_range = 20.0  # Maximum displayed/processed range (m).
        self.min_range = 1.0  # Minimum usable range to suppress near-field leakage (m).

        # ── Processing ──
        self.mti_mode = '3pulse'  # MTI filter mode selection.
        self.mti_highpass_cutoff = 0.05  # High-pass cutoff for MTI filtering.
        self.range_window = 'hann'  # Window function applied before range FFT.
        self.doppler_window = 'hann'  # Window function applied before Doppler FFT.

        # ── CFAR ──
        self.cfar_enabled = True  # Enables/disables CFAR detection.
        self.cfar_guard_range = 3  # Number of guard cells in range dimension.
        self.cfar_guard_doppler = 3  # Number of guard cells in Doppler dimension.
        self.cfar_ref_range = 6  # Number of reference cells in range dimension.
        self.cfar_ref_doppler = 6  # Number of reference cells in Doppler dimension.
        self.cfar_bias_db = 15.0  # Detection threshold offset above noise estimate (dB).
        self.cfar_method = 'average'  # CFAR noise estimation method.
        self.cfar_min_cluster = 3  # Minimum cluster size to accept as a detection.

        # ── Tracking ──
        self.tracking_enabled = True  # Enables/disables multi-target tracking.
        self.track_confirm_m = 3  # Required hits for track confirmation (M-of-N).
        self.track_confirm_n = 5  # Confirmation window length for M-of-N logic.
        self.track_max_misses = 3  # Max consecutive misses before deleting a track.
        self.track_gate_distance = 5.0  # Association gate radius for matching detections (m).
        self.min_angle_confidence = 0.0  # Minimum monopulse confidence used for tracker updates.

        # ── Calibration Policy ──
        # Channel calibration is generally robust and useful; per-element gain/phase
        # calibration can be environment-sensitive (multipath) and may hurt runtime
        # performance if generated from poor calibration conditions.
        self.use_channel_calibration = True
        self.use_gain_phase_calibration = True

        # ── Display ──
        self.display_range_db = 30.0  # Dynamic display range for intensity visualization (dB).
        self.save_data = False  # Enables/disables raw/processed data logging.
        self.save_file = "enhanced_radar_data.npy"  # Output file path for saved radar data.
        self.det_log_path = "detections.json"  # Output path for detection log JSON.
        self.trk_log_path = "tracks.json"  # Output path for track log JSON.

        # ── Derived values (set by recompute_derived) ──
        self.ramp_time_s = 0.0  # Actual chirp ramp duration in seconds.
        self.PRI_ms = 0.0  # Pulse repetition interval in milliseconds.
        self.PRI_s = 0.0  # Pulse repetition interval in seconds.
        self.N_frame = 0  # Number of ADC samples in one PRI/frame.
        self.good_ramp_samples = 0  # Usable ramp samples after front-end offset removal.
        self.start_offset_samples = 0  # Sample index where valid ramp data starts.
        self.fft_size = 0  # Range FFT length (power-of-two).
        self.buffer_size = 0  # RX buffer size chosen to hold full acquisition.
        self.axes = {}  # Precomputed range/Doppler axis vectors.
        self.range_crop_idx = np.array([], dtype=int)  # Indices for selected in-range bins.
        self.range_axis_cropped = np.array([])  # Cropped range axis used for display/processing.
        self.frame_time = 0.0  # Total CPI/frame duration in seconds.

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

    def update_range_crop(self):
        """Recompute range crop for current max_range (lightweight, no HW touch)."""
        range_crop_mask = (
            (self.axes['range_axis'] >= 0) &
            (self.axes['range_axis'] <= self.max_range)
        )
        self.range_crop_idx = np.where(range_crop_mask)[0]
        self.range_axis_cropped = self.axes['range_axis'][self.range_crop_idx]
