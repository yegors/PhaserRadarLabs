"""
angle_estimator.py — Monopulse azimuth estimation using sum/difference channels.

The Phaser has 8 antenna elements but only 2 ADC channels (Rx0, Rx1).
The ADAR1000 beamformers sum elements 1-4 into Rx0 and elements 5-8 into Rx1.
By forming sum (Σ = Rx0 + Rx1) and difference (Δ = Rx0 - Rx1) beams,
we can estimate the azimuth angle of detected targets via the monopulse ratio.

This is a two-element interferometer operating on the sub-array centers,
with an effective baseline of ~3.5 * element_spacing = 49 mm.
"""

import numpy as np


class MonopulseEstimator:
    """Estimates target azimuth from sum and difference channel data."""

    def __init__(self, element_spacing=0.014, num_elements=8, output_freq=9.9e9):
        """
        Args:
            element_spacing: Spacing between individual elements (m). Default 14 mm.
            num_elements: Total number of elements in the array.
            output_freq: Operating frequency (Hz) at the antenna.
        """
        self.element_spacing = element_spacing
        self.num_elements = num_elements
        self.output_freq = output_freq

        c = 3e8
        self.wavelength = c / output_freq

        # Effective baseline between the two sub-array phase centers
        # Sub-array 1: elements 0-3, center at index 1.5
        # Sub-array 2: elements 4-7, center at index 5.5
        # Separation = (5.5 - 1.5) * element_spacing = 4 * d
        self.baseline = 4 * element_spacing

        # Monopulse slope (sensitivity) at boresight
        # For two-element interferometer: K = λ / (π * d * cos(θ))
        # At boresight (θ=0): K = λ / (π * d)
        self.K_boresight = self.wavelength / (np.pi * self.baseline)

    def form_sum_diff(self, chan0, chan1):
        """Form sum and difference channels from the two Rx inputs.

        Args:
            chan0: Complex IQ data from Rx channel 0 (sub-array 1).
            chan1: Complex IQ data from Rx channel 1 (sub-array 2).

        Returns:
            sum_channel: Σ = chan0 + chan1
            diff_channel: Δ = chan0 - chan1
        """
        return chan0 + chan1, chan0 - chan1

    def estimate_angle(self, sum_data, diff_data):
        """Estimate azimuth angle from monopulse ratio.

        The monopulse ratio Δ/Σ is purely real for a single point target
        at a given angle. The real part encodes the angle; the imaginary
        part is ideally zero (non-zero indicates multipath, noise, or
        multiple targets in the same cell).

        Args:
            sum_data: Complex sum-channel value(s). Can be scalar or array.
            diff_data: Complex difference-channel value(s). Same shape.

        Returns:
            angle_deg: Estimated azimuth angle(s) in degrees. Positive = right.
            confidence: Quality metric (0 to 1). Based on how purely real the
                        monopulse ratio is. Low confidence means unreliable estimate.
        """
        # Avoid division by zero
        sum_mag = np.abs(sum_data)
        valid = sum_mag > 1e-10

        ratio = np.where(valid, diff_data / sum_data, 0 + 0j)

        # The angle is encoded in the real part of the ratio
        # sin(θ) = (λ / (π * d)) * Re(Δ/Σ)  for small angles
        sin_theta = self.K_boresight * np.real(ratio)

        # Clip to valid range
        sin_theta = np.clip(sin_theta, -1, 1)
        angle_rad = np.arcsin(sin_theta)
        angle_deg = np.degrees(angle_rad)

        # Confidence: ratio of real to total magnitude of the monopulse ratio
        # A pure single target gives a purely real ratio → confidence ≈ 1
        ratio_mag = np.abs(ratio)
        confidence = np.where(
            ratio_mag > 1e-10,
            np.abs(np.real(ratio)) / ratio_mag,
            0.0
        )

        return angle_deg, confidence

    def estimate_from_rd_cell(self, sum_rd, diff_rd, range_idx, doppler_idx):
        """Estimate angle for a specific cell in the range-Doppler map.

        Args:
            sum_rd: Complex 2D FFT of sum channel (doppler, range).
            diff_rd: Complex 2D FFT of difference channel (doppler, range).
            range_idx: Range bin index (float — will be rounded).
            doppler_idx: Doppler bin index (float — will be rounded).

        Returns:
            angle_deg: Estimated azimuth in degrees.
            confidence: Quality metric (0 to 1).
        """
        ri = int(round(range_idx))
        di = int(round(doppler_idx))

        # Clamp to valid indices
        ri = max(0, min(ri, sum_rd.shape[1] - 1))
        di = max(0, min(di, sum_rd.shape[0] - 1))

        return self.estimate_angle(sum_rd[di, ri], diff_rd[di, ri])

    def estimate_targets(self, sum_rd, diff_rd, targets):
        """Add angle estimates to a list of CFAR-detected targets.

        Args:
            sum_rd: Complex 2D FFT of sum channel.
            diff_rd: Complex 2D FFT of difference channel.
            targets: List of target dicts from cfar_2d.cluster_detections().
                     Each must have 'range_idx' and 'doppler_idx'.

        Returns:
            Same list with 'angle_deg' and 'angle_confidence' added to each target.
        """
        for t in targets:
            angle, conf = self.estimate_from_rd_cell(
                sum_rd, diff_rd, t['range_idx'], t['doppler_idx']
            )
            t['angle_deg'] = float(angle)
            t['angle_confidence'] = float(conf)
        return targets
