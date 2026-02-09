"""
tracker.py — Multi-target Kalman filter tracker.

Associates CFAR detections across frames into persistent tracks.
Each track maintains a state estimate [range, velocity, azimuth]
smoothed by a linear Kalman filter. Tracks are initiated, updated,
and deleted based on detection-to-track association.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Track:
    """A single target track."""
    track_id: int
    state: np.ndarray          # [range, velocity, azimuth]
    covariance: np.ndarray     # 3x3 covariance matrix
    hits: int = 1              # number of frames with a matched detection
    misses: int = 0            # consecutive frames without a detection
    age: int = 1               # total number of frames since initiation
    confirmed: bool = False    # True after M-of-N hits
    history: list = field(default_factory=list)  # list of past states for trail display


class KalmanTracker:
    """Multi-target tracker using linear Kalman filter.

    State vector: [range (m), velocity (m/s), azimuth (deg)]
    Measurement: [range (m), velocity (m/s), azimuth (deg)]

    Process model: constant velocity in range, constant velocity and azimuth.
        range(k+1) = range(k) + velocity(k) * dt
        velocity(k+1) = velocity(k)
        azimuth(k+1) = azimuth(k)
    """

    def __init__(self, dt=0.1, confirm_m=3, confirm_n=5, max_misses=5,
                 gate_distance=10.0,
                 process_noise_range=1.0, process_noise_vel=0.5, process_noise_az=5.0,
                 meas_noise_range=0.5, meas_noise_vel=0.3, meas_noise_az=10.0,
                 max_tracks=20, history_length=50):
        """
        Args:
            dt: Time between frames (seconds). Estimated from PRI * num_chirps.
            confirm_m: Minimum hits within confirm_n frames to confirm a track.
            confirm_n: Window of frames for M-of-N confirmation logic.
            max_misses: Delete track after this many consecutive misses.
            gate_distance: Mahalanobis distance threshold for association gating.
            process_noise_range: Process noise std dev for range (m).
            process_noise_vel: Process noise std dev for velocity (m/s).
            process_noise_az: Process noise std dev for azimuth (deg).
            meas_noise_range: Measurement noise std dev for range (m).
            meas_noise_vel: Measurement noise std dev for velocity (m/s).
            meas_noise_az: Measurement noise std dev for azimuth (deg).
            max_tracks: Maximum number of simultaneous tracks.
            history_length: Number of past states to keep for trail display.
        """
        self.dt = dt
        self.confirm_m = confirm_m
        self.confirm_n = confirm_n
        self.max_misses = max_misses
        self.gate_distance = gate_distance
        self.max_tracks = max_tracks
        self.history_length = history_length

        # State transition matrix F
        self.F = np.array([
            [1, dt, 0],   # range += velocity * dt
            [0, 1,  0],   # velocity constant
            [0, 0,  1],   # azimuth constant
        ], dtype=float)

        # Measurement matrix H (we directly observe all three states)
        self.H = np.eye(3, dtype=float)

        # Process noise covariance Q
        self.Q = np.diag([
            process_noise_range**2,
            process_noise_vel**2,
            process_noise_az**2,
        ])

        # Measurement noise covariance R
        self.R = np.diag([
            meas_noise_range**2,
            meas_noise_vel**2,
            meas_noise_az**2,
        ])

        self.tracks: List[Track] = []
        self._next_id = 1

    def predict(self):
        """Predict all tracks forward by one time step."""
        for track in self.tracks:
            track.state = self.F @ track.state
            track.covariance = self.F @ track.covariance @ self.F.T + self.Q

    def _mahalanobis(self, track, measurement):
        """Compute Mahalanobis distance between a track prediction and measurement."""
        innovation = measurement - self.H @ track.state
        S = self.H @ track.covariance @ self.H.T + self.R
        try:
            S_inv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            return float('inf')
        d = np.sqrt(innovation.T @ S_inv @ innovation)
        return float(d)

    def associate_and_update(self, detections):
        """Associate detections to tracks and update.

        Uses greedy nearest-neighbor association with Mahalanobis gating.

        Args:
            detections: List of target dicts, each with 'range_m', 'velocity_mps',
                       and optionally 'angle_deg'.

        Returns:
            List of (track, detection) pairs that were matched.
        """
        if not detections and not self.tracks:
            return []

        # Convert detections to measurement vectors
        measurements = []
        for det in detections:
            meas = np.array([
                det['range_m'],
                det['velocity_mps'],
                det.get('angle_deg', 0.0),
            ])
            measurements.append(meas)

        # Compute cost matrix (Mahalanobis distance)
        num_tracks = len(self.tracks)
        num_meas = len(measurements)

        if num_tracks == 0 and num_meas == 0:
            return []

        matched_pairs = []
        used_tracks = set()
        used_meas = set()

        if num_tracks > 0 and num_meas > 0:
            cost = np.full((num_tracks, num_meas), float('inf'))
            for ti, track in enumerate(self.tracks):
                for mi, meas in enumerate(measurements):
                    cost[ti, mi] = self._mahalanobis(track, meas)

            # Greedy nearest-neighbor (simple and effective for sparse scenarios)
            while True:
                if cost.size == 0:
                    break
                min_idx = np.unravel_index(np.argmin(cost), cost.shape)
                min_val = cost[min_idx]
                if min_val > self.gate_distance:
                    break

                ti, mi = min_idx
                if ti in used_tracks or mi in used_meas:
                    cost[ti, mi] = float('inf')
                    continue

                # Update track with measurement
                self._kalman_update(self.tracks[ti], measurements[mi])
                self.tracks[ti].hits += 1
                self.tracks[ti].misses = 0
                matched_pairs.append((self.tracks[ti], detections[mi]))

                used_tracks.add(ti)
                used_meas.add(mi)
                cost[ti, :] = float('inf')
                cost[:, mi] = float('inf')

        # Handle unmatched tracks (coast)
        for ti, track in enumerate(self.tracks):
            if ti not in used_tracks:
                track.misses += 1

        # Handle unmatched measurements (initiate new tracks)
        for mi, meas in enumerate(measurements):
            if mi not in used_meas and len(self.tracks) < self.max_tracks:
                self._initiate_track(measurements[mi])

        # Age all tracks and update history
        for track in self.tracks:
            track.age += 1
            track.history.append(track.state.copy())
            if len(track.history) > self.history_length:
                track.history.pop(0)

            # M-of-N confirmation
            if not track.confirmed and track.hits >= self.confirm_m:
                track.confirmed = True

        # Delete tracks that have exceeded max misses
        self.tracks = [t for t in self.tracks if t.misses <= self.max_misses]

        return matched_pairs

    def _kalman_update(self, track, measurement):
        """Apply Kalman filter measurement update to a track."""
        innovation = measurement - self.H @ track.state
        S = self.H @ track.covariance @ self.H.T + self.R
        try:
            K = track.covariance @ self.H.T @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            return  # Skip update if S is singular

        track.state = track.state + K @ innovation
        track.covariance = (np.eye(3) - K @ self.H) @ track.covariance

    def _initiate_track(self, measurement):
        """Create a new track from an unassociated measurement."""
        track = Track(
            track_id=self._next_id,
            state=measurement.copy(),
            covariance=np.diag([
                self.R[0, 0] * 4,  # Initial uncertainty: 4x measurement noise
                self.R[1, 1] * 4,
                self.R[2, 2] * 4,
            ]),
        )
        self._next_id += 1
        self.tracks.append(track)

    def update(self, detections):
        """Full predict-associate-update cycle.

        Call this once per frame with the current frame's CFAR detections.

        Args:
            detections: List of target dicts from cfar_2d.cluster_detections().

        Returns:
            List of confirmed tracks (for display).
        """
        self.predict()
        self.associate_and_update(detections)
        return self.get_confirmed_tracks()

    def get_confirmed_tracks(self):
        """Return all confirmed tracks."""
        return [t for t in self.tracks if t.confirmed]

    def get_all_tracks(self):
        """Return all tracks (including tentative)."""
        return list(self.tracks)
