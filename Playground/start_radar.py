"""
start_radar.py — FMCW Range-Doppler Radar System
=================================================

This script launches a real-time FMCW (Frequency-Modulated Continuous Wave) radar
using the Analog Devices CN0566 Phaser development platform.

Hardware:
  - CN0566 Phaser board: 8-element linear patch antenna array (10.0–10.5 GHz)
  - Two ADAR1000 beamformers: BEAM0 (elements 5-8), BEAM1 (elements 1-4)
  - ADF4159 PLL + HMC735 VCO: chirp generator (sawtooth frequency ramp)
  - PlutoSDR (AD9361): digitizes the IF signal (~2.1 GHz) at 4 MSPS
  - TDD engine: synchronizes chirp bursts with data capture

Signal Flow:
  TX waveform (100 kHz tone) --> PlutoSDR DAC --> CN0566 upconverter --> antenna
  --> target reflection --> antenna array --> CN0566 downconverter --> PlutoSDR ADC
  --> IQ samples --> this script

Processing Pipeline (each frame):
  1. CAPTURE       Trigger a burst of N chirps via GPIO, read IQ from DMA buffer
  2. EXTRACT       Slice the continuous IQ stream into individual chirp segments
  3. BEAMFORM      Sum beam (ch0+ch1) for detection, Diff beam (ch0-ch1) for angle
  4. MTI FILTER    Cancel stationary clutter (ground, walls) via pulse cancellation
  5. WINDOW        Apply Hann window in range and Doppler to reduce spectral leakage
  6. 2D FFT        Range FFT (fast-time) + Doppler FFT (slow-time) = range-Doppler map
  7. CFAR DETECT   Cell-Averaging CFAR: adaptive threshold above local noise floor
  8. CLUSTER       Group adjacent detection cells into discrete target reports
  9. MONOPULSE     Estimate azimuth angle from sum/diff beam ratio (if targets found)
  10. TRACK        Kalman filter: predict-update cycle with M-of-N confirmation

Display:
  - Range-Doppler heatmap (velocity x range), CFAR detection markers, track markers
  - Trail lines showing track history, info panel with detection/track details
  - GUI controls: chirps, CFAR bias, min cluster, min range, display dB, persist mode

Run from Playground/ directory:
  python start_radar.py
"""

import signal
import pyqtgraph as pg

# =============================================================================
# STAGE 0: System Setup
# =============================================================================
# Allow Ctrl+C handling to be installed after GUI creation so we can request
# a graceful Qt shutdown (instead of interrupting NumPy/Qt internals).

# Set global PyQtGraph options BEFORE any widget is created.
# Dark theme (GitHub-dark inspired), no antialiasing for speed on Raspberry Pi.
pg.setConfigOptions(antialias=False, background='#0d1117', foreground='#c9d1d9')

# =============================================================================
# STAGE 1: Configuration
# =============================================================================
# RadarConfig holds every tunable parameter and all derived timing values.
#
# Key base parameters:
#   sample_rate  = 4 MHz     PlutoSDR ADC/DAC sample rate
#   output_freq  = 9.9 GHz   Chirp start frequency (VCO output)
#   chirp_BW     = 500 MHz   Chirp bandwidth --> range resolution ~ 0.3 m
#   ramp_time    = 300 us    Single chirp duration
#   num_chirps   = 64        Chirps per burst --> Doppler resolution
#   center_freq  = 2.1 GHz   IF frequency (LO - RF)
#
# Derived values (computed after hardware init tells us actual timing):
#   PRI                  Pulse repetition interval (ramp_time + 200us overhead)
#   N_frame              Samples per PRI at 4 MSPS
#   good_ramp_samples    Usable samples per chirp (excluding 10% ramp start transient)
#   range/velocity axes  Physical units (m, m/s) for each FFT bin
#   buffer_size          DMA buffer large enough for num_chirps x N_frame samples
from lib.config import RadarConfig

cfg = RadarConfig()

# =============================================================================
# STAGE 2: Hardware Initialization
# =============================================================================
# init_hardware() connects to and configures all ADI devices, then computes
# the derived timing values that depend on actual hardware response.
#
# Step-by-step:
#
#   1. Auto-detect environment (local on Pi vs remote over network)
#      - Local:  SDR at 192.168.2.1, Phaser at localhost
#      - Remote: both at phaser.local (requires iiod)
#
#   2. Configure the 8-element phaser array (CN0566)
#      - Rx-only mode, 14mm element spacing
#      - All 8 elements at max gain (127), zero phase (boresight beam)
#      - Load factory gain/phase calibration from EEPROM
#
#   3. Configure PlutoSDR (AD9361)
#      - Rx: 2 channels enabled (ch0 = BEAM0 elements 5-8, ch1 = BEAM1 elements 1-4)
#        Manual gain control at 60 dB, no AGC (stable amplitude for monopulse)
#      - Tx: cyclic buffer with 100 kHz IF tone that feeds the CN0566 mixer
#        Ch0 muted (-88 dB), Ch1 active (0 dB)
#
#   4. Program the ADF4159 chirp generator
#      - VCO frequency = output_freq + signal_freq + center_freq
#        (HMC735 VCO has /4 feedback divider, so PLL sees VCO_freq/4)
#      - Single sawtooth burst mode: 500 MHz sweep in 300 us
#      - Triggered by TDD engine GPIO (tx_trig_en = 1)
#      - delay_word = 4095 for max ramp-to-ramp settling
#
#   5. Configure TDD (Time-Division Duplex) engine
#      - Synchronizes chirp triggers with Rx data capture
#      - PRI = ramp_time/1000 + 0.2 ms (overhead for settling/readout)
#      - Burst count = num_chirps (e.g., 64 chirps per capture)
#      - Three channels enabled (Tx trigger, Rx enable, chirp gate)
#      - External sync mode for precise timing
#
#   6. Compute derived timing values (updates cfg in-place)
#      - Reads actual ramp time back from ADF4159 (may differ from requested)
#      - Computes N_frame, good_ramp_samples, buffer_size, FFT size
#      - Builds range and velocity axes with physical units
#      - Crops range axis to [0, max_range] meters
#
#   7. Create MonopulseEstimator for azimuth angle estimation
#      - Sub-array baseline: 4 x 14mm = 56mm between BEAM0/BEAM1 phase centers
#      - Center frequency: 10.15 GHz (output_freq + chirp_BW/2) --> lambda ~ 29.6mm
#      - d/lambda ~ 1.89 --> unambiguous FOV about +/-15 degrees
#      - Uses sum/diff beam ratio: angle = K * Re(diff/sum)
#
#   8. Create KalmanTracker for multi-target tracking
#      - State vector: [range, velocity, azimuth]
#      - Constant-velocity motion model
#      - M-of-N confirmation: 3 hits in 5 frames before track is "confirmed"
#      - Greedy nearest-neighbor association with 10m gate distance
#      - Max 5 consecutive misses before track is dropped
#
#   9. Upload TX waveform to PlutoSDR
#      - 100 kHz complex sinusoid (2^18 samples, cyclic playback)
#      - This feeds the CN0566 mixer to generate the transmitted signal
#
# Returns HardwareContext with: sdr, phaser, tdd, sdr_pins, monopulse, tracker
from lib.hardware import init_hardware

hw = init_hardware(cfg)

# =============================================================================
# STAGE 3: GUI & Real-Time Loop
# =============================================================================
# RadarGUI creates the display and runs the frame-by-frame processing loop.
#
# Display layout:
#   Left  (75%): Range-Doppler heatmap with overlay markers
#   Right (25%): Info panel (detection/track tables) + parameter controls
#
# Each frame (QTimer at 0ms interval = as-fast-as-possible):
#
#   a) READ CONTROLS
#      GUI parameter tree --> update cfg (chirps, CFAR bias, min cluster, etc.)
#      If chirps changed: reconfigure TDD burst count, recompute buffer/axes/tracker
#
#   b) CAPTURE (hardware.get_radar_data)
#      GPIO pulse triggers ADF4159 to fire N chirps in rapid succession.
#      PlutoSDR DMA fills buffer with interleaved ch0/ch1 IQ samples.
#      Returns two complex arrays: chan0 (BEAM0) and chan1 (BEAM1).
#
#   c) PROCESS FRAME (pipeline.process_frame) — the full radar pipeline:
#
#      [1] EXTRACT CHIRPS
#          The raw IQ is one continuous stream. We know each chirp starts at
#          PRI intervals. Slice out the "good" portion of each ramp (skip the
#          first 10% where the frequency hasn't settled). Result: a 2D matrix
#          of shape (num_chirps x good_ramp_samples) for each channel.
#
#      [2] BEAMFORM
#          Sum beam  = ch0 + ch1  (max gain at boresight, used for detection)
#          Diff beam = ch0 - ch1  (null at boresight, used for angle estimation)
#          This is amplitude-comparison monopulse using the two 4-element sub-arrays.
#
#      [3] MTI CLUTTER FILTER (sum channel only)
#          3-pulse canceller: H(z) = 1 - 2*z^-1 + z^-2
#          Subtracts the mean across slow-time (chirps), removing stationary
#          clutter (walls, ground, furniture) that has zero Doppler shift.
#          Costs 2 chirps (64 --> 62 output chirps).
#
#      [4] WINDOWING
#          Hann window applied in both dimensions:
#          - Range (fast-time): reduces range sidelobes from -13 dB to -31 dB
#          - Doppler (slow-time): reduces velocity sidelobes similarly
#          Trade-off: main lobe widens ~2x, but sidelobe suppression prevents
#          strong targets from masking weaker ones in adjacent bins.
#
#      [5] 2D FFT --> RANGE-DOPPLER MAP
#          - Range FFT along fast-time (each chirp): beat frequency --> range
#          - Doppler FFT along slow-time (across chirps): phase shift --> velocity
#          - Convert magnitude to dB: 20*log10(|FFT|)
#          - Crop to [0, max_range] meters (discard negative/aliased range bins)
#
#      [6] CA-CFAR DETECTION (Cell-Averaging Constant False Alarm Rate)
#          For each cell in the range-Doppler map:
#          - Define a rectangular annulus of reference cells around it
#          - Guard cells (3x3) prevent the target itself from raising the threshold
#          - Reference cells (6x6) estimate the local noise floor
#          - Threshold = mean(reference cells) + bias_db
#          - Cell is a detection if its power exceeds the threshold
#          Implemented as a single ndimage.convolve() for speed (~4300 cells).
#
#      [7] CLUSTERING
#          Connected-component labeling (8-connectivity) groups adjacent
#          detection cells into discrete targets. For each cluster:
#          - Power-weighted centroid gives sub-bin range and velocity
#          - Peak power reported in dB
#          - Clusters below min_range or min_cluster_size are filtered out
#
#      [8] MONOPULSE ANGLE (only if targets found -- saves ~40% CPU)
#          Process the diff channel through the same MTI/window/FFT pipeline.
#          At each target's range-Doppler bin:
#            angle = K_boresight * Re(diff_complex / sum_complex)
#          where K_boresight = lambda / (pi * baseline).
#          Gives azimuth estimate in degrees for each detection.
#
#      [9] KALMAN TRACKING
#          - Predict: propagate each track's state forward by one frame time
#          - Associate: match detections to tracks (nearest-neighbor, gated)
#          - Update: correct track state with matched detection (Kalman gain)
#          - Confirm: new tracks need M hits in N frames to become confirmed
#          - Coast: unmatched tracks increment miss counter
#          - Drop: tracks exceeding max_misses are deleted
#
#   d) UPDATE DISPLAY
#      - Heatmap: threshold-relative range-Doppler image (CFAR removes noise floor)
#      - Detection markers: bright red stars at each CFAR detection
#      - Track markers: cyan open circles at each confirmed track position
#      - Trail lines: track history paths (up to 30 tracks)
#      - Persist mode: keeps historical detections (faded dots) and dead tracks
#
#   e) UPDATE INFO PANEL
#      Text display showing frame count, FPS, detection table (range, velocity,
#      azimuth, power), and track table (ID, range, velocity, azimuth, age).
#
#   f) LOG
#      Every frame writes a JSON line to radar_log.jsonl with full detection
#      and track data for offline analysis.
#
# Keyboard shortcuts:
#   P  Toggle persist mode (show/hide historical detections and dead tracks)
#   R  Reset plot zoom to default range
from lib.gui import RadarGUI

gui = RadarGUI(cfg, hw)


def _handle_sigint(_signum, _frame):
  try:
    gui.win.close()
    gui.app.quit()
  except Exception:
    pass


signal.signal(signal.SIGINT, _handle_sigint)

# =============================================================================
# STAGE 4: Run & Cleanup
# =============================================================================
# gui.run() starts a QTimer(0) and enters the Qt event loop.
# The timer fires update() as fast as possible — typical rate on Raspberry Pi
# is ~3 FPS (limited by PlutoSDR DMA transfer + FFT computation).
#
# On exit (Ctrl+C or window close):
#   - Stop the QTimer (no more frames)
#   - Close the JSONL log file
#   - Destroy PlutoSDR TX buffer (stops transmitting the 100 kHz tone)
#   - Optionally save raw IQ data to .npy for offline replay
try:
    gui.run()
finally:
    gui.cleanup()
