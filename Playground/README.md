# Range-Doppler Radar System

Real-time FMCW radar signal processing for the Analog Devices CN0566 Phaser platform. Captures chirp bursts via PlutoSDR, processes range-Doppler maps with automatic target detection, monopulse angle estimation, and multi-target tracking.

## Running

On the Phaser Pi (requires Pluto firmware v0.39+ for TDD engine):

```bash
cd Playground
python start_radar.py
```

Keyboard shortcuts: **P** = toggle persist, **R** = reset view, **Ctrl+C** = quit.

## Architecture

```
start_radar.py          Entry point (~20 lines)
lib/
  config.py             RadarConfig — all parameters + derived timing/axes
  hardware.py           ADI device init, burst capture, chirp reconfig, cleanup
  pipeline.py           process_frame() — MTI, FFT, CFAR, monopulse, tracking
  gui.py                RadarGUI — PyQtGraph display, update loop, info panel
  radar_processor.py    Core DSP: chirp extraction, windowing, 2D FFT, MTI filters
  cfar_2d.py            2D CA-CFAR detection + connected-component clustering
  angle_estimator.py    Monopulse azimuth estimation (sum/difference beams)
  tracker.py            Linear Kalman filter multi-target tracker
```

Data flows through a clean dependency chain with no circular imports:
`config` <- `hardware` <- `pipeline` <- `gui` <- `start_radar`

All shared state lives in a single mutable `RadarConfig` object — no global variables.

## GUI Controls

The right panel has a parameter tree with live-adjustable controls:

| Control | Range | Default | Effect |
|---|---|---|---|
| **Chirps** | 64 / 128 / 256 | 64 | Number of chirps per burst. More chirps = better velocity resolution (0.47 → 0.23 → 0.12 m/s) and +3/+6 dB Doppler integration gain, but slower update rate. Reconfigures TDD, buffer size, and tracker on change. |
| **CFAR Bias (dB)** | 5–30 | 20.0 | Detection threshold above estimated noise floor. Higher = fewer false alarms but misses weaker targets. Lower = more sensitive but noisier. |
| **Min Cluster** | 1–50 | 10 | Minimum number of connected CFAR cells to count as a detection. Filters out single-pixel noise spikes. Increase if seeing too many ghost detections. |
| **Min Range (m)** | 0–10 | 1.0 | Ignore detections closer than this. Filters TX leakage ghosts that appear at near-zero range. Increase if seeing persistent false targets at short range. |
| **Display Range (dB)** | 5–60 | 30 | Dynamic range of the heatmap colorscale. Maps 0 dB (at CFAR threshold) to full brightness. Lower values = more contrast on weak targets. Higher = see stronger returns without saturation. |
| **Persist** | on/off | on | Accumulate detection and track history across frames. When on: past detections show as faded red dots, stale tracks as dim open circles. When off: only current frame is displayed. |

**Keyboard shortcuts:** **P** = toggle persist, **R** = auto-range (reset zoom)

### Info Panel

The left sidebar shows real-time status in monospace columns:

- **STATUS** — frame count, FPS, MTI mode, CFAR cell/cluster counts, detection and track totals
- **DETECTIONS** — current-frame detections marked with `▸`, historical (persist) unmarked. Columns: Range, Velocity, Azimuth, Power
- **TRACKS** — confirmed Kalman tracks. Active tracks marked `●`, stale tracks `○`. Columns: ID, Range, Velocity, Azimuth, Age (seconds since first detection)

### Map Display

- **Heatmap**: threshold-relative range-Doppler map (inferno colormap). Black = at or below CFAR noise floor, bright = signal above threshold
- **Red stars**: current-frame CFAR detections
- **Cyan circles**: active confirmed tracks (filled = active `●`, open = stale `○`)
- **Cyan lines**: track trajectory trails (smoothed by Kalman filter)
- **Faded red dots**: historical detection positions (persist mode)

## What's Implemented

### Signal Processing
- **2D windowing** (Hann) on both range and Doppler dimensions to suppress spectral leakage
- **MTI clutter filtering** — selectable 2-pulse, 3-pulse (default), or Butterworth high-pass along slow-time
- **2D FFT** for range-Doppler map generation
- **Range cropping** to [0, max_range] before CFAR — reduces processing from ~1080 to ~67 range bins

### Detection
- **2D CA-CFAR** (Cell-Averaging Constant False Alarm Rate) with configurable guard/reference cells and bias
- **Connected-component clustering** to group adjacent CFAR cells into discrete targets
- **Threshold-relative display** — noise floor maps to black, only signal above CFAR threshold is visible
- **Near-range filtering** to reject TX leakage ghosts below configurable minimum range

### Angle Estimation
- **Monopulse** using sum (ch0+ch1) and difference (ch0-ch1) beams
- Effective baseline: 56 mm (4 x 14 mm element spacing between sub-array phase centers)
- Operating at chirp center frequency (10.15 GHz) for accurate wavelength
- Unambiguous FOV ~+/-15 degrees from boresight (d/lambda ~1.89)
- **Lazy evaluation** — diff channel MTI+FFT only computed when CFAR finds targets (~40% CPU savings on quiet frames)

### Tracking
- **Linear Kalman filter** with state [range, velocity, azimuth]
- Constant-velocity process model (range += velocity * dt)
- Greedy nearest-neighbor association with Mahalanobis distance gating
- **M-of-N confirmation** (3 hits in 5 frames) before track promotion
- Configurable coast/delete after N consecutive misses
- Track trajectory trails on the display

### Display (PyQtGraph)
- Range-Doppler heatmap with inferno colormap
- Detection markers (stars) and track markers (circles) with distinct active/stale styling
- Trajectory trail lines for tracked targets
- Info panel with aligned columns: detections table (range, velocity, azimuth, power) and tracks table (ID, range, velocity, azimuth, age)
- **Persist mode** — accumulates detection/track history across frames with visual distinction (filled vs open markers)
- **Live-configurable** chirp count (64/128/256), CFAR bias, min cluster size, min range, display dynamic range
- JSONL frame logging for offline analysis

## Hardware Reference

| Parameter | Value |
|---|---|
| Array | 8-element linear patch, 10.0-10.5 GHz |
| Element spacing | 14 mm |
| Beamformers | 2x ADAR1000 (elements 1-4 -> Rx1, elements 5-8 -> Rx0) |
| LO | ADF4159 PLL + HMC735 VCO, /4 divider |
| IF | ~2.1 GHz, digitized by PlutoSDR (AD9361) |
| Chirp BW | 500 MHz (range res ~0.3 m) |
| Ramp time | 300 us |
| Sample rate | 4 MSPS |
| Default chirps | 64 (velocity res ~0.47 m/s) |

## Next Steps — Improving Output Quality

### Signal Quality
- **Coherent integration gain**: Average multiple bursts before CFAR to improve SNR at the cost of update rate. Even 2-burst averaging gives ~3 dB improvement for slow targets.
- **Sidelobe suppression**: Replace Hann with Taylor or Dolph-Chebyshev windows for better sidelobe/mainlobe trade-off. Hann gives -31 dB sidelobes; Taylor can achieve -40 dB with narrower mainlobe widening.
- **DC offset removal**: Subtract per-chirp mean before FFT to eliminate the zero-Doppler DC spike that can leak through MTI filters on short bursts.
- **Phase noise compensation**: Estimate and correct range-dependent phase noise from the ADF4159 PLL using the TX leakage signal as a reference. This extends usable dynamic range at longer ranges.

### Detection Refinement
- **OS-CFAR** (Ordered Statistic): Replace cell-averaging with order-statistic CFAR for better performance in non-homogeneous clutter (e.g., near walls or at clutter edges). Uses the k-th sorted reference cell instead of the mean.
- **Interpolated peak finding**: Use parabolic or sinc interpolation on the FFT magnitude around each detection peak to get sub-bin range and velocity estimates. Currently limited to bin resolution (~0.3 m range, ~0.47 m/s velocity).
- **CFAR map caching**: Pre-compute the reference cell kernel as a convolution filter (uniform_filter) instead of per-cell sliding window — 5-10x faster for large maps.

### Angle Estimation
- **Beam steering for wide-angle targets**: When a track approaches the +/-15 degree monopulse ambiguity limit, steer the ADAR1000 phase shifters to re-center the beam on the target. The hardware supports 2.8 degree resolution and 20 ns switching.
- **Calibration-aware monopulse**: Load the CN0566's stored gain/phase calibration (.pkl files) to compensate sub-array mismatch. This corrects systematic angle bias, especially off-boresight.
- **Virtual array extension**: Use both TX outputs (TX1, TX2) in alternating bursts to create a 16-element virtual array. Doubles the aperture, halving the beamwidth and extending unambiguous monopulse FOV.

### Tracking
- **Extended Kalman Filter (EKF)**: The current linear KF assumes range changes linearly with velocity, which is exact. But adding azimuth rate (turning targets) requires a nonlinear state transition — EKF handles this naturally.
- **IMM (Interacting Multiple Model)**: Run parallel KFs with different motion models (constant velocity, constant acceleration, stationary) and blend their outputs. Better handles targets that stop, start, or turn.
- **Track smoothing / retrodiction**: After each update, run the Kalman filter backwards over the last N states (Rauch-Tung-Striebel smoother) to refine the full trajectory. Useful for post-processing logged data.
- **Probabilistic data association (JPDA)**: Replace greedy nearest-neighbor with joint probabilistic association for better handling of crossing targets and dense scenarios.

### Prediction and Extrapolation
- **Track-based gating**: Use the predicted track state to narrow the CFAR search area in the next frame — reduces false alarms and computation.
- **Coasting with prediction**: When a track loses detection (occlusion, fade), continue predicting its state forward using the last known velocity. Display predicted position as a distinct marker.
- **Collision / proximity alerts**: Compare predicted trajectories of multiple tracks to detect potential intersections or close approaches.

### System-Level
- **Automatic gain control**: Monitor peak power levels and adjust rx_gain / tx_gain to keep the signal within the ADC's linear range. Prevents saturation on close targets and improves sensitivity for distant ones.
- **Adaptive chirp parameters**: Increase num_chirps (better velocity resolution) when targets are slow, decrease (faster update rate) when targets are fast. Driven by tracker velocity estimates.
- **Multi-frame accumulation display**: Waterfall or history plot showing target range vs. time, complementing the instantaneous range-Doppler map.

## Planned: HMC451LP3 Power Amplifier Integration

Adding an [HMC451LP3](../docs/hmc451lp3.pdf) GaAs pHEMT MMIC PA in the TX path to increase transmit power and extend detection range.

### PA Specs (at 10 GHz)

| Parameter | Value |
|---|---|
| Gain | 18 dB (typ) |
| P1dB | 19.5 dBm |
| Psat | +21 dBm |
| OIP3 | +28 dBm |
| Noise figure | 7 dB |
| Supply | +5V @ 120 mA |
| Package | 3x3 mm QFN-16, 50 ohm matched I/O, DC-blocked RF |

### Expected Range Improvement

The radar range equation scales as R ~ P_tx^(1/4). Currently the PlutoSDR TX output is roughly 0 dBm (configurable via `tx_gain`). Adding 18 dB of PA gain:

- **TX power**: 0 dBm + 18 dB = +18 dBm (~63 mW vs ~1 mW)
- **Range factor**: 10^(18/40) = ~2.8x range increase
- If currently detecting targets reliably to ~15 m, expect ~40 m with the PA
- P1dB of 19.5 dBm means up to ~90 mW before compression — sufficient since the chirp waveform is CW (no PAPR concern)

### Hardware Integration

1. **Placement**: PA goes between PlutoSDR TX output and the CN0566 TX input (before the ADF4159 mixer). The HMC451LP3 RF I/Os are DC-blocked and 50 ohm matched — no external matching needed.

2. **Power supply**: +5V @ 120 mA, with bypass caps (100 pF + 1 nF + 2.2 uF) on both Vdd1 (pin 15) and Vdd2 (pin 13). Can tap the Pi's 5V rail or use a separate LDO for cleaner supply.

3. **Thermal**: 78 C/W thermal resistance (junction to ground pad). At 5V x 120 mA = 600 mW dissipation, junction rise is ~47 C above ground pad. Exposed paddle must be soldered to ground plane with adequate vias. No heatsink needed at this power level if ground plane is reasonable.

4. **ESD**: Class 1A (250V HBM) — handle with standard RF precautions.

### Software Changes Required

1. **`tx_gain` recalibration**: The PlutoSDR `tx_hardwaregain_chan1` (currently 0, range 0 to -88 dB) now drives the PA input. Need to verify we're not exceeding the PA's +10 dBm max input power at any tx_gain setting. At tx_gain=0 dBm output from Pluto, PA input is well within spec. May want to reduce tx_gain for close-range work to avoid receiver saturation.

2. **AGC / saturation protection**: With 18 dB more TX power, strong returns from close targets could saturate the AD9361 ADC. Options:
   - Reduce `rx_gain` (currently 60, max 70) by 10-18 dB to compensate
   - Implement software AGC: monitor peak levels and adjust rx_gain dynamically
   - Add `min_range` filtering (already exists at 1.0 m) to ignore saturated near-range bins

3. **CFAR threshold adjustment**: Higher TX power means higher SNR on real targets but also stronger clutter returns. The CFAR bias may need adjustment — likely increase `cfar_bias_db` from 20 to 22-25 dB to maintain similar false alarm rate.

4. **Range axis extension**: Currently `max_range = 20 m`. With the PA, increase to 40-50 m. This means more range bins survive the crop, slightly increasing CFAR computation time (still small — ~200 bins instead of ~67).

5. **Display range**: `plot_w.setYRange(0, max_range)` needs to match the new max_range. Already driven from `cfg.max_range` so just change the default.

6. **TX leakage**: More TX power means stronger TX-to-RX leakage at zero range. The existing `min_range` filter handles this, but may need to increase from 1.0 m to 1.5-2.0 m depending on leakage levels.

### Validation Plan

1. **Bench test**: Verify PA output power with a spectrum analyzer at 10.15 GHz. Confirm P1dB and check for spurs/harmonics.
2. **Loopback**: Connect TX to RX through known attenuation, verify the 18 dB gain shows up as expected increase in beat frequency amplitude.
3. **Range test**: Compare detection range with and without PA on the same target (corner reflector at known distances).
4. **Saturation check**: Walk toward the radar from max range — verify no ADC clipping or CFAR breakdown at close range. Adjust rx_gain as needed.

## Alternative: ADPA1120 GaN Power Amplifier

The [ADPA1120](../docs/adpa1120.pdf) is a dramatically more capable option — a 4-stage GaN MMIC that delivers **4.5 W** (36.5 dBm) pulsed output with 35.5 dB of power gain. Purpose-built for X-band radar applications (weather, marine, military).

### ADPA1120 Specs (9.5-11.5 GHz, pulsed 100 us / 10% duty)

| Parameter | Value |
|---|---|
| Power gain | 35.5 dB (at Pin = 1 dBm) |
| Small signal gain | 38.5 dB |
| Pout | 36.5 dBm (4.5 W) |
| PAE | 47% |
| Supply | +20V drain, -3V to -1V gate, IDQ = 50 mA |
| Max RF input | +12 dBm (abs max) |
| Integrated power detector | Yes (VDET/VREF, temperature-compensated) |
| Package | 32-lead 5x5 mm LFCSP, 50 ohm matched, AC-coupled |
| Thermal resistance | 8.2 C/W pulsed, 10 C/W CW |
| Max channel temp | 225 C |
| ESD | Class 0B (200V HBM) — very sensitive |

### Comparison: HMC451LP3 vs ADPA1120

| | HMC451LP3 | ADPA1120 |
|---|---|---|
| Technology | GaAs pHEMT | GaN |
| Power gain | 18 dB | 35.5 dB |
| Pout (typ) | +21 dBm (125 mW) | +36.5 dBm (4.5 W) |
| Supply | +5V / 120 mA (simple) | +20V drain + negative gate bias (complex) |
| Duty cycle | CW capable | Pulsed recommended (10%, 100 us) |
| Range factor vs baseline | ~2.8x | ~6.7x |
| Complexity | Drop-in, no bias sequencing | Requires gate sequencing, pulsed bias, thermal design |
| Package | 3x3 mm QFN-16 | 5x5 mm LFCSP-32 |

### Expected Range Improvement

With 35.5 dB power gain at Pin = 1 dBm (Pluto output ~0 dBm):
- **TX power**: ~35.5 dBm (~3.5 W)
- **Range factor**: 10^(35.5/40) = **~6.7x** range increase
- If currently detecting to ~15 m, expect **~100 m** with the ADPA1120
- This enters serious short-range radar territory (parking sensors, perimeter security, drone detection)

### Hardware Integration — Key Differences from HMC451LP3

1. **Dual-polarity supply**: Needs +20V drain bias and -3V to -1V negative gate bias. The Pi's 5V rail won't work — requires a separate 20V supply (boost converter or bench supply) plus a negative voltage generator or adjustable negative LDO for the gate. The gate voltage sets IDQ and must be adjustable.

2. **Bias sequencing is mandatory**: Gate must be pinched off (-4V) before drain voltage is applied, and RF must only be present when bias is established. Sequence: (1) set VGG = -4V, (2) apply VDD = 20V, (3) adjust VGG to set IDQ = 50 mA, (4) apply RF. Reverse for shutdown. Getting this wrong can destroy the device.

3. **Pulsed operation**: The ADPA1120 is spec'd at 100 us pulse width, 10% duty cycle. Our FMCW chirps are 300 us ramp time, with PRI ~500 us. During a burst of 64 chirps, the duty cycle is effectively ~60% (300/500). This exceeds the 10% duty cycle spec and pushes thermal limits. Options:
   - **Gate pulsing**: Pulse the gate bias in sync with TDD — the ADPA1120 supports gate pulsing between -4V (off) and operating voltage (on). The TDD engine already generates TX enable signals that could drive a gate pulse circuit.
   - **Drain pulsing**: Pulse VDD between 0V and 20V in sync with chirps — requires MOSFET switch driver circuit (eval board includes a plugin pulser for this).
   - **Reduce duty cycle**: Increase PRI (add dead time between chirps) to bring average duty cycle below 10%. Costs frame rate but keeps the PA in safe operating region.
   - **Derate supply voltage**: Run at 14V instead of 20V — reduces Pout but improves thermal margin for higher duty cycle. Datasheet shows performance at various VDD.

4. **Thermal management**: At 4.5 W output with 47% PAE, total dissipation is ~5 W during pulses. With 8.2 C/W thermal resistance pulsed, that's ~41 C junction rise per pulse. The exposed pad must be soldered to a copper ground plane with thermal vias and possibly a heatsink. Much more critical than the HMC451LP3's 600 mW.

5. **ESD sensitivity**: Class 0B (200V HBM) — significantly more sensitive than the HMC451LP3 (250V Class 1A). Requires careful handling with grounded wrist straps, ESD-safe workstation.

6. **Integrated power detector**: The VDET/VREF pins provide a temperature-compensated DC voltage proportional to RF output power. This can be read with an ADC (Pi GPIO or external) to implement **closed-loop power monitoring** — useful for AGC, fault detection (antenna disconnected), and verifying PA health.

### Software Changes (vs HMC451LP3)

All HMC451LP3 software changes apply, plus:

1. **TX power is ~17 dB higher than HMC451LP3** — rx_gain likely needs to drop to 40-45 (from 60) to avoid ADC saturation. Software AGC becomes much more important.

2. **Pulsed bias control**: If gate-pulsing, the TDD engine's existing TX enable GPIO output could drive a gate pulse circuit. The software already manages TDD channel timing — may need to add a channel for PA gate enable with appropriate timing margins (gate on before chirp, off after).

3. **Power detector readback**: Read VDET/VREF via ADC, compute `VREF - VDET` for temperature-compensated power measurement. Display in GUI info panel. Set alarm threshold for over/under power conditions.

4. **max_range should increase to 80-120 m**: More range bins to process (~400-600 after crop), but still fast. May want to increase `cfar_ref_range` to match the larger map.

5. **Chirp BW considerations**: At 100 m range, the beat frequency for a 500 MHz / 300 us chirp is ~1.1 MHz — well within the 4 MSPS sample rate (2 MHz Nyquist). No sample rate changes needed. But consider whether range sidelobes from the Hann window (-31 dB) are sufficient when dynamic range now spans >50 dB — may want to upgrade to Taylor or Chebyshev windows.

## Deployment

```bash
bash tools/deploy.sh
```

Copies `Playground/` (including `lib/`) to `analog@phaser.local:/home/analog/yolo` via SSH.
