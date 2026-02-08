# PhaserRadarLabs
Example Radar programs for the Phaser (CN0566)

These are example radar programs for the Phaser (www.analog.com/cn0566).
Getting started instructions are here:  https://wiki.analog.com/phaser
Video walkthrough of these files are available here:  https://www.youtube.com/@jonkraft

Example radar data, such as can be used with the "Radar_Doppler_Processing" file, is here:  https://ez.analog.com/adieducation/university-program/m/file-uploads

## Scripts

### Radar demos (root directory)

**`CW_RADAR_Waterfall.py`** -- The simplest radar mode. Sends out a steady signal at one frequency and listens for reflections. When something moves toward or away from the radar, the reflected signal comes back at a slightly different frequency (the Doppler effect -- same reason a siren changes pitch as it passes you). This script shows that frequency shift on a live display so you can see motion happening in real time. It can tell you something is moving and roughly how fast, but it cannot tell you how far away it is.

- Transmit frequency: 10.25 GHz (fixed, no sweep)
- ADF4159 PLL ramp mode: disabled (constant tone)
- IF signal: 100 kHz via PlutoSDR DDS hardware
- Sample rate: 0.6 MHz, FFT size: 4096
- Array taper: Blackman (gain pattern `[8, 34, 84, 127, 127, 84, 34, 8]`)
- GUI: PyQt5/pyqtgraph with live FFT + scrolling waterfall
- Measures: velocity (Doppler). Cannot measure range.

**`FMCW_RADAR_Waterfall.py`** -- Instead of transmitting at one frequency, this script sweeps the signal up and down in frequency over time (called a "chirp"). When the chirp bounces off a target and comes back, it's delayed by the round-trip travel time. Mixing this delayed return with the current transmit creates a "beat" tone whose frequency is proportional to the target's distance. Farther targets produce higher beat frequencies. The GUI lets you adjust the sweep bandwidth (which controls how precisely you can distinguish two close objects) and steer the antenna beam electronically.

- Transmit frequency: ~12.145 GHz, sweeping 500 MHz bandwidth
- ADF4159 PLL ramp mode: continuous triangular (sweeps up and down)
- Ramp time: 0.5 ms, chirp slope: 1 THz/s
- Range formula: `distance = beat_freq * c / (2 * slope)`
- Range resolution: 0.3 m (at 500 MHz BW), adjustable via slider down to 1.5 m (at 100 MHz)
- Sample rate: 0.6 MHz, FFT size: 4096
- GUI adds: chirp BW slider, "Convert to Distance" checkbox, beam steering (-80 to +80 deg)
- Measures: range. Limited velocity info.

**`CFAR_RADAR_Waterfall.py`** -- Same range-measuring radar as above, but with automatic target detection added. Instead of staring at the spectrum and trying to decide "is that bump a real target or just noise?", the CFAR algorithm does it for you. It slides a window across the spectrum, estimates the local noise level around each point, and flags anything that sticks out significantly above that noise floor. Detected targets are marked with red dots on the plot. You can tune how sensitive the detector is with sliders.

- Same FMCW chirp setup as `FMCW_RADAR_Waterfall.py`
- CFAR algorithm: slides a window of "reference cells" around each point to estimate local noise
- "Guard cells" are skipped around the test point so the target itself doesn't inflate the noise estimate
- Detection threshold = estimated noise + user-adjustable bias (in dB)
- GUI adds: guard cell count slider, reference cell count slider, bias slider
- Uses `target_detection_dbfs.py` module
- Measures: range + automatic target detection

**`FMCW_RADAR_Waterfall_ChirpSync.py`** -- A cleaner version of the FMCW radar. The basic FMCW script runs its chirp continuously and captures data whenever -- so the receive buffer might contain a random mix of up-sweeps and down-sweeps, making the spectrum messy. This script instead fires one chirp at a time on command and knows exactly where in the receive buffer that chirp's data lands. It then extracts only the good linear portion of that chirp, producing a much cleaner range measurement.

- ADF4159 PLL ramp mode: single sawtooth burst (one chirp per trigger)
- Chirp triggered by PlutoSDR's TDD engine (`adi.tddn`) via GPIO burst pulse
- Only the linear portion of each ramp is used (first 10% discarded for PLL settling)
- Sample rate: 5 MHz (higher than basic FMCW for better range coverage)
- FFT size: dynamically sized to fit one chirp frame
- Buffer size: dynamically calculated from chirp timing
- Requires Pluto firmware v0.39+ (for TDD engine support)
- Measures: range (cleaner than basic FMCW)

**`CFAR_RADAR_Waterfall_ChirpSync.py`** -- The best of both worlds: chirp-synchronized clean data capture combined with automatic CFAR target detection. Gives you the cleanest automatic "there's a target at X meters" output.

- Same TDD-synchronized single-sawtooth-burst approach as `FMCW_RADAR_Waterfall_ChirpSync.py`
- Same CFAR detection as `CFAR_RADAR_Waterfall.py`
- Sample rate: 2 MHz, FFT size: 8192
- GUI has all controls: chirp BW, beam steering, waterfall levels, CFAR guard/reference/bias
- Requires Pluto firmware v0.39+
- Measures: range + automatic target detection (cleanest mode)

**`Range_Doppler_Plot.py`** -- The most advanced demo. Instead of firing one chirp at a time, it fires a rapid burst of 256 chirps back-to-back and captures the entire sequence. It then processes this data in two dimensions: within each chirp, it extracts range (how far away things are); across chirps at the same range, it extracts velocity (how fast things are moving). The result is a 2D heatmap where one axis is distance and the other is speed. It also includes a clutter filter that removes stationary objects (walls, furniture) so only moving targets show up.

- 256 consecutive sawtooth chirps per burst via TDD engine
- PRI (pulse repetition interval): ramp_time + 0.2 ms
- 2D FFT: range-FFT within each chirp, Doppler-FFT across chirps
- MTI filter: 2-pulse canceller subtracts consecutive chirps (with phase correction) to remove static clutter
- Display: matplotlib heatmap, x-axis = velocity (m/s), y-axis = range (m)
- Can save raw data to `.npy` files for offline replay
- Sample rate: 4 MHz, ramp time: 300 us, chirp BW: 500 MHz
- Requires Pluto firmware v0.39+
- Measures: range AND velocity simultaneously

**`Range_Doppler_Processing.py`** -- Offline analysis companion to `Range_Doppler_Plot.py`. Loads previously saved `.npy` radar data files and runs the same 2D FFT + MTI processing without needing the Phaser hardware connected. Useful for tweaking processing parameters, trying different filters, or analyzing captured data on a different computer.

- No hardware required -- loads saved `.npy` data files
- Same 2D FFT and MTI processing as `Range_Doppler_Plot.py`
- Measures: range + velocity (from saved data)

**`target_detection_dbfs.py`** -- Shared helper module used by the CFAR scripts above. Implements the CFAR sliding-window algorithm that decides what counts as a real target vs. background noise.

- Four detection modes:
  - `average` (CA-CFAR): uses the average of leading and trailing reference cells
  - `greatest` (GO-CFAR): uses whichever side (leading or trailing) has the higher noise estimate
  - `smallest` (SO-CFAR): uses the lower of the two sides
  - `false_alarm`: sets threshold based on a desired false alarm probability
- Inputs: FFT spectrum (dBFS), number of guard cells, reference cells, bias offset
- Output: adaptive threshold curve

### Phaser utilities (`phaser/` directory)

**`phaser_gui.py`** -- All-in-one GUI for controlling and testing the Phaser board. Lets you steer the beam, adjust per-element gains and phases, run calibration routines, and view live radar data from a single interface. Automatically detects whether you're running it on the Pi itself or from a remote computer.

- Controls: beam steering angle, per-element gain/phase, array taper (Blackman/rectangular/custom), RX/TX gain, chirp parameters
- Displays: live FFT spectrum, polar beam pattern, antenna array diagram
- Auto-detects local (`ip:localhost` + `ip:192.168.2.1`) vs remote (`ip:phaser.local` + `ip:phaser.local:50901`)
- Includes HB100 signal search and calibration routines

**`SDR_functions.py`** -- Helper functions for setting up the PlutoSDR radio. Handles all the low-level radio configuration so other scripts don't have to repeat it.

- `SDR_LO_init()`: programs the ADF4159 PLL (chirp generator)
- `SDR_init()`: configures the AD9361 transceiver (sample rate, frequency, gain, FDD mode, DDS waveform)
- `SDR_setRx()` / `SDR_setTx()`: adjust gains on the fly
- `SDR_getData()`: reads one buffer of receive data
- Loads per-channel gain calibration from `channel_cal_val.pkl`

**`phaser_functions.py`** -- Helper functions for controlling the Phaser board's antenna array.

- `Phaser_Init()`: initializes the ADAR1000 beamformer chips
- `set_Taper()`: applies a gain pattern across the 8 elements (e.g. Blackman for low sidelobes)
- `Steer()`: calculates and applies phase shifts to steer the beam to a given angle
- `update_gains()` / `update_phases()`: per-element control

**`ADAR_pyadi_functions.py`** -- Low-level register access for the ADAR1000 beamformer chips. Two chips with 4 channels each control the 8 antenna elements. Most users won't need this directly -- `phaser_functions.py` wraps it.

**`config.py`** -- Shared defaults used by the phaser utilities: signal frequency (100 kHz IF), sample rate (0.6 MHz), center frequency (2.1 GHz), output frequency (10.25 GHz), chirp bandwidth, buffer sizes, GUI layout.

**`phaser_examples.py`** -- Example routines demonstrating beam sweeping, gain/phase calibration, FFT display, and basic radar operation using the `phaser_functions` API.

**`phaser_minimal_example.py`** -- Bare-minimum "hello world". Connects to the hardware, captures one buffer, prints/plots the result. Good starting template.

**`phaser_find_hb100.py`** -- Sweeps the receiver frequency to find the 10.525 GHz signal from an HB100 Doppler module (a small microwave source used as a known test target for calibration).

**`phaser_prod_tst.py`** -- Factory test script. Checks ADAR1000 communication, GPIO states, TX/RX paths, and gain/phase calibration. Used for hardware QA.

**`RADAR_FFT_Waterfall.py`** -- Older radar waterfall display that uses `SDR_functions.py` and `phaser_functions.py` instead of configuring hardware directly. Functionally similar to `CW_RADAR_Waterfall.py`.
