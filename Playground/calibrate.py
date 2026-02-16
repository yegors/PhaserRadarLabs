#!/usr/bin/env python3
"""
Minimal CN0566 calibration utility.

Calibrates one or more scopes and saves the corresponding file(s):
- channel -> channel_cal_val.pkl
- gain    -> gain_cal_val.pkl
- phase   -> phase_cal_val.pkl

No args runs all three in order: channel, gain, phase.

Important:
- This script is configured for the on-board TX path (OUT1 horn / attached antenna),
  not an external microwave source (for example HB100 in some example scripts).

Hardware notes (CN0566, from project docs/examples):
- 8-element phased array (two 4-element sub-arrays)
- PlutoSDR provides two Rx channels and TX DDS tone source
- Calibration is done in CW mode (ADF4159 ramp disabled)
"""

import argparse
import socket
import sys
import time
from statistics import mean

import numpy as np
from scipy.signal import windows

import adi

# --- CW calibration settings (kept simple and close to examples) ---
SIGNAL_FREQ = 10.25e9
RX_LO = int(2.1e9)
SAMPLE_RATE = 30_000_000
BUFFER_SIZE = 1024
RX_GAIN = 6
TX_GAIN = -3
DDS_FREQ = 2_000_000
NUM_AVERAGES = 4
PHASE_STEP = 2.8125
SETTLE_TIME = 0.5
PEAK_WIDTH = 10
SPARK_CHARS = "▁▂▃▄▅▆▇█"
MIN_GOOD_SNR_DB = 25.0
EXPECTED_BIN_TOL = 8


def _fmt_db(x):
    return f"{x:+.2f} dB"


def _expected_tone_bin():
    center = BUFFER_SIZE // 2
    offset = int(round(DDS_FREQ / SAMPLE_RATE * BUFFER_SIZE))
    return int((center + offset) % BUFFER_SIZE)


def _is_bad_run(peak_bin, snr_db):
    expected = _expected_tone_bin()
    bin_error = abs(int(peak_bin) - int(expected))
    bad = (snr_db < MIN_GOOD_SNR_DB) or (bin_error > EXPECTED_BIN_TOL)
    return bad, expected, bin_error


def _ascii_bar(value, vmin, vmax, width=28):
    if vmax <= vmin:
        fill = width
    else:
        ratio = (value - vmin) / (vmax - vmin)
        ratio = max(0.0, min(1.0, ratio))
        fill = int(round(ratio * width))
    return "#" * fill + "-" * (width - fill)


def _sparkline(values, width=64):
    if not values:
        return ""
    arr = np.asarray(values, dtype=float)
    if len(arr) > width:
        idx = np.linspace(0, len(arr) - 1, width).astype(int)
        arr = arr[idx]
    vmin = float(np.min(arr))
    vmax = float(np.max(arr))
    if vmax <= vmin:
        return SPARK_CHARS[0] * len(arr)
    scaled = (arr - vmin) / (vmax - vmin)
    bins = np.clip((scaled * (len(SPARK_CHARS) - 1)).astype(int), 0, len(SPARK_CHARS) - 1)
    return "".join(SPARK_CHARS[b] for b in bins)


def connect():
    """Connect to Phaser + PlutoSDR (local-first, then remote)."""
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
    """Configure SDR + CN0566 for CW calibration mode."""
    phaser.configure(device_mode="rx")
    phaser.element_spacing = 0.014

    # Match runtime GPIO routing used for on-board TX path.
    try:
        phaser._gpios.gpio_tx_sw = 0
        phaser._gpios.gpio_vctrl_1 = 1
        phaser._gpios.gpio_vctrl_2 = 1
    except Exception:
        pass

    # SDR RX
    sdr.sample_rate = SAMPLE_RATE
    sdr.rx_lo = RX_LO
    sdr.rx_enabled_channels = [0, 1]
    sdr.rx_buffer_size = BUFFER_SIZE
    sdr.rx_rf_bandwidth = int(10e6)
    sdr.gain_control_mode_chan0 = "manual"
    sdr.gain_control_mode_chan1 = "manual"
    sdr.rx_hardwaregain_chan0 = RX_GAIN
    sdr.rx_hardwaregain_chan1 = RX_GAIN

    # ADF4159 in CW mode
    pll_freq = (int(SIGNAL_FREQ) + RX_LO) // 4
    phaser.frequency = pll_freq
    phaser.freq_dev_step = 5690
    phaser.freq_dev_range = 0
    phaser.freq_dev_time = 0
    phaser.powerdown = 0
    phaser.ramp_mode = "disabled"

    # SDR TX DDS tone
    sdr.tx_lo = RX_LO
    sdr.tx_enabled_channels = [0, 1]
    # OUT1 path: keep TX0 muted, drive TX1 with DDS tone.
    sdr.tx_hardwaregain_chan0 = -88
    sdr.tx_hardwaregain_chan1 = TX_GAIN
    try:
        sdr.tx_destroy_buffer()
    except Exception:
        pass
    sdr.dds_single_tone(DDS_FREQ, 0.9, 1)

    # Array defaults
    phaser.set_all_gain(127)
    phaser.set_beam_phase_diff(0.0)
    phaser.Averages = NUM_AVERAGES

    time.sleep(1.0)
    print("  CW mode configured\n")


def find_peak_bin_and_snr(sdr, phaser, verbose=False):
    """Simple signal sanity check (single-capture)."""
    win = np.blackman(BUFFER_SIZE)
    phaser.set_all_gain(127)
    phaser.set_beam_phase_diff(0.0)
    time.sleep(0.2)

    data = sdr.rx()
    y0 = data[0] * win
    y1 = data[1] * win
    ysum = (data[0] + data[1]) * win
    spec0 = np.fft.fftshift(np.absolute(np.fft.fft(y0)))
    spec1 = np.fft.fftshift(np.absolute(np.fft.fft(y1)))
    spectrum = np.fft.fftshift(np.absolute(np.fft.fft(ysum)))
    peak = int(np.argmax(spectrum))
    noise = float(np.median(spectrum))
    snr = 20 * np.log10(spectrum[peak] / (noise + 1e-20) + 1e-20)

    if verbose:
        peak0 = int(np.argmax(spec0))
        peak1 = int(np.argmax(spec1))
        snr0 = 20 * np.log10(spec0[peak0] / (np.median(spec0) + 1e-20) + 1e-20)
        snr1 = 20 * np.log10(spec1[peak1] / (np.median(spec1) + 1e-20) + 1e-20)
        print("  Debug signal summary:")
        print(f"    CH0 peak bin {peak0:4d}  SNR {snr0:6.2f} dB")
        print(f"    CH1 peak bin {peak1:4d}  SNR {snr1:6.2f} dB")
        print(f"    SUM peak bin {peak:4d}  SNR {snr:6.2f} dB")

    return peak, float(snr)


def measure_at_peak(sdr, peak_bin):
    win = windows.flattop(BUFFER_SIZE)
    win /= np.average(np.abs(win))
    total = 0.0
    for _ in range(NUM_AVERAGES):
        sdr.rx()  # flush
        data = sdr.rx()
        y = (data[0] + data[1]) * win
        spectrum = np.fft.fftshift(np.absolute(np.fft.fft(y)))
        lo = max(0, peak_bin - PEAK_WIDTH)
        hi = min(BUFFER_SIZE, peak_bin + PEAK_WIDTH)
        total += float(np.max(spectrum[lo:hi]))
    return total / (NUM_AVERAGES * BUFFER_SIZE)


def run_channel_cal(sdr, phaser, peak_bin, verbose=False, graph=False):
    print("Step: Channel Calibration")
    win = windows.flattop(BUFFER_SIZE)
    win /= np.average(np.abs(win))

    channel_levels = []
    channel_db = []
    for ch in range(2):
        phaser.set_all_gain(0, apply_cal=False)
        for el in range(4):
            phaser.set_chan_gain((1 - ch) * 4 + el, 127, apply_cal=False)
        time.sleep(SETTLE_TIME)

        total = 0.0
        for _ in range(NUM_AVERAGES):
            sdr.rx()
            data = sdr.rx()
            y = (data[0] + data[1]) * win
            spectrum = np.fft.fftshift(np.absolute(np.fft.fft(y)))
            lo = max(0, peak_bin - PEAK_WIDTH)
            hi = min(BUFFER_SIZE, peak_bin + PEAK_WIDTH)
            total += float(np.max(spectrum[lo:hi]))

        level = total / (NUM_AVERAGES * BUFFER_SIZE)
        channel_levels.append(level)
        level_db = 20 * np.log10(level + 1e-20)
        channel_db.append(level_db)
        print(f"  Sub-array {ch}: {level:.6f} ({level_db:.2f} dBFS-ish)")

    mismatch_db = 20.0 * np.log10(channel_levels[0] / channel_levels[1])
    if mismatch_db > 0:
        phaser.ccal = [0.0, mismatch_db]
    else:
        phaser.ccal = [-mismatch_db, 0.0]

    print(f"  Mismatch: {mismatch_db:+.1f} dB")
    print(f"  ccal = [{phaser.ccal[0]:.1f}, {phaser.ccal[1]:.1f}]\n")

    if graph:
        lo, hi = min(channel_levels), max(channel_levels)
        print("  Channel level graph:")
        for ch, lv in enumerate(channel_levels):
            print(f"    CH{ch} |{_ascii_bar(lv, lo, hi)}| {lv:.4f}")
        print()

    if verbose:
        print("  Channel summary:")
        print(f"    mean level: {mean(channel_levels):.6f}")
        print(f"    mismatch : {_fmt_db(mismatch_db)}")
        print(f"    ccal     : [{phaser.ccal[0]:.2f}, {phaser.ccal[1]:.2f}]\n")

    return {
        "levels": channel_levels,
        "levels_db": channel_db,
        "mismatch_db": float(mismatch_db),
    }


def run_gain_cal(sdr, phaser, peak_bin, verbose=False, graph=False):
    print("Step: Gain Calibration")
    element_levels = []

    for el in range(8):
        phaser.set_all_gain(0, apply_cal=False)
        phaser.set_chan_gain(el, 127, apply_cal=False)
        time.sleep(SETTLE_TIME)

        level = measure_at_peak(sdr, peak_bin)
        element_levels.append(level)
        print(f"  Element {el}: {level:.6f}")

    min_level = min(element_levels)
    max_level = max(element_levels)
    for idx in range(8):
        phaser.gcal[idx] = min_level / element_levels[idx]

    spread_db = 20 * np.log10(max_level / min_level)
    print(f"  Gain spread: {spread_db:.1f} dB")
    print(f"  gcal = [{', '.join(f'{g:.3f}' for g in phaser.gcal)}]\n")

    if graph:
        print("  Element gain graph:")
        for el, lv in enumerate(element_levels):
            print(f"    E{el} |{_ascii_bar(lv, min_level, max_level)}| {lv:.4f}")
        print()

    if verbose:
        strongest = int(np.argmax(element_levels))
        weakest = int(np.argmin(element_levels))
        print("  Gain summary:")
        print(f"    strongest element: E{strongest} ({max_level:.6f})")
        print(f"    weakest element  : E{weakest} ({min_level:.6f})")
        print(f"    spread           : {_fmt_db(spread_db)}\n")

    return {
        "element_levels": element_levels,
        "spread_db": float(spread_db),
        "gcal": list(phaser.gcal),
    }


def run_phase_cal(sdr, phaser, peak_bin, verbose=False, graph=False):
    print("Step: Phase Calibration")
    win = windows.flattop(BUFFER_SIZE)
    win /= np.average(np.abs(win))
    sweep_phases = np.arange(-180, 180, PHASE_STEP)

    # Prefer gain-cal-compensated pair sweeps if gain cal file exists
    try:
        phaser.load_gain_cal()
        apply_gain_cal = True
    except Exception:
        apply_gain_cal = False

    phaser.pcal = [0.0] * 8
    pair_sweeps = []
    pair_deltas = []

    for pair in range(7):
        ref_el = pair
        cal_el = pair + 1

        phaser.set_all_gain(0, apply_cal=apply_gain_cal)
        phaser.set_chan_gain(ref_el, 127, apply_cal=apply_gain_cal)
        phaser.set_chan_gain(cal_el, 127, apply_cal=apply_gain_cal)
        time.sleep(SETTLE_TIME)

        phaser.set_chan_phase(ref_el, 0.0, apply_cal=False)

        gains = []
        for phase in sweep_phases:
            phaser.set_chan_phase(cal_el, phase, apply_cal=False)
            total = 0.0
            for _ in range(NUM_AVERAGES):
                sdr.rx()
                data = sdr.rx()
                y = (data[0] + data[1]) * win
                spectrum = np.fft.fftshift(np.absolute(np.fft.fft(y)))
                lo = max(0, peak_bin - PEAK_WIDTH)
                hi = min(BUFFER_SIZE, peak_bin + PEAK_WIDTH)
                total += float(np.max(spectrum[lo:hi]))
            gains.append(total / (NUM_AVERAGES * BUFFER_SIZE))
        pair_sweeps.append(gains)

        null_idx = int(np.argmin(gains))
        null_phase = float(sweep_phases[null_idx])
        ph_delta = (180 - null_phase) % 360
        if ph_delta > 180:
            ph_delta -= 360
        pair_deltas.append(float(ph_delta))

        phaser.pcal[cal_el] = (phaser.pcal[ref_el] - ph_delta) % 360
        if phaser.pcal[cal_el] > 180:
            phaser.pcal[cal_el] -= 360

        print(f"  Pair {ref_el}-{cal_el}: null {null_phase:+6.1f}°, delta {ph_delta:+6.1f}°")

    print(f"  pcal = [{', '.join(f'{p:+.1f}' for p in phaser.pcal)}]\n")

    if graph:
        print("  Phase sweep graphs (sparkline, low->high amplitude):")
        for pair, gains in enumerate(pair_sweeps):
            print(f"    P{pair}-{pair+1} {_sparkline(gains)}")
        print()

    if verbose:
        abs_d = np.abs(np.asarray(pair_deltas))
        print("  Phase summary:")
        print(f"    mean |delta|: {float(np.mean(abs_d)):.1f} deg")
        print(f"    max  |delta|: {float(np.max(abs_d)):.1f} deg")
        print(f"    pcal vector : [{', '.join(f'{p:+.1f}' for p in phaser.pcal)}]\n")

    return {
        "pair_deltas": pair_deltas,
        "pcal": list(phaser.pcal),
        "pair_sweeps": pair_sweeps,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="CN0566 calibration utility")
    parser.add_argument(
        "scopes",
        nargs="*",
        choices=["all", "channel", "gain", "phase"],
        help="Calibration scopes to run (default: all)",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Print extended debug summaries",
    )
    parser.add_argument(
        "-g", "--graph",
        action="store_true",
        help="Print ASCII CLI graphs for calibration sweeps",
    )

    args = parser.parse_args()

    if not args.scopes or "all" in args.scopes:
        scopes = ["channel", "gain", "phase"]
    else:
        ordered = ["channel", "gain", "phase"]
        selected = set(args.scopes)
        scopes = [scope for scope in ordered if scope in selected]

    return scopes, args


def main():
    scopes, args = parse_args()

    print()
    print("=" * 60)
    print("  CN0566 Phaser — Minimal Calibration")
    print("=" * 60)
    print(f"Scopes: {', '.join(scopes)}")
    print(f"Options: verbose={'ON' if args.verbose else 'OFF'}, graph={'ON' if args.graph else 'OFF'}")
    print()

    print("Connecting to hardware...")
    sdr, phaser = connect()
    setup_cw(sdr, phaser)
    print("TX source: on-board DDS -> CN0566 OUT1 (attached horn path)")

    print("Checking signal quality...")
    t_sig0 = time.perf_counter()
    peak_bin, snr = find_peak_bin_and_snr(sdr, phaser, verbose=args.verbose)
    t_sig1 = time.perf_counter()
    bad_run, expected_bin, bin_error = _is_bad_run(peak_bin, snr)
    print(f"  Peak bin: {peak_bin}/{BUFFER_SIZE}, SNR: {snr:.0f} dB")
    print(f"  Expected tone bin: {expected_bin}, error: {bin_error}")
    if args.verbose:
        print(f"  Signal check time: {(t_sig1 - t_sig0)*1000:.1f} ms")
    if bad_run:
        print("  WARNING: This looks like a bad calibration run (low SNR or wrong tone bin).")
    print()

    t0 = time.perf_counter()
    scope_results = {}

    if "channel" in scopes:
        phaser.set_beam_phase_diff(0.0)
        scope_results["channel"] = run_channel_cal(
            sdr, phaser, peak_bin, verbose=args.verbose, graph=args.graph
        )
        print()

    if "gain" in scopes:
        phaser.set_beam_phase_diff(0.0)
        scope_results["gain"] = run_gain_cal(
            sdr, phaser, peak_bin, verbose=args.verbose, graph=args.graph
        )
        print()

    if "phase" in scopes:
        scope_results["phase"] = run_phase_cal(
            sdr, phaser, peak_bin, verbose=args.verbose, graph=args.graph
        )
        print()

    keep_files = True
    if bad_run:
        resp = input("Bad run detected. Keep and save calibration files anyway? (y/n): ").strip().lower()
        keep_files = (resp == "y")

    if keep_files:
        if "channel" in scopes:
            phaser.save_channel_cal()
            print("  Saved: channel_cal_val.pkl")
        if "gain" in scopes:
            phaser.save_gain_cal()
            print("  Saved: gain_cal_val.pkl")
        if "phase" in scopes:
            phaser.save_phase_cal()
            print("  Saved: phase_cal_val.pkl")
    else:
        print("  Discarded calibration files (nothing written).")

    print()

    t1 = time.perf_counter()

    print("Run summary:")
    print(f"  scopes run : {', '.join(scopes)}")
    print(f"  peak bin   : {peak_bin}")
    print(f"  expected   : {expected_bin} (error {bin_error})")
    print(f"  SNR        : {snr:.1f} dB")
    print(f"  quality    : {'BAD' if bad_run else 'GOOD'}")
    if "channel" in scope_results:
        print(f"  ch mismatch: {scope_results['channel']['mismatch_db']:+.2f} dB")
    if "gain" in scope_results:
        print(f"  gain spread: {scope_results['gain']['spread_db']:.2f} dB")
    if "phase" in scope_results:
        max_abs_delta = float(np.max(np.abs(np.asarray(scope_results['phase']['pair_deltas']))))
        print(f"  max |Δφ|   : {max_abs_delta:.1f} deg")
    print(f"  elapsed    : {t1 - t0:.2f} s")
    print("Done.")

    try:
        sdr.tx_destroy_buffer()
    except Exception:
        pass


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(130)
