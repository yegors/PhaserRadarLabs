"""
hardware.py — Hardware initialization, data capture, and reconfiguration.

Handles all ADI device setup (Phaser, PlutoSDR, TDD engine),
burst triggering, chirp count reconfiguration, and cleanup.
"""

import socket
import numpy as np
from types import SimpleNamespace

import adi

from lib.config import RadarConfig
from lib.radar_processor import compute_axes
from lib.angle_estimator import MonopulseEstimator
from lib.tracker import KalmanTracker


def init_hardware(cfg):
    """Initialize all hardware and compute derived config values.

    Args:
        cfg: RadarConfig instance (mutated — derived values are set).

    Returns:
        HardwareContext SimpleNamespace with: sdr, phaser, tdd, sdr_pins, monopulse, tracker
    """
    # Auto-detect: running on the Phaser Pi locally, or remotely?
    if "phaser" in socket.gethostname():
        rpi_ip = "ip:localhost"
        sdr_ip = "ip:192.168.2.1"
    else:
        rpi_ip = "ip:phaser.local"
        sdr_ip = "ip:phaser.local:50901"

    print(f"Connecting: SDR={sdr_ip}, Phaser={rpi_ip}")
    sdr = adi.ad9361(uri=sdr_ip)
    phaser = adi.CN0566(uri=rpi_ip, sdr=sdr)

    print(f"pyadi-iio version: {adi.__version__}")

    # Initialize phaser
    phaser.configure(device_mode="rx")
    phaser.element_spacing = 0.014
    phaser.load_gain_cal()
    phaser.load_phase_cal()
    for i in range(8):
        phaser.set_chan_phase(i, 0)

    gain_list = [127] * 8
    for i in range(len(gain_list)):
        phaser.set_chan_gain(i, gain_list[i], apply_cal=True)

    # GPIO setup
    phaser._gpios.gpio_tx_sw = 0
    phaser._gpios.gpio_vctrl_1 = 1
    phaser._gpios.gpio_vctrl_2 = 1

    # Configure SDR Rx — keep both channels SEPARATE for monopulse
    sdr.sample_rate = int(cfg.sample_rate)
    sdr.rx_lo = int(cfg.center_freq)
    sdr.rx_enabled_channels = [0, 1]
    sdr.gain_control_mode_chan0 = 'manual'
    sdr.gain_control_mode_chan1 = 'manual'
    # Apply channel calibration (compensates sub-array / SDR Rx gain mismatch)
    try:
        phaser.load_channel_cal()
        ccal = phaser.ccal
        print(f"Channel cal loaded: [{ccal[0]:.1f}, {ccal[1]:.1f}] dB")
    except Exception:
        ccal = [0.0, 0.0]
    sdr.rx_hardwaregain_chan0 = int(cfg.rx_gain + ccal[0])
    sdr.rx_hardwaregain_chan1 = int(cfg.rx_gain + ccal[1])

    # Configure SDR Tx
    sdr.tx_lo = int(cfg.center_freq)
    sdr.tx_enabled_channels = [0, 1]
    sdr.tx_cyclic_buffer = True
    sdr.tx_hardwaregain_chan0 = -88
    sdr.tx_hardwaregain_chan1 = int(cfg.tx_gain)

    # Configure ADF4159 chirp generator
    vco_freq = int(cfg.output_freq + cfg.signal_freq + cfg.center_freq)
    num_steps = int(cfg.ramp_time)
    phaser.frequency = int(vco_freq / 4)
    phaser.freq_dev_range = int(cfg.chirp_BW / 4)
    phaser.freq_dev_step = int((cfg.chirp_BW / 4) / num_steps)
    phaser.freq_dev_time = int(cfg.ramp_time)
    print(f"Requested freq dev time (us): {cfg.ramp_time}")
    phaser.delay_word = 4095
    phaser.delay_clk = "PFD"
    phaser.delay_start_en = 0
    phaser.ramp_delay_en = 0
    phaser.trig_delay_en = 0
    phaser.ramp_mode = "single_sawtooth_burst"
    phaser.sing_ful_tri = 0
    phaser.tx_trig_en = 1
    phaser.enable = 0

    # TDD Synchronization
    sdr_pins = adi.one_bit_adc_dac(sdr_ip)
    sdr_pins.gpio_tdd_ext_sync = True
    tdd = adi.tddn(sdr_ip)
    sdr_pins.gpio_phaser_enable = True
    tdd.enable = False
    tdd.sync_external = True
    tdd.startup_delay_ms = 0
    PRI_ms = cfg.ramp_time / 1e3 + 0.2
    tdd.frame_length_ms = PRI_ms
    tdd.burst_count = cfg.num_chirps

    tdd.channel[0].enable = True
    tdd.channel[0].polarity = False
    tdd.channel[0].on_raw = 0
    tdd.channel[0].off_raw = 10
    tdd.channel[1].enable = True
    tdd.channel[1].polarity = False
    tdd.channel[1].on_raw = 0
    tdd.channel[1].off_raw = 10
    tdd.channel[2].enable = True
    tdd.channel[2].polarity = False
    tdd.channel[2].on_raw = 0
    tdd.channel[2].off_raw = 10
    tdd.enable = True

    # Compute derived timing values
    ramp_time_actual = int(phaser.freq_dev_time)
    print(f"Actual freq dev time: {ramp_time_actual} us")
    cfg.recompute_derived(ramp_time_actual, tdd.frame_length_ms, tdd.channel[0].on_ms)

    print(f"FFT size: {cfg.fft_size}")
    print(f"Total chirp burst time: {cfg.frame_time * 1000:.1f} ms")
    print(f"Buffer size: {cfg.buffer_size}, buffer time: {cfg.buffer_size / cfg.sample_rate * 1000:.1f} ms")
    sdr.rx_buffer_size = cfg.buffer_size

    print(f"Range resolution: {cfg.axes['range_res']:.2f} m")
    print(f"Velocity resolution: {cfg.axes['velocity_res']:.3f} m/s")
    print(f"Max unambiguous velocity: {cfg.axes['max_velocity']:.1f} m/s")
    print(f"Range bins: {len(cfg.axes['range_axis'])} total, "
          f"{len(cfg.range_crop_idx)} after crop to {cfg.max_range}m")

    # Initialize processing modules
    monopulse = MonopulseEstimator(
        element_spacing=0.014,
        num_elements=8,
        output_freq=cfg.output_freq + cfg.chirp_BW / 2,
    )

    tracker = KalmanTracker(
        dt=cfg.frame_time,
        confirm_m=cfg.track_confirm_m,
        confirm_n=cfg.track_confirm_n,
        max_misses=cfg.track_max_misses,
        gate_distance=cfg.track_gate_distance,
    )

    # Create TX waveform
    N = int(2**18)
    fc = int(cfg.signal_freq)
    ts = 1 / float(cfg.sample_rate)
    t = np.arange(0, N * ts, ts)
    i_sig = np.cos(2 * np.pi * t * fc) * 2**14
    q_sig = np.sin(2 * np.pi * t * fc) * 2**14
    iq = 0.9 * (i_sig + 1j * q_sig)
    sdr.tx([iq, iq])

    hw = SimpleNamespace(
        sdr=sdr,
        phaser=phaser,
        tdd=tdd,
        sdr_pins=sdr_pins,
        monopulse=monopulse,
        tracker=tracker,
    )

    print(f"Sample rate: {cfg.sample_rate/1e6} MHz, Ramp time: {cfg.ramp_time} us, "
          f"Chirps: {cfg.num_chirps}, MTI: {cfg.mti_mode}")
    print(f"CFAR: {'ON' if cfg.cfar_enabled else 'OFF'}, "
          f"Tracking: {'ON' if cfg.tracking_enabled else 'OFF'}")

    return hw


def get_radar_data(hw):
    """Capture one burst and return both channels separately.

    Returns:
        (chan0, chan1): Complex IQ arrays for each Rx channel.
    """
    hw.phaser._gpios.gpio_burst = 0
    hw.phaser._gpios.gpio_burst = 1
    hw.phaser._gpios.gpio_burst = 0
    data = hw.sdr.rx()
    return data[0], data[1]


def apply_chirp_config(cfg, hw, new_num):
    """Reconfigure hardware and processing for a new chirp count.

    Args:
        cfg: RadarConfig instance (mutated).
        hw: HardwareContext.
        new_num: New number of chirps.

    Returns:
        True if the chirp count actually changed (axes need GUI update).
    """
    if new_num == cfg.num_chirps:
        return False

    cfg.num_chirps = new_num

    # Reconfigure TDD burst count
    hw.tdd.enable = False
    hw.tdd.burst_count = cfg.num_chirps
    hw.tdd.enable = True

    # Recompute derived values (buffer, axes, frame time)
    ramp_time_actual = int(hw.phaser.freq_dev_time)
    cfg.recompute_derived(ramp_time_actual, hw.tdd.frame_length_ms, hw.tdd.channel[0].on_ms)

    hw.sdr.rx_buffer_size = cfg.buffer_size

    # Reset tracker with new frame time
    hw.tracker = KalmanTracker(
        dt=cfg.frame_time,
        confirm_m=cfg.track_confirm_m,
        confirm_n=cfg.track_confirm_n,
        max_misses=cfg.track_max_misses,
        gate_distance=cfg.track_gate_distance,
    )

    print(f"Chirps -> {cfg.num_chirps}: burst {cfg.frame_time*1000:.0f}ms, "
          f"vel res {cfg.axes['velocity_res']:.3f} m/s, buf {cfg.buffer_size}")

    return True


def cleanup(hw):
    """Release hardware resources."""
    hw.sdr.tx_destroy_buffer()
    print("Pluto TX buffer cleared")
