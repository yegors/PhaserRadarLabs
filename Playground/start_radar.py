"""start_radar.py — Launch the FMCW range-Doppler radar."""

import signal
import pyqtgraph as pg

signal.signal(signal.SIGINT, signal.SIG_DFL)
pg.setConfigOptions(antialias=False, background='#0d1117', foreground='#c9d1d9')

from lib.config import RadarConfig
from lib.hardware import init_hardware
from lib.gui import RadarGUI

cfg = RadarConfig()
hw = init_hardware(cfg)
gui = RadarGUI(cfg, hw)

try:
    gui.run()
finally:
    gui.cleanup()
