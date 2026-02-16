"""
gui.py — PyQtGraph radar display and main loop.

RadarGUI class encapsulates the entire display: range-Doppler heatmap,
detection/track markers, info panel, parameter controls, and the
frame update loop.
"""

import json
import time
import numpy as np

import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets, QtCore, QtGui
from pyqtgraph.parametertree import Parameter, ParameterTree

from lib.hardware import get_radar_data, apply_chirp_config, cleanup as hw_cleanup
from lib.pipeline import process_frame

MAX_TRAILS = 30


class RadarGUI:
    """Real-time radar display with PyQtGraph."""

    def __init__(self, cfg, hw):
        self.cfg = cfg
        self.hw = hw

        # ── Window ──
        self.app = pg.mkQApp("Range-Doppler Radar")

        self.win = QtWidgets.QMainWindow()
        self.win.setWindowTitle("Range-Doppler Radar")
        self.win.resize(1400, 800)
        self.win.setStyleSheet("""
            QMainWindow { background-color: #0d1117; }
            QLabel { color: #c9d1d9; }
            QTreeView { background-color: #161b22; color: #c9d1d9; border: 1px solid #30363d; }
            QHeaderView::section { background-color: #21262d; color: #c9d1d9; border: 1px solid #30363d; }
            QDoubleSpinBox, QSpinBox { background-color: #21262d; color: #f0f6fc; border: 1px solid #30363d; }
            QCheckBox { color: #c9d1d9; }
            QStatusBar { color: #8b949e; border-top: 1px solid #30363d; }
        """)

        central = QtWidgets.QWidget()
        self.win.setCentralWidget(central)
        main_layout = QtWidgets.QHBoxLayout(central)
        main_layout.setContentsMargins(4, 4, 4, 4)
        main_layout.setSpacing(6)

        # ── Range-Doppler plot ──
        self.plot_w = pg.PlotWidget()
        self.plot_w.setLabel('bottom', 'Velocity', units='m/s')
        self.plot_w.setLabel('left', 'Range', units='m')
        self.plot_w.showGrid(x=True, y=True, alpha=0.15)
        self.plot_w.setXRange(-10, 10, padding=0)
        self.plot_w.setYRange(0, cfg.max_range, padding=0)

        # Heatmap image
        self.rd_img = pg.ImageItem()
        try:
            _cmap = pg.colormap.get('inferno')
        except Exception:
            _cmap = pg.colormap.get('CET-L8')
        self.rd_img.setLookupTable(_cmap.getLookupTable(nPts=256))
        self.rd_img.setLevels([0, cfg.display_range_db])
        self._update_image_transform()
        self.plot_w.addItem(self.rd_img)

        # Persist: historical detections (same shape, dimmed)
        self.hist_det_plot = pg.ScatterPlotItem(
            symbol='star', size=18,
            pen=pg.mkPen(255, 255, 255, 60, width=1.5),
            brush=pg.mkBrush(255, 50, 30, 60),
        )
        self.plot_w.addItem(self.hist_det_plot)

        # Persist: stale/dead track markers (same size, dimmed)
        self.stale_trk_plot = pg.ScatterPlotItem(
            symbol='o', size=14,
            pen=pg.mkPen(0, 200, 255, 80, width=2),
            brush=pg.mkBrush(0, 0, 0, 0),
        )
        self.plot_w.addItem(self.stale_trk_plot)

        # Trail lines (pre-allocated pool)
        _trail_pen = pg.mkPen(color=(0, 200, 255, 100), width=1.5)
        self.trail_lines = []
        for _ in range(MAX_TRAILS):
            _line = pg.PlotDataItem(pen=_trail_pen)
            self.plot_w.addItem(_line)
            self.trail_lines.append(_line)

        # Tentative track markers (small yellow diamonds)
        self.tent_trk_plot = pg.ScatterPlotItem(
            symbol='d', size=10,
            pen=pg.mkPen(255, 200, 50, width=1.5),
            brush=pg.mkBrush(255, 200, 50, 60),
        )
        self.plot_w.addItem(self.tent_trk_plot)

        # CFAR detection markers (bright stars)
        self.det_plot = pg.ScatterPlotItem(
            symbol='star', size=18,
            pen=pg.mkPen('w', width=1.5),
            brush=pg.mkBrush(255, 50, 30),
            name='Detections',
        )
        self.plot_w.addItem(self.det_plot)

        # Track markers (cyan open circles)
        self.trk_plot = pg.ScatterPlotItem(
            symbol='o', size=14,
            pen=pg.mkPen(0, 200, 255, width=2),
            brush=pg.mkBrush(0, 0, 0, 0),
            name='Tracks',
        )
        self.plot_w.addItem(self.trk_plot)

        # Detection labels on map (pool of 15, matching det table limit)
        self.det_labels = []
        for _ in range(15):
            label = pg.TextItem(color=(255, 100, 80), anchor=(0, 1))
            label.setFont(QtGui.QFont('Consolas', 9))
            label.setVisible(False)
            self.plot_w.addItem(label)
            self.det_labels.append(label)

        # Track ID labels on map (pool of 20, matching max_tracks)
        self.trk_labels = []
        for _ in range(20):
            label = pg.TextItem(color=(0, 200, 255), anchor=(0, 1))
            label.setFont(QtGui.QFont('Consolas', 9))
            label.setVisible(False)
            self.plot_w.addItem(label)
            self.trk_labels.append(label)

        self.plot_w.addLegend(offset=(10, 10))
        main_layout.addWidget(self.plot_w, stretch=3)

        # ── Right panel: info + controls ──
        right_panel = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(6)

        self.info_label = QtWidgets.QLabel("")
        self.info_label.setFont(QtGui.QFont('Consolas', 11))
        self.info_label.setWordWrap(False)
        self.info_label.setMinimumWidth(300)
        self.info_label.setStyleSheet(
            "padding: 10px; background: #161b22; border: 1px solid #30363d; border-radius: 4px;"
        )
        try:
            self.info_label.setAlignment(QtCore.Qt.AlignTop | QtCore.Qt.AlignLeft)
        except AttributeError:
            self.info_label.setAlignment(
                QtCore.Qt.AlignmentFlag.AlignTop | QtCore.Qt.AlignmentFlag.AlignLeft
            )
        right_layout.addWidget(self.info_label, stretch=1)

        # Parameter controls
        self.params = Parameter.create(name='Controls', type='group', children=[
            {'name': 'max_range', 'title': 'Range (m)', 'type': 'list',
             'limits': [20, 50, 100], 'value': int(cfg.max_range)},
            {'name': 'rx_gain', 'title': 'Rx Gain (dB)', 'type': 'int',
             'value': int(cfg.rx_gain), 'limits': (0, 70), 'step': 5},
            {'name': 'chirps', 'title': 'Chirps', 'type': 'list',
             'limits': [64, 128, 256, 512], 'value': cfg.num_chirps},
            {'name': 'cfar_bias', 'title': 'CFAR Bias (dB)', 'type': 'float',
             'value': cfg.cfar_bias_db, 'limits': (5, 30), 'step': 0.5},
            {'name': 'min_cluster', 'title': 'Min Cluster', 'type': 'int',
             'value': cfg.cfar_min_cluster, 'limits': (1, 50), 'step': 1},
            {'name': 'min_range', 'title': 'Min Range (m)', 'type': 'float',
             'value': cfg.min_range, 'limits': (0, 10), 'step': 0.5},
            {'name': 'persist', 'title': 'Persist', 'type': 'bool', 'value': True},
            {'name': 'heatmap', 'title': 'Heatmap', 'type': 'bool', 'value': True},
            {'name': 'display_db', 'title': 'Display Range (dB)', 'type': 'float',
             'value': cfg.display_range_db, 'limits': (5, 60), 'step': 1},
        ])
        self.params.child('heatmap').sigValueChanged.connect(
            lambda _, v: self.params.child('display_db').show(v))
        ptree = ParameterTree(showHeader=False)
        ptree.setParameters(self.params)
        ptree.setMaximumHeight(290)
        right_layout.addWidget(ptree)

        main_layout.addWidget(right_panel, stretch=1)

        # Status bar
        self.status_bar = self.win.statusBar()
        self._status_perf_state = None
        self._status_styles = {
            'ok': "color: #8b949e;",
            'warn': "color: rgb(255, 200, 50);",
            'over': "color: rgb(255, 50, 30);",
        }
        self.status_bar.setStyleSheet(self._status_styles['ok'])

        # ── Persist state ──
        self._trail_cache = {}
        self._persist_dets = []
        self._persist_tracks = {}
        self._hist_style_cache = {}
        self._trail_style_cache = {}
        self._trail_active_pen = pg.mkPen(color=(0, 200, 255, 150), width=2.5)

        # ── Keyboard shortcuts ──
        QtWidgets.QShortcut(QtGui.QKeySequence('P'), self.win,
            lambda: self.params.child('persist').setValue(
                not self.params.child('persist').value()))
        QtWidgets.QShortcut(QtGui.QKeySequence('R'), self.win,
            lambda: self.plot_w.autoRange())
        QtWidgets.QShortcut(QtGui.QKeySequence('H'), self.win,
            lambda: self.params.child('heatmap').setValue(
                not self.params.child('heatmap').value()))

        # ── Frame state ──
        self.frame_count = 0
        self.t_start = time.time()
        self._last_frame_start = None
        self._fps_inst = 0.0
        self._fps_ema = 0.0
        self._ema_alpha = 0.2
        self._perf = {
            'capture_ms': 0.0,
            'process_ms': 0.0,
            'log_ms': 0.0,
            'ui_ms': 0.0,
            'total_ms': 0.0,
        }
        self.all_data = []
        self.det_log = open(cfg.det_log_path, 'w')
        self.trk_log = open(cfg.trk_log_path, 'w')
        print(f"Logging to {cfg.det_log_path}, {cfg.trk_log_path}")
        print("CTRL+C to stop | P = persist | H = heatmap | R = reset view")

    @staticmethod
    def _age_color(age_s):
        """Color by age: <10s red, <60s yellow, >=60s grey."""
        if age_s < 10:
            return (255, 50, 30)
        elif age_s < 60:
            return (255, 200, 50)
        else:
            return (140, 140, 140)

    def _update_image_transform(self):
        """Recompute the image transform from current config axes."""
        cfg = self.cfg
        n_doppler = max(1, len(cfg.axes.get('velocity_axis', [])))
        n_range = max(1, len(cfg.range_crop_idx))
        vel_min = -cfg.axes['max_velocity']
        vel_span = 2 * cfg.axes['max_velocity']
        rng_min = float(cfg.range_axis_cropped.min()) if len(cfg.range_axis_cropped) > 0 else 0
        rng_span = (float(cfg.range_axis_cropped.max() - cfg.range_axis_cropped.min())
                    if len(cfg.range_axis_cropped) > 1 else 1)
        tr = QtGui.QTransform()
        tr.translate(vel_min, rng_min)
        tr.scale(vel_span / n_doppler, rng_span / n_range)
        self.rd_img.setTransform(tr)

    def update(self):
        """Main loop body — called by QTimer each frame."""
        cfg = self.cfg
        hw = self.hw
        frame_t0 = time.perf_counter()

        if self._last_frame_start is not None:
            dt = frame_t0 - self._last_frame_start
            if dt > 0:
                self._fps_inst = 1.0 / dt
                if self._fps_ema <= 0:
                    self._fps_ema = self._fps_inst
                else:
                    a = self._ema_alpha
                    self._fps_ema = (1.0 - a) * self._fps_ema + a * self._fps_inst
        self._last_frame_start = frame_t0

        try:
            # Read runtime parameters from controls
            new_chirps = int(self.params.child('chirps').value())
            axes_changed = apply_chirp_config(cfg, hw, new_chirps)
            if axes_changed:
                self._update_image_transform()
                self._trail_cache.clear()
                self._persist_tracks.clear()
                self.trk_plot.setData([], [])
                self.tent_trk_plot.setData([], [])

            # Range preset
            new_range = int(self.params.child('max_range').value())
            if new_range != int(cfg.max_range):
                cfg.max_range = float(new_range)
                cfg.update_range_crop()
                self._update_image_transform()
                self.plot_w.setYRange(0, cfg.max_range, padding=0)

            # Rx gain (live hardware update)
            new_rx_gain = int(self.params.child('rx_gain').value())
            if new_rx_gain != int(cfg.rx_gain):
                cfg.rx_gain = new_rx_gain
                ccal0, ccal1 = getattr(hw, 'channel_cal', (0.0, 0.0))
                hw.sdr.rx_hardwaregain_chan0 = int(cfg.rx_gain + ccal0)
                hw.sdr.rx_hardwaregain_chan1 = int(cfg.rx_gain + ccal1)

            cfg.cfar_bias_db = self.params.child('cfar_bias').value()
            cfg.cfar_min_cluster = int(self.params.child('min_cluster').value())
            cfg.min_range = self.params.child('min_range').value()
            new_db = self.params.child('display_db').value()
            persist = self.params.child('persist').value()

            if new_db != cfg.display_range_db:
                cfg.display_range_db = new_db
                self.rd_img.setLevels([0, cfg.display_range_db])

            # Capture + process
            cap_t0 = time.perf_counter()
            chan0, chan1 = get_radar_data(hw)
            cap_t1 = time.perf_counter()
            self._perf['capture_ms'] = (cap_t1 - cap_t0) * 1000.0

            if cfg.save_data:
                self.all_data.append((chan0.copy(), chan1.copy()))

            proc_t0 = time.perf_counter()
            rd_display, targets, tracks, diag = process_frame(chan0, chan1, cfg, hw)
            proc_t1 = time.perf_counter()
            self._perf['process_ms'] = (proc_t1 - proc_t0) * 1000.0
            all_tracks = hw.tracker.get_all_tracks()
            tentative = [t for t in all_tracks if not t.confirmed]
            self.frame_count += 1

            # Log
            log_t0 = time.perf_counter()
            elapsed = time.time() - self.t_start
            dt_est = elapsed / self.frame_count if self.frame_count > 0 else 0.3
            # Detection log (only when detections exist)
            if targets:
                self.det_log.write(json.dumps({
                    'frame': self.frame_count,
                    'time_s': round(elapsed, 3),
                    'dt_s': round(dt_est, 4),
                    'noise_floor_db': round(diag['noise_floor_db'], 1),
                    'cfar_cells': diag['cfar_cells'],
                    'raw_clusters': diag['raw_clusters'],
                    'cfar_bias_db': cfg.cfar_bias_db,
                    'min_cluster': cfg.cfar_min_cluster,
                    'min_range_m': cfg.min_range,
                    'num_chirps': cfg.num_chirps,
                    'mti_mode': cfg.mti_mode,
                    'rx_gain': int(cfg.rx_gain),
                    'detections': [{
                        'id': i,
                        'range_m': round(t['range_m'], 3),
                        'velocity_mps': round(t['velocity_mps'], 4),
                        'power_db': round(t['power_db'], 1),
                        'angle_deg': round(t.get('angle_deg', 0), 2),
                        'pixel_count': t['pixel_count'],
                    } for i, t in enumerate(targets, 1)],
                }) + '\n')

            # Track log (only when tracks exist)
            if tracks or tentative:
                self.trk_log.write(json.dumps({
                    'frame': self.frame_count,
                    'time_s': round(elapsed, 3),
                    'dt_s': round(dt_est, 4),
                    'num_detections': len(targets),
                    'confirmed': [{
                        'id': int(t.track_id),
                        'range_m': round(float(t.state[0]), 3),
                        'vel_mps': round(float(t.state[1]), 4),
                        'az_deg': round(float(t.state[2]), 2),
                        'hits': int(t.hits),
                        'misses': int(t.misses),
                        'age': int(t.age),
                        'cov_rng': round(float(t.covariance[0, 0]), 4),
                        'cov_vel': round(float(t.covariance[1, 1]), 4),
                        'cov_az': round(float(t.covariance[2, 2]), 4),
                    } for t in tracks],
                    'tentative': [{
                        'id': int(t.track_id),
                        'range_m': round(float(t.state[0]), 3),
                        'vel_mps': round(float(t.state[1]), 4),
                        'az_deg': round(float(t.state[2]), 2),
                        'hits': int(t.hits),
                        'misses': int(t.misses),
                        'age': int(t.age),
                    } for t in tentative],
                }) + '\n')

            if self.frame_count % 10 == 0:
                self.det_log.flush()
                self.trk_log.flush()
            log_t1 = time.perf_counter()
            self._perf['log_ms'] = (log_t1 - log_t0) * 1000.0

            # ── Update display ──
            ui_t0 = time.perf_counter()

            # Heatmap (display-only — detections/tracks come from CFAR, not this image)
            show_heatmap = self.params.child('heatmap').value()
            self.rd_img.setVisible(show_heatmap)
            if show_heatmap:
                self.rd_img.setImage(rd_display.T, autoLevels=False)

            # CFAR detection markers
            if targets:
                self.det_plot.setData(
                    x=[t['velocity_mps'] for t in targets],
                    y=[t['range_m'] for t in targets],
                )
            else:
                self.det_plot.setData([], [])

            # Persist: historical detection stars (age-colored)
            if persist and self._persist_dets:
                hist_spots = []
                for fr, r, v, p, a in self._persist_dets:
                    if fr == self.frame_count:
                        continue
                    age = (self.frame_count - fr) * dt_est
                    c = self._age_color(age)
                    if c not in self._hist_style_cache:
                        self._hist_style_cache[c] = (
                            pg.mkPen(*c, 180, width=1.5),
                            pg.mkBrush(*c, 120),
                        )
                    pen, brush = self._hist_style_cache[c]
                    hist_spots.append({
                        'pos': (v, r), 'symbol': 'star', 'size': 18,
                        'pen': pen,
                        'brush': brush,
                    })
                if hist_spots:
                    self.hist_det_plot.setData(spots=hist_spots)
                else:
                    self.hist_det_plot.setData([], [])
            else:
                self.hist_det_plot.setData([], [])

            # Detection labels on map (color matches star age)
            dlbl_idx = 0
            for i, t in enumerate(targets[:8], 1):
                if dlbl_idx >= len(self.det_labels):
                    break
                self.det_labels[dlbl_idx].setText(f"D{i}")
                self.det_labels[dlbl_idx].setPos(t['velocity_mps'], t['range_m'])
                self.det_labels[dlbl_idx].setColor(self._age_color(0))
                self.det_labels[dlbl_idx].setVisible(True)
                dlbl_idx += 1
            if persist:
                for fr, r, v, p, a in self._persist_dets:
                    if fr == self.frame_count:
                        continue
                    if dlbl_idx >= len(self.det_labels):
                        break
                    age = (self.frame_count - fr) * dt_est
                    c = self._age_color(age)
                    self.det_labels[dlbl_idx].setText(f"D{dlbl_idx + 1}")
                    self.det_labels[dlbl_idx].setPos(v, r)
                    self.det_labels[dlbl_idx].setColor((*c, 180))
                    self.det_labels[dlbl_idx].setVisible(True)
                    dlbl_idx += 1
            for i in range(dlbl_idx, len(self.det_labels)):
                self.det_labels[i].setVisible(False)

            # Cache trail data for all active tracks
            active_ids = {trk.track_id for trk in tracks}
            for trk in tracks:
                if len(trk.history) > 1:
                    self._trail_cache[trk.track_id] = np.array(trk.history)

            # In non-persist mode, prune dead tracks from cache
            if not persist:
                self._trail_cache = {
                    k: v for k, v in self._trail_cache.items() if k in active_ids
                }

            # Track markers: active tracks
            if tracks:
                self.trk_plot.setData(
                    x=[float(t.state[1]) for t in tracks],
                    y=[float(t.state[0]) for t in tracks],
                )
            else:
                self.trk_plot.setData([], [])

            # Track ID labels on map (active = bright, stale = dim)
            label_idx = 0
            for trk in tracks:
                if label_idx >= len(self.trk_labels):
                    break
                self.trk_labels[label_idx].setText(f"T{trk.track_id}")
                self.trk_labels[label_idx].setPos(float(trk.state[1]), float(trk.state[0]))
                self.trk_labels[label_idx].setColor((0, 200, 255))
                self.trk_labels[label_idx].setVisible(True)
                label_idx += 1
            if persist:
                for tid, t in self._persist_tracks.items():
                    if not t['active'] and label_idx < len(self.trk_labels):
                        self.trk_labels[label_idx].setText(f"T{tid}")
                        self.trk_labels[label_idx].setPos(t['v'], t['r'])
                        self.trk_labels[label_idx].setColor((0, 200, 255, 80))
                        self.trk_labels[label_idx].setVisible(True)
                        label_idx += 1
            for i in range(label_idx, len(self.trk_labels)):
                self.trk_labels[i].setVisible(False)

            # Persist: stale/dead track markers
            if persist and self._persist_tracks:
                stale = [(t['v'], t['r']) for t in self._persist_tracks.values()
                         if not t['active']]
                if stale:
                    self.stale_trk_plot.setData(
                        x=[s[0] for s in stale], y=[s[1] for s in stale])
                else:
                    self.stale_trk_plot.setData([], [])
            else:
                self.stale_trk_plot.setData([], [])

            # Trail lines (active = bold cyan, inactive = age-colored)
            now_trail = time.time()
            trail_idx = 0
            for tid, hist in self._trail_cache.items():
                if trail_idx >= MAX_TRAILS:
                    break
                if tid in active_ids:
                    pen = self._trail_active_pen
                else:
                    pinfo = self._persist_tracks.get(tid)
                    age_s = (now_trail - pinfo['last_seen']) if pinfo else 0
                    c = self._age_color(age_s)
                    if c not in self._trail_style_cache:
                        self._trail_style_cache[c] = pg.mkPen(*c, 100, width=1.5)
                    pen = self._trail_style_cache[c]
                self.trail_lines[trail_idx].setPen(pen)
                self.trail_lines[trail_idx].setData(x=hist[:, 1], y=hist[:, 0])
                trail_idx += 1
            for i in range(trail_idx, MAX_TRAILS):
                self.trail_lines[i].setData([], [])

            # Tentative track markers (unconfirmed — yellow diamonds)
            if tentative:
                self.tent_trk_plot.setData(
                    x=[float(t.state[1]) for t in tentative],
                    y=[float(t.state[0]) for t in tentative],
                )
            else:
                self.tent_trk_plot.setData([], [])

            # ── Info panel ──
            fps_avg = self.frame_count / elapsed if elapsed > 0 else 0
            _W = 36
            _sec = lambda s: f"\u2500\u2500 {s} " + "\u2500" * max(0, _W - len(s) - 4)

            # Update persist history
            if persist:
                for t in targets:
                    self._persist_dets.append((
                        self.frame_count, t['range_m'], t['velocity_mps'],
                        t['power_db'], t.get('angle_deg', 0.0),
                    ))
                if len(self._persist_dets) > 10:
                    self._persist_dets[:] = self._persist_dets[-10:]
                active_ids = {trk.track_id for trk in tracks}
                now = time.time()
                for trk in tracks:
                    first = self._persist_tracks.get(
                        trk.track_id, {}).get('first_seen', now)
                    self._persist_tracks[trk.track_id] = {
                        'r': float(trk.state[0]), 'v': float(trk.state[1]),
                        'az': float(trk.state[2]), 'first_seen': first,
                        'last_seen': now, 'active': True,
                    }
                for tid in self._persist_tracks:
                    if tid not in active_ids:
                        self._persist_tracks[tid]['active'] = False
                # Limit persist tracks to last 10
                if len(self._persist_tracks) > 10:
                    for k in list(self._persist_tracks.keys())[:-10]:
                        del self._persist_tracks[k]
            else:
                self._persist_dets.clear()
                self._persist_tracks.clear()

            peak_power = max((t['power_db'] for t in targets), default=None)

            # ── STATUS ──
            lines = [
                _sec('STATUS'),
                f" Frame {self.frame_count:<8} FPS {self._fps_ema:.1f} (avg {fps_avg:.1f})",
                f" Noise {diag['noise_floor_db']:.1f} dB",
                f" MTI {cfg.mti_mode:<10} Chirps {cfg.num_chirps}",
                f" Rng res {cfg.axes['range_res']:.2f} m"
                f"   Vel res {cfg.axes['velocity_res']:.2f} m/s",
                "",
                _sec('PIPELINE'),
                f" CFAR  {diag['cfar_cells']} cells  {diag['raw_clusters']} clusters",
                f" Det {len(targets):<5} Tracks {len(tracks)}"
                f"   Tent {len(tentative)}",
                f" Peak {f'{peak_power:.0f} dB' if peak_power is not None else chr(0x2014)}",
            ]

            # ── DETECTIONS ──
            det_rows = []
            for t in targets[:8]:
                det_rows.append((t['range_m'], t['velocity_mps'], t['power_db'],
                                 t.get('angle_deg', 0.0), 0.0, True))
            if persist:
                hist = [(r, v, p, a, (self.frame_count - fr) * dt_est, False)
                        for fr, r, v, p, a in self._persist_dets
                        if fr != self.frame_count]
                det_rows.extend(hist[-(10 - len(det_rows)):])

            _dfmt = " {:1s}{:>4s} {:>6s} {:>9s} {:>6s} {:>6s} {:>5s}"
            if det_rows:
                lines.append("")
                lines.append(_sec('DETECTIONS'))
                lines.append(_dfmt.format(" ", "#", "Rng", "Vel", "Pwr", "Az", "Age"))
                for idx, (r, v, p, az, age, live) in enumerate(det_rows, 1):
                    mark = "\u25b8" if live else " "
                    lines.append(_dfmt.format(mark, f"D{idx}", f"{r:.1f}m",
                        f"{v:+.2f}m/s", f"{p:.0f}dB", f"{az:+.1f}\u00b0",
                        f"{age:.1f}"))

            # ── TRACKS ──
            trk_rows = []
            now_t = time.time()
            if persist and self._persist_tracks:
                for tid in sorted(self._persist_tracks):
                    t = self._persist_tracks[tid]
                    age_s = now_t - t['first_seen']
                    dur_s = t.get('last_seen', now_t) - t['first_seen']
                    trk_rows.append((tid, t['r'], t['v'], t['az'], dur_s, age_s, t['active']))
            else:
                for trk in tracks[:8]:
                    dur_est = trk.age * dt_est
                    trk_rows.append((trk.track_id, float(trk.state[0]),
                                     float(trk.state[1]), float(trk.state[2]),
                                     dur_est, dur_est, True))

            _tfmt = " {:1s}{:>4s} {:>6s} {:>9s} {:>5s} {:>5s} {:>5s}"
            if trk_rows:
                lines.append("")
                lines.append(_sec('TRACKS'))
                lines.append(_tfmt.format(" ", "ID", "Rng", "Vel", "Az", "Dur", "Age"))
                for tid, r, v, az, dur_s, age_s, active in trk_rows[:10]:
                    mark = "\u25cf" if active else "\u25cb"
                    lines.append(_tfmt.format(mark, str(tid), f"{r:.1f}m",
                        f"{v:+.2f}m/s", f"{az:+.0f}\u00b0", f"{dur_s:.1f}", f"{age_s:.0f}s"))

            self.info_label.setText('\n'.join(lines))
            ui_t1 = time.perf_counter()
            self._perf['ui_ms'] = (ui_t1 - ui_t0) * 1000.0
            self._perf['total_ms'] = (ui_t1 - frame_t0) * 1000.0

            # Status bar
            flags = []
            if persist: flags.append("PERSIST")
            if not show_heatmap: flags.append("NO-HEATMAP")
            cpi_ms = max(cfg.frame_time * 1000.0, 1e-6)
            cpi_util = self._perf['total_ms'] / cpi_ms

            if cpi_util > 1.0:
                perf_state = 'over'
            elif cpi_util > 0.8:
                perf_state = 'warn'
            else:
                perf_state = 'ok'
            if perf_state != self._status_perf_state:
                self.status_bar.setStyleSheet(self._status_styles[perf_state])
                self._status_perf_state = perf_state

            self.status_bar.showMessage(
                f"FPS i/e/a {self._fps_inst:.1f}/{self._fps_ema:.1f}/{fps_avg:.1f}  "
                f"| ms C{self._perf['capture_ms']:.0f} P{self._perf['process_ms']:.0f} "
                f"U{self._perf['ui_ms']:.0f} L{self._perf['log_ms']:.0f} T{self._perf['total_ms']:.0f}  "
                f"| CPIx {cpi_util:.1f}  | R{int(cfg.max_range)}m Rx{int(cfg.rx_gain)}dB"
                f" {' '.join(flags)}")

        except Exception as e:
            import traceback
            print(f"Frame error: {e}")
            traceback.print_exc()

    def run(self):
        """Start the timer and enter the Qt event loop."""
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update)
        self.timer.start(0)

        self.win.show()

        try:
            self.app.exec_()
        except (KeyboardInterrupt, SystemExit):
            pass

    def cleanup(self):
        """Stop timer, close log, release hardware, optionally save data."""
        self.timer.stop()
        self.det_log.close()
        self.trk_log.close()
        print(f"\nProcessed {self.frame_count} frames")
        print(f"  {self.cfg.det_log_path}  {self.cfg.trk_log_path}")

        hw_cleanup(self.hw)

        if self.cfg.save_data and self.all_data:
            np.save(self.cfg.save_file, self.all_data, allow_pickle=True)
            config = {
                'sample_rate': self.cfg.sample_rate,
                'signal_freq': self.cfg.signal_freq,
                'output_freq': self.cfg.output_freq,
                'num_chirps': self.cfg.num_chirps,
                'chirp_BW': self.cfg.chirp_BW,
                'ramp_time_s': self.cfg.ramp_time_s,
                'frame_length_ms': self.cfg.PRI_ms,
                'mti_mode': self.cfg.mti_mode,
                'range_window': self.cfg.range_window,
                'doppler_window': self.cfg.doppler_window,
            }
            np.save(self.cfg.save_file.replace('.npy', '_config.npy'),
                    config, allow_pickle=True)
            print(f"Saved {len(self.all_data)} frames to {self.cfg.save_file}")
