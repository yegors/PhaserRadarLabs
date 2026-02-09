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

        # Persist: historical detection dots (faded red)
        self.hist_det_plot = pg.ScatterPlotItem(
            symbol='o', size=4, pen=None,
            brush=pg.mkBrush(255, 80, 80, 50),
        )
        self.plot_w.addItem(self.hist_det_plot)

        # Persist: stale/dead track markers (dim open circles)
        self.stale_trk_plot = pg.ScatterPlotItem(
            symbol='o', size=10,
            pen=pg.mkPen(0, 200, 255, 80, width=1),
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
            {'name': 'chirps', 'title': 'Chirps', 'type': 'list',
             'limits': [64, 128, 256], 'value': cfg.num_chirps},
            {'name': 'cfar_bias', 'title': 'CFAR Bias (dB)', 'type': 'float',
             'value': cfg.cfar_bias_db, 'limits': (5, 30), 'step': 0.5},
            {'name': 'min_cluster', 'title': 'Min Cluster', 'type': 'int',
             'value': cfg.cfar_min_cluster, 'limits': (1, 50), 'step': 1},
            {'name': 'min_range', 'title': 'Min Range (m)', 'type': 'float',
             'value': cfg.min_range, 'limits': (0, 10), 'step': 0.5},
            {'name': 'display_db', 'title': 'Display Range (dB)', 'type': 'float',
             'value': cfg.display_range_db, 'limits': (5, 60), 'step': 1},
            {'name': 'persist', 'title': 'Persist', 'type': 'bool', 'value': True},
        ])
        ptree = ParameterTree(showHeader=False)
        ptree.setParameters(self.params)
        ptree.setMaximumHeight(200)
        right_layout.addWidget(ptree)

        main_layout.addWidget(right_panel, stretch=1)

        # Status bar
        self.status_bar = self.win.statusBar()

        # ── Persist state ──
        self._trail_cache = {}
        self._persist_dets = []
        self._persist_tracks = {}

        # ── Keyboard shortcuts ──
        QtWidgets.QShortcut(QtGui.QKeySequence('P'), self.win,
            lambda: self.params.child('persist').setValue(
                not self.params.child('persist').value()))
        QtWidgets.QShortcut(QtGui.QKeySequence('R'), self.win,
            lambda: self.plot_w.autoRange())

        # ── Frame state ──
        self.frame_count = 0
        self.t_start = time.time()
        self.all_data = []
        self.log_f = open(cfg.log_file_path, 'w')
        print(f"Logging to {cfg.log_file_path}")
        print("CTRL+C to stop | P = toggle persist | R = reset view")

    def _update_image_transform(self):
        """Recompute the image transform from current config axes."""
        cfg = self.cfg
        vel_min = -cfg.axes['max_velocity']
        vel_span = 2 * cfg.axes['max_velocity']
        rng_min = float(cfg.range_axis_cropped.min()) if len(cfg.range_axis_cropped) > 0 else 0
        rng_span = (float(cfg.range_axis_cropped.max() - cfg.range_axis_cropped.min())
                    if len(cfg.range_axis_cropped) > 1 else 1)
        tr = QtGui.QTransform()
        tr.translate(vel_min, rng_min)
        tr.scale(vel_span / cfg.num_chirps, rng_span / len(cfg.range_crop_idx))
        self.rd_img.setTransform(tr)

    def update(self):
        """Main loop body — called by QTimer each frame."""
        cfg = self.cfg
        hw = self.hw

        try:
            # Read runtime parameters from controls
            new_chirps = int(self.params.child('chirps').value())
            axes_changed = apply_chirp_config(cfg, hw, new_chirps)
            if axes_changed:
                self._update_image_transform()

            cfg.cfar_bias_db = self.params.child('cfar_bias').value()
            cfg.cfar_min_cluster = int(self.params.child('min_cluster').value())
            cfg.min_range = self.params.child('min_range').value()
            new_db = self.params.child('display_db').value()
            persist = self.params.child('persist').value()

            if new_db != cfg.display_range_db:
                cfg.display_range_db = new_db
                self.rd_img.setLevels([0, cfg.display_range_db])

            # Capture + process
            chan0, chan1 = get_radar_data(hw)

            if cfg.save_data:
                self.all_data.append((chan0.copy(), chan1.copy()))

            rd_display, targets, tracks, diag = process_frame(chan0, chan1, cfg, hw)
            self.frame_count += 1

            # Log
            elapsed = time.time() - self.t_start
            log_entry = {
                'frame': self.frame_count,
                'time_s': round(elapsed, 3),
                'cfar_cells': diag['cfar_cells'],
                'raw_clusters': diag['raw_clusters'],
                'num_detections': len(targets),
                'num_tracks': len(tracks),
                'detections': [{
                    'range_m': round(t['range_m'], 2),
                    'velocity_mps': round(t['velocity_mps'], 3),
                    'power_db': round(t['power_db'], 1),
                    'pixels': t['pixel_count'],
                    'angle_deg': round(t.get('angle_deg', 0), 1),
                } for t in targets],
                'tracks': [{
                    'id': int(t.track_id),
                    'range_m': round(float(t.state[0]), 2),
                    'vel_mps': round(float(t.state[1]), 3),
                    'az_deg': round(float(t.state[2]), 1),
                    'age': int(t.age),
                    'hits': int(t.hits),
                    'misses': int(t.misses),
                } for t in tracks],
            }
            self.log_f.write(json.dumps(log_entry) + '\n')
            self.log_f.flush()

            # ── Update display ──

            # Heatmap
            self.rd_img.setImage(rd_display.T, autoLevels=False)

            # CFAR detection markers
            if targets:
                self.det_plot.setData(
                    x=[t['velocity_mps'] for t in targets],
                    y=[t['range_m'] for t in targets],
                )
            else:
                self.det_plot.setData([], [])

            # Persist: historical detection dots
            if persist and self._persist_dets:
                hist = [(v, r) for fr, r, v, p, a in self._persist_dets
                        if fr != self.frame_count]
                if hist:
                    self.hist_det_plot.setData(
                        x=[h[0] for h in hist], y=[h[1] for h in hist])
                else:
                    self.hist_det_plot.setData([], [])
            else:
                self.hist_det_plot.setData([], [])

            # Cache trail data for all active tracks
            for trk in tracks:
                if len(trk.history) > 1:
                    self._trail_cache[trk.track_id] = np.array(trk.history)

            # In non-persist mode, prune dead tracks from cache
            if not persist:
                active_ids = {trk.track_id for trk in tracks}
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

            # Trail lines (active + persisted dead tracks)
            trail_items = list(self._trail_cache.values())
            for i, hist in enumerate(trail_items[:MAX_TRAILS]):
                self.trail_lines[i].setData(x=hist[:, 1], y=hist[:, 0])
            for i in range(len(trail_items), MAX_TRAILS):
                self.trail_lines[i].setData([], [])

            # ── Info panel ──
            fps = self.frame_count / elapsed if elapsed > 0 else 0
            _W = 28
            _sec = lambda s: f"\u2500\u2500 {s} " + "\u2500" * max(0, _W - len(s) - 4)

            # Update persist history
            if persist:
                for t in targets:
                    self._persist_dets.append((
                        self.frame_count, t['range_m'], t['velocity_mps'],
                        t['power_db'], t.get('angle_deg', 0.0),
                    ))
                if len(self._persist_dets) > 30:
                    self._persist_dets[:] = self._persist_dets[-30:]
                active_ids = {trk.track_id for trk in tracks}
                now = time.time()
                for trk in tracks:
                    first = self._persist_tracks.get(
                        trk.track_id, {}).get('first_seen', now)
                    self._persist_tracks[trk.track_id] = {
                        'r': float(trk.state[0]), 'v': float(trk.state[1]),
                        'az': float(trk.state[2]), 'first_seen': first,
                        'active': True,
                    }
                for tid in self._persist_tracks:
                    if tid not in active_ids:
                        self._persist_tracks[tid]['active'] = False
            else:
                self._persist_dets.clear()
                self._persist_tracks.clear()

            # ── STATUS ──
            lines = [
                _sec('STATUS'),
                f" Frame {self.frame_count:<10}FPS {fps:.1f}",
                f" MTI   {cfg.mti_mode}",
                f" CFAR  {diag['cfar_cells']} cells  {diag['raw_clusters']} cls",
                f"       {len(targets)} det   {len(tracks)} tracks",
            ]

            # ── DETECTIONS ──
            det_rows = []
            for t in targets[:8]:
                det_rows.append((t['range_m'], t['velocity_mps'], t['power_db'],
                                 t.get('angle_deg', 0.0), True))
            if persist:
                hist = [(r, v, p, a, False) for fr, r, v, p, a in self._persist_dets
                        if fr != self.frame_count]
                det_rows.extend(hist[-(15 - len(det_rows)):])

            _dfmt = "{:1s}{:>6s} {:>9s} {:>6s} {:>6s}"
            if det_rows:
                lines.append("")
                lines.append(_sec('DETECTIONS'))
                lines.append(_dfmt.format(" ", "Rng", "Vel", "Az", "Pwr"))
                for r, v, p, az, live in det_rows:
                    mark = "\u25b8" if live else " "
                    lines.append(_dfmt.format(mark, f"{r:.1f}m",
                        f"{v:+.2f}m/s", f"{az:+.1f}\u00b0", f"{p:.0f}dB"))

            # ── TRACKS ──
            trk_rows = []
            now_t = time.time()
            if persist and self._persist_tracks:
                for tid in sorted(self._persist_tracks):
                    t = self._persist_tracks[tid]
                    age_s = now_t - t['first_seen']
                    trk_rows.append((tid, t['r'], t['v'], t['az'], age_s, t['active']))
            else:
                dt_est = elapsed / self.frame_count if self.frame_count > 0 else 0.3
                for trk in tracks[:8]:
                    trk_rows.append((trk.track_id, float(trk.state[0]),
                                     float(trk.state[1]), float(trk.state[2]),
                                     trk.age * dt_est, True))

            _tfmt = " {:1s}{:>4s} {:>6s} {:>9s} {:>6s} {:>6s}"
            if trk_rows:
                lines.append("")
                lines.append(_sec('TRACKS'))
                lines.append(_tfmt.format(" ", "ID", "Rng", "Vel", "Az", "Age"))
                for tid, r, v, az, age_s, active in trk_rows[:10]:
                    mark = "\u25cf" if active else "\u25cb"
                    lines.append(_tfmt.format(mark, str(tid), f"{r:.1f}m",
                        f"{v:+.2f}m/s", f"{az:+.1f}\u00b0", f"{age_s:.1f}s"))

            self.info_label.setText('\n'.join(lines))

            # Status bar
            p_str = "PERSIST" if persist else ""
            self.status_bar.showMessage(f"FPS: {fps:.1f}   {p_str}")

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
        self.log_f.close()
        print(f"\nProcessed {self.frame_count} frames, log saved to {self.cfg.log_file_path}")

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
