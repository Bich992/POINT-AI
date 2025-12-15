from __future__ import annotations
import os, traceback
import numpy as np

from PySide6.QtCore import QObject, Signal, QThread
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QTabWidget,
    QPushButton, QLabel, QFileDialog, QLineEdit, QTextEdit,
    QProgressBar, QComboBox, QSpinBox, QDoubleSpinBox
)

from ui.viewer_widget import PointCloudViewer
from core.stream_loaders import load_las_laz_reservoir, load_e57_sample
from core.oc_build import build_store_from_source
from core.oc_store import PointStore
from core.oc_query import load_roi, pick_lod
from core.oc_export import export_filtered_ply
from core.nl_assistant import NaturalLanguageAssistant


class LoadWorker(QObject):
    progress = Signal(float, str)
    finished = Signal(object, object, str)
    error = Signal(str)

    def __init__(self, path: str, target_points: int):
        super().__init__()
        self.path = path
        self.target_points = target_points

    def run(self):
        try:
            ext = os.path.splitext(self.path)[1].lower()

            def cb(pct, msg):
                self.progress.emit(float(pct), str(msg))

            if ext in (".las", ".laz"):
                pts, cols = load_las_laz_reservoir(self.path, self.target_points, cb)
                self.finished.emit(pts, cols, self.path)
                return

            if ext == ".e57":
                pts, cols = load_e57_sample(self.path, self.target_points, cb)
                self.finished.emit(pts, cols, self.path)
                return

            import open3d as o3d
            cb(5.0, "Caricamento (open3d)...")
            pcd = o3d.io.read_point_cloud(self.path)
            pts = np.asarray(pcd.points)
            cols = np.asarray(pcd.colors) if pcd.has_colors() else None

            if len(pts) > self.target_points:
                step = max(1, len(pts) // self.target_points)
                pts = pts[::step]
                if cols is not None:
                    cols = cols[::step]

            cb(100.0, f"Caricato {len(pts)} punti")
            self.finished.emit(pts, cols, self.path)

        except Exception as e:
            self.error.emit(str(e) + "\n" + traceback.format_exc())


class UnifiedMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PointAI v5.6")
        self.resize(1600, 900)

        self.ai = NaturalLanguageAssistant()

        root = QWidget()
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)

        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)

        # -------- TAB QUICK --------
        tab_quick = QWidget()
        self.tabs.addTab(tab_quick, "Quick")
        ql = QHBoxLayout(tab_quick)

        left = QVBoxLayout()
        ql.addLayout(left, 0)

        self.btn_load = QPushButton("Carica nuvola punti")
        self.progress = QProgressBar()
        self.status = QLabel("Pronto")

        self.max_points = QSpinBox()
        self.max_points.setRange(100_000, 10_000_000)
        self.max_points.setValue(2_000_000)

        self.ai_mode = QComboBox()
        self.ai_mode.addItems(["rules", "ollama"])

        self.ai_input = QLineEdit()
        self.ai_input.setPlaceholderText("Scrivi un comando o una frase…")

        self.log = QTextEdit()
        self.log.setReadOnly(True)

        left.addWidget(self.btn_load)
        left.addWidget(QLabel("Max punti display"))
        left.addWidget(self.max_points)
        left.addWidget(QLabel("AI mode"))
        left.addWidget(self.ai_mode)
        left.addWidget(self.progress)
        left.addWidget(self.status)
        left.addWidget(self.ai_input)
        left.addWidget(self.log, 1)

        self.viewer = PointCloudViewer()
        ql.addWidget(self.viewer, 1)

        # signals
        self.btn_load.clicked.connect(self.on_load)
        self.ai_input.returnPressed.connect(self.on_ai_command)
        self.ai_mode.currentTextChanged.connect(self.ai.set_mode)

        self._load_thread = None
        self._worker = None

    def log_msg(self, msg: str):
        self.log.append(msg)

    def on_load(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Apri nuvola punti", "",
            "Point Clouds (*.las *.laz *.e57 *.ply *.pcd *.xyz);;All files (*.*)"
        )
        if not path:
            return

        self.progress.setValue(0)
        self.status.setText("Caricamento…")
        self.btn_load.setEnabled(False)

        self._load_thread = QThread()
        self._worker = LoadWorker(path, int(self.max_points.value()))
        self._worker.moveToThread(self._load_thread)

        self._load_thread.started.connect(self._worker.run)
        self._worker.progress.connect(self.on_progress)
        self._worker.finished.connect(self.on_loaded)
        self._worker.error.connect(self.on_error)

        self._worker.finished.connect(self._load_thread.quit)
        self._worker.error.connect(self._load_thread.quit)
        self._load_thread.finished.connect(self._load_thread.deleteLater)

        self._load_thread.start()

    def on_progress(self, pct: float, msg: str):
        self.progress.setValue(int(pct))
        self.status.setText(msg)

    def on_loaded(self, pts, cols, path):
        mn = pts.min(axis=0); mx = pts.max(axis=0)
        self.log_msg(f"DEBUG bbox min={mn}, max={mx}, n={len(pts)}")
        self.viewer.set_pointcloud(pts, cols)
        self.viewer.autofit()
        self.status.setText(f"Caricato: {os.path.basename(path)}")
        self.btn_load.setEnabled(True)

    def on_error(self, msg: str):
        self.log_msg("ERRORE:\n" + msg)
        self.status.setText("Errore")
        self.btn_load.setEnabled(True)

    def on_ai_command(self):
        text = self.ai_input.text().strip()
        if not text:
            return
        self.ai_input.clear()

        cmd = self.ai.translate_to_command(text)
        if not cmd:
            self.log_msg(f"AI: non capisco → {text}")
            return

        self.log_msg(f"AI → {cmd}")
        parts = cmd.split()

        if parts[0] == "pointsize" and len(parts) == 2:
            try:
                self.viewer.set_point_size(float(parts[1]))
            except Exception:
                pass
