from __future__ import annotations
import traceback
import open3d as o3d

from PySide6.QtCore import Qt, QObject, Signal, QThread
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QLineEdit, QTextEdit, QComboBox
)

from core.io_loaders import load_pointcloud
from core.pointcloud_ops import (
    get_bounds_info, lowest_points, voxel_downsample,
    denoise_statistical, dbscan_clusters, extract_ground_ransac
)
from core.command_parser import parse_command
from core.nl_assistant import NaturalLanguageAssistant

class LoaderWorker(QObject):
    finished = Signal(object, str)   # pcd, path
    error = Signal(str)

    def __init__(self, path: str):
        super().__init__()
        self.path = path

    def run(self):
        try:
            pcd = load_pointcloud(self.path)
            self.finished.emit(pcd, self.path)
        except Exception as e:
            self.error.emit(f"{e}\n\n{traceback.format_exc()}")

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PointAI v3 - Qt")
        self.resize(980, 640)

        self.original = None
        self.current = None

        self._load_thread = None
        self._load_worker = None

        self.nl = NaturalLanguageAssistant()

        root = QWidget()
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)

        top = QHBoxLayout()
        self.btn_load = QPushButton("Carica (PLY/PCD/XYZ/LAS/LAZ/E57)")
        self.btn_view = QPushButton("Visualizza (finestra separata)")
        self.btn_save = QPushButton("Salva (PLY)")
        self.btn_save.setEnabled(False)

        self.ai_mode = QComboBox()
        self.ai_mode.addItems(["AI: Rules (offline)", "AI: Ollama (local)"])
        self.ai_mode.currentIndexChanged.connect(self.on_ai_mode_changed)

        self.ollama_model = QLineEdit("llama3.2:3b")
        self.ollama_model.editingFinished.connect(self.on_ollama_model_changed)

        top.addWidget(self.btn_load)
        top.addWidget(self.btn_view)
        top.addWidget(self.btn_save)
        top.addSpacing(10)
        top.addWidget(self.ai_mode)
        top.addWidget(self.ollama_model)
        layout.addLayout(top)

        self.status = QLabel("Pronto.")
        self.status.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout.addWidget(self.status)

        cmd_row = QHBoxLayout()
        self.cmd = QLineEdit()
        self.cmd.setPlaceholderText("Comando o frase naturale… (help per elenco)")
        self.btn_run = QPushButton("Esegui")
        cmd_row.addWidget(self.cmd)
        cmd_row.addWidget(self.btn_run)
        layout.addLayout(cmd_row)

        self.log = QTextEdit()
        self.log.setReadOnly(True)
        layout.addWidget(self.log, stretch=1)

        self.btn_load.clicked.connect(self.on_load)
        self.btn_view.clicked.connect(self.on_view)
        self.btn_run.clicked.connect(self.on_run)
        self.btn_save.clicked.connect(self.on_save)
        self.cmd.returnPressed.connect(self.on_run)

    def _log(self, msg: str):
        self.log.append(msg)

    def _set_loading_ui(self, loading: bool):
        self.btn_load.setEnabled(not loading)
        self.btn_view.setEnabled(not loading)
        self.btn_save.setEnabled((not loading) and (self.current is not None))
        self.btn_run.setEnabled(not loading)
        self.cmd.setEnabled(not loading)

    def on_ai_mode_changed(self, idx: int):
        self.nl.set_mode("ollama" if idx == 1 else "rules")
        self._log("AI mode: Ollama (local)" if idx == 1 else "AI mode: Rules (offline)")

    def on_ollama_model_changed(self):
        self.nl.set_ollama_model(self.ollama_model.text().strip() or "llama3.2:3b")

    def on_load(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Apri nuvola punti",
            "",
            "Point Clouds (*.ply *.pcd *.xyz *.xyzn *.xyzrgb *.las *.laz *.e57);;All files (*.*)"
        )
        if not path:
            return

        self._log(f"Load start: {path}")
        self.status.setText("Caricamento in corso… (file grande: può richiedere minuti)")
        self._set_loading_ui(True)

        self._load_thread = QThread()
        self._load_worker = LoaderWorker(path)
        self._load_worker.moveToThread(self._load_thread)

        self._load_thread.started.connect(self._load_worker.run)
        self._load_worker.finished.connect(self._on_loaded)
        self._load_worker.error.connect(self._on_load_error)

        self._load_worker.finished.connect(self._load_thread.quit)
        self._load_worker.error.connect(self._load_thread.quit)
        self._load_thread.finished.connect(self._load_thread.deleteLater)
        self._load_thread.start()

    def _on_loaded(self, pcd, path: str):
        self.original = pcd
        self.current = pcd
        info = get_bounds_info(self.current)
        self.status.setText(
            f"Caricato: {path} | Punti: {info['count']} | Z min/max: {info['min'][2]:.3f} / {info['max'][2]:.3f}"
        )
        self._log("Load OK.")
        self.btn_save.setEnabled(True)
        self._set_loading_ui(False)

    def _on_load_error(self, msg: str):
        self._log("ERRORE load:\n" + msg)
        self.status.setText("Errore durante il caricamento.")
        self._set_loading_ui(False)

    def on_view(self):
        if self.current is None:
            self._log("Nessuna nuvola caricata.")
            return
        o3d.visualization.draw_geometries([self.current], window_name="PointAI Viewer")

    def on_save(self):
        if self.current is None:
            return
        out_path, _ = QFileDialog.getSaveFileName(self, "Salva come PLY", "", "PLY (*.ply)")
        if not out_path:
            return
        ok = o3d.io.write_point_cloud(out_path, self.current)
        self._log(f"Salvato: {out_path} ({'OK' if ok else 'FAIL'})")

    def on_run(self):
        if self.original is None:
            self._log("Carica prima una nuvola punti.")
            return

        text = self.cmd.text().strip()
        self._log(f"> {text}")

        c = parse_command(text)
        if c.name in ("unknown", "noop"):
            translated = self.nl.translate_to_command(text)
            if translated:
                self._log(f"AI → {translated}")
                c = parse_command(translated)
            else:
                if c.name == "unknown":
                    self._log("Comando non riconosciuto. Scrivi: help")
                return

        try:
            if c.name == "help":
                self._log("Comandi: info | lowest <p> | downsample <v> | denoise <n> <std> | cluster <eps> <min> | ground <dist> <n> <iters> | reset")
            elif c.name == "info":
                info = get_bounds_info(self.current)
                self._log(f"Punti: {info['count']}\nMin: {info['min']}\nMax: {info['max']}")
            elif c.name == "reset":
                self.current = self.original
                self._log("Reset a nuvola originale.")
            elif c.name == "lowest":
                self.current = lowest_points(self.current, c.args["percentile"])
                self._log("OK lowest")
            elif c.name == "downsample":
                self.current = voxel_downsample(self.current, c.args["voxel"])
                self._log("OK downsample")
            elif c.name == "denoise":
                self.current = denoise_statistical(self.current, c.args["nb_neighbors"], c.args["std_ratio"])
                self._log("OK denoise")
            elif c.name == "cluster":
                self.current, n = dbscan_clusters(self.current, c.args["eps"], c.args["min_points"])
                self._log(f"OK cluster {n}")
            elif c.name == "ground":
                ground, non_ground, plane = extract_ground_ransac(
                    self.current,
                    c.args["distance_threshold"],
                    c.args["ransac_n"],
                    c.args["num_iterations"]
                )
                self.current = non_ground
                self._log(f"OK ground plane={plane} (current=NON-terreno)")
        except Exception as e:
            self._log(f"ERRORE comando: {e}\n{traceback.format_exc()}")
