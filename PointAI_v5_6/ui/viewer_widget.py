from __future__ import annotations
import numpy as np

import pyqtgraph.opengl as gl
import pyqtgraph as pg
from PySide6.QtWidgets import QWidget


class PointCloudViewer(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.view = gl.GLViewWidget()
        self.view.setCameraPosition(distance=50)

        layout = pg.QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.view)

        self._scatter = None
        self._points = None  # centered points (float32)

        # helpers
        axis = gl.GLAxisItem()
        axis.setSize(50, 50, 50)
        self.view.addItem(axis)

        grid = gl.GLGridItem()
        grid.setSize(200, 200)
        grid.setSpacing(10, 10)
        self.view.addItem(grid)

    def set_pointcloud(self, points: np.ndarray, colors: np.ndarray | None = None, point_size: float = 2.0):
        self.view.clear()

        pts = np.asarray(points, dtype=np.float64)
        if pts.ndim != 2 or pts.shape[1] != 3:
            raise ValueError("points deve essere un array Nx3")

        # 1) rimuovi NaN/Inf (altrimenti OpenGL spesso non disegna nulla)
        mask = np.isfinite(pts).all(axis=1)
        pts = pts[mask]
        if pts.shape[0] == 0:
            raise ValueError("Tutti i punti erano NaN/Inf. Nulla da visualizzare.")

        # 2) centra punti (niente translate accumulato)
        center = pts.mean(axis=0)
        pts_c = (pts - center).astype(np.float32, copy=False)
        self._points = pts_c

        # colori
        if colors is None:
            rgba = np.ones((len(pts_c), 4), dtype=np.float32)
            rgba[:, :3] = 0.35
        else:
            c = np.asarray(colors, dtype=np.float32)
            c = c[mask]
            if c.ndim == 2 and c.shape[1] >= 3:
                c = c[:, :3]
            if c.max() > 1.0:
                c = c / 255.0
            rgba = np.concatenate([c, np.ones((len(c), 1), dtype=np.float32)], axis=1)

        self._scatter = gl.GLScatterPlotItem(pos=pts_c, color=rgba, size=float(point_size), pxMode=True)
        self.view.addItem(self._scatter)

        # axis+grid di nuovo (clear li rimuove)
        axis = gl.GLAxisItem()
        axis.setSize(50, 50, 50)
        self.view.addItem(axis)

        grid = gl.GLGridItem()
        grid.setSize(200, 200)
        grid.setSpacing(10, 10)
        self.view.addItem(grid)

        self.autofit()

    def autofit(self):
        if self._points is None or len(self._points) == 0:
            return
        mn = self._points.min(axis=0)
        mx = self._points.max(axis=0)
        extent = float(np.linalg.norm(mx - mn))
        dist = max(10.0, extent * 1.2)

        # imposta camera “sensata” e non all’origine random
        self.view.setCameraPosition(distance=dist, elevation=20, azimuth=45)

    def set_point_size(self, px: float):
        if self._scatter is not None:
            self._scatter.setData(size=float(px))
