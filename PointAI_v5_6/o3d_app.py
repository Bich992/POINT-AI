from __future__ import annotations
import os
import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering

from core.io_loaders import load_pointcloud


class PointAIViewerApp:
    """
    PointAI v3: Viewer integrato + comandi + preview/apply/cancel.
    - Orbit / Pan / Zoom integrati nello SceneWidget.
    - Preview colora:
        keep = verde, remove = rosso, resto = grigio
    """
    def __init__(self):
        self.pcd: o3d.geometry.PointCloud | None = None
        self._original: o3d.geometry.PointCloud | None = None
        self.preview_pcd: o3d.geometry.PointCloud | None = None
        self.preview_result = None  # ("filter_mask", keep_bool) oppure ("replace_pcd", pcd)

        gui.Application.instance.initialize()
        self.window = gui.Application.instance.create_window("PointAI v3 - Viewer", 1280, 820)

        # Layout
        self.root = gui.Horiz(0, gui.Margins(8, 8, 8, 8))
        self.panel = gui.Vert(8, gui.Margins(8, 8, 8, 8))
        self.viewer = gui.SceneWidget()
        self.viewer.scene = rendering.Open3DScene(self.window.renderer)

        self.root.add_child(self.panel)
        self.root.add_child(self.viewer)
        self.window.add_child(self.root)

        # Scene setup
        self.viewer.scene.set_background([1, 1, 1, 1])
        self.viewer.scene.scene.set_sun_light([0.577, -0.577, -0.577], [1, 1, 1], 75000)
        self.viewer.scene.scene.enable_sun_light(True)

        self.mat = rendering.MaterialRecord()
        self.mat.shader = "defaultUnlit"
        self.mat.point_size = 2.0

        # Panel UI
        self.btn_load = gui.Button("Carica nuvola (PLY/PCD/XYZ/LAS/LAZ/E57)")
        self.btn_load.set_on_clicked(self.on_load)

        self.btn_reset_view = gui.Button("Reset vista")
        self.btn_reset_view.set_on_clicked(self.reset_camera)

        self.btn_reset_cloud = gui.Button("Reset nuvola (undo totale)")
        self.btn_reset_cloud.set_on_clicked(self.on_reset_cloud)

        self.lbl = gui.Label("Pronto.")

        self.panel.add_child(self.btn_load)
        self.panel.add_child(self.btn_reset_view)
        self.panel.add_child(self.btn_reset_cloud)
        self.panel.add_fixed(8)
        self.panel.add_child(self.lbl)

        self.panel.add_fixed(10)
        self.panel.add_child(gui.Label("Comando (preview/apply):"))
        self.cmd = gui.TextEdit()
        self.cmd.placeholder_text = (
            "Esempi:\n"
            "  denoise 20 2.0\n"
            "  downsample 0.05\n"
            "  zrange 0 10\n"
            "  slicez 5 0.2\n"
            "  bbox xmin xmax ymin ymax zmin zmax\n"
            "  ground 0.05 3 2000\n"
            "  cluster 0.12 30\n"
            "  color z  |  color gray\n"
            "  info"
        )
        self.panel.add_child(self.cmd)

        self.btn_preview = gui.Button("Preview")
        self.btn_preview.set_on_clicked(self.on_preview)

        self.btn_apply = gui.Button("Apply")
        self.btn_apply.set_on_clicked(self.on_apply)

        self.btn_cancel = gui.Button("Cancel preview")
        self.btn_cancel.set_on_clicked(self.on_cancel_preview)

        self.panel.add_child(self.btn_preview)
        self.panel.add_child(self.btn_apply)
        self.panel.add_child(self.btn_cancel)

        self.panel.add_fixed(10)
        self.panel.add_child(gui.Label("Point size"))
        self.point_size = gui.Slider(gui.Slider.DOUBLE)
        self.point_size.set_limits(1.0, 8.0)
        self.point_size.double_value = 2.0
        self.point_size.set_on_value_changed(self.on_point_size)
        self.panel.add_child(self.point_size)

        self.panel.add_fixed(10)
        self.panel.add_child(gui.Label("Mouse: orbit/pan/zoom"))

    # ---------- helpers ----------
    def on_point_size(self, v):
        self.mat.point_size = float(v)
        self._refresh_view()

    def _refresh_view(self):
        geom = self.preview_pcd if self.preview_pcd is not None else self.pcd
        if geom is None:
            return
        name = "preview" if self.preview_pcd is not None else "pcd"
        self.viewer.scene.clear_geometry()
        self.viewer.scene.add_geometry(name, geom, self.mat)

    def _clone_pcd(self, pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
        out = o3d.geometry.PointCloud()
        out.points = o3d.utility.Vector3dVector(np.asarray(pcd.points))
        if pcd.has_colors():
            out.colors = o3d.utility.Vector3dVector(np.asarray(pcd.colors))
        return out

    def _set_all_gray(self, pcd: o3d.geometry.PointCloud, g: float = 0.35):
        n = len(pcd.points)
        pcd.colors = o3d.utility.Vector3dVector(np.full((n, 3), g, dtype=np.float64))

    def _ensure_colors(self, pcd: o3d.geometry.PointCloud):
        if not pcd.has_colors():
            self._set_all_gray(pcd, 0.35)

    def _overlay_mask_colors(self, base_pcd: o3d.geometry.PointCloud, mask_keep=None, mask_remove=None):
        p = self._clone_pcd(base_pcd)
        self._set_all_gray(p, 0.35)
        cols = np.asarray(p.colors)
        if mask_keep is not None:
            cols[mask_keep] = np.array([0.2, 0.8, 0.2])
        if mask_remove is not None:
            cols[mask_remove] = np.array([0.9, 0.2, 0.2])
        p.colors = o3d.utility.Vector3dVector(cols)
        return p

    def _parse_cmd(self, text: str):
        t = (text or "").strip().lower().split()
        if not t:
            return None, []
        return t[0], t[1:]

    def _z_colorize(self, pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
        pts = np.asarray(pcd.points)
        z = pts[:, 2]
        zmin, zmax = float(z.min()), float(z.max())
        denom = (zmax - zmin) if (zmax > zmin) else 1.0
        t = (z - zmin) / denom
        colors = np.stack([t, 0.2 * np.ones_like(t), 1.0 - t], axis=1)
        out = self._clone_pcd(pcd)
        out.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))
        return out

    # ---------- actions ----------
    def on_reset_cloud(self):
        if self._original is None:
            return
        self.pcd = self._clone_pcd(self._original)
        self.preview_pcd = None
        self.preview_result = None
        self._ensure_colors(self.pcd)
        self._refresh_view()
        self.reset_camera()
        self.lbl.text = "Reset nuvola: tornato all'originale."

    def on_load(self):
        """
        Usa il selezionatore file "classico" (Windows) via tkinter,
        così puoi navigare cartelle come nel QFileDialog.
        """
        try:
            import tkinter as tk
            from tkinter import filedialog
        except Exception as e:
            self.lbl.text = f"Tkinter non disponibile: {e}"
            return

        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)

        path = filedialog.askopenfilename(
            title="Apri nuvola punti",
            filetypes=[
                ("Point Clouds", "*.ply *.pcd *.xyz *.xyzn *.xyzrgb *.las *.laz *.e57"),
                ("All files", "*.*"),
            ],
        )
        root.destroy()

        if not path:
            return

        try:
            self.lbl.text = "Caricamento... (file grande: può richiedere minuti)"
            gui.Application.instance.post_to_main_thread(self.window, lambda: None)

            pcd = load_pointcloud(path)
            self.pcd = pcd
            self._original = self._clone_pcd(pcd)
            self.preview_pcd = None
            self.preview_result = None

            self._ensure_colors(self.pcd)
            self._refresh_view()
            self.reset_camera()

            self.lbl.text = f"Caricato: {os.path.basename(path)} | punti: {len(pcd.points)}"
        except Exception as e:
            self.lbl.text = f"Errore load: {e}"

    def reset_camera(self):
        if self.pcd is None or len(self.pcd.points) == 0:
            return
        bounds = self.pcd.get_axis_aligned_bounding_box()
        self.viewer.setup_camera(60.0, bounds, bounds.get_center())

    # ---------- preview/apply ----------
    def on_preview(self):
        if self.pcd is None:
            self.lbl.text = "Carica una nuvola prima."
            return

        cmd_text = (self.cmd.text or "").strip()
        name, args = self._parse_cmd(cmd_text)
        if not name:
            self.lbl.text = "Inserisci un comando."
            return

        pts = np.asarray(self.pcd.points)
        n = pts.shape[0]

        try:
            self.preview_result = None
            self.preview_pcd = None

            if name == "info":
                mn = pts.min(axis=0); mx = pts.max(axis=0)
                self.lbl.text = f"Punti: {n} | Z min/max: {mn[2]:.3f}/{mx[2]:.3f}"
                return

            if name == "zrange":
                zmin, zmax = float(args[0]), float(args[1])
                keep = (pts[:, 2] >= zmin) & (pts[:, 2] <= zmax)
                self.preview_pcd = self._overlay_mask_colors(self.pcd, mask_keep=keep, mask_remove=~keep)
                self.preview_result = ("filter_mask", keep)

            elif name == "slicez":
                z0, th = float(args[0]), float(args[1])
                keep = (np.abs(pts[:, 2] - z0) <= (th / 2.0))
                self.preview_pcd = self._overlay_mask_colors(self.pcd, mask_keep=keep, mask_remove=~keep)
                self.preview_result = ("filter_mask", keep)

            elif name == "bbox":
                xmin, xmax, ymin, ymax, zmin, zmax = map(float, args[:6])
                keep = (
                    (pts[:, 0] >= xmin) & (pts[:, 0] <= xmax) &
                    (pts[:, 1] >= ymin) & (pts[:, 1] <= ymax) &
                    (pts[:, 2] >= zmin) & (pts[:, 2] <= zmax)
                )
                self.preview_pcd = self._overlay_mask_colors(self.pcd, mask_keep=keep, mask_remove=~keep)
                self.preview_result = ("filter_mask", keep)

            elif name == "downsample":
                v = float(args[0])
                ds = self.pcd.voxel_down_sample(v)
                self._ensure_colors(ds)
                self.preview_pcd = ds
                self.preview_result = ("replace_pcd", ds)

            elif name == "denoise":
                nn = int(args[0]) if len(args) >= 1 else 20
                std = float(args[1]) if len(args) >= 2 else 2.0
                _, ind = self.pcd.remove_statistical_outlier(nb_neighbors=nn, std_ratio=std)
                keep = np.zeros(n, dtype=bool)
                keep[np.array(ind, dtype=int)] = True
                self.preview_pcd = self._overlay_mask_colors(self.pcd, mask_keep=keep, mask_remove=~keep)
                self.preview_result = ("filter_mask", keep)

            elif name == "ground":
                dist = float(args[0]) if len(args) >= 1 else 0.05
                rn = int(args[1]) if len(args) >= 2 else 3
                iters = int(args[2]) if len(args) >= 3 else 2000
                plane, inliers = self.pcd.segment_plane(distance_threshold=dist, ransac_n=rn, num_iterations=iters)
                is_ground = np.zeros(n, dtype=bool)
                is_ground[np.array(inliers, dtype=int)] = True
                self.preview_pcd = self._overlay_mask_colors(self.pcd, mask_keep=is_ground, mask_remove=~is_ground)
                self.preview_result = ("filter_mask", ~is_ground)  # apply keeps non-ground
                self.lbl.text = f"Preview ground | plane={plane}"

            elif name == "cluster":
                eps = float(args[0]) if len(args) >= 1 else 0.10
                minp = int(args[1]) if len(args) >= 2 else 20
                labels = np.array(self.pcd.cluster_dbscan(eps=eps, min_points=minp, print_progress=False))
                out = self._clone_pcd(self.pcd)
                colors = np.full((len(labels), 3), 0.35, dtype=np.float64)
                max_label = labels.max() if labels.size else -1
                for k in range(int(max_label) + 1):
                    mask = labels == k
                    rgb = np.array([((k * 53) % 255), ((k * 97) % 255), ((k * 193) % 255)]) / 255.0
                    colors[mask] = rgb
                out.colors = o3d.utility.Vector3dVector(colors)
                self.preview_pcd = out
                self.preview_result = ("replace_pcd", out)

            elif name == "color":
                mode = args[0] if args else "gray"
                if mode == "gray":
                    out = self._clone_pcd(self.pcd); self._set_all_gray(out, 0.35)
                    self.preview_pcd = out; self.preview_result = ("replace_pcd", out)
                elif mode == "z":
                    out = self._z_colorize(self.pcd)
                    self.preview_pcd = out; self.preview_result = ("replace_pcd", out)
                else:
                    self.lbl.text = "color supporta: gray | z"
                    return

            else:
                self.lbl.text = f"Comando non supportato: {name}"
                return

            if self.preview_pcd is not None:
                self._refresh_view()
                if not self.lbl.text.startswith("Preview ground"):
                    self.lbl.text = f"Preview: {cmd_text}"

        except Exception as e:
            self.lbl.text = f"Errore preview: {e}"

    def on_apply(self):
        if self.pcd is None:
            return
        if self.preview_result is None:
            self.lbl.text = "Nessun preview attivo."
            return

        kind, payload = self.preview_result

        if kind == "filter_mask":
            keep = payload
            idx = np.where(keep)[0].tolist()
            self.pcd = self.pcd.select_by_index(idx)
            self._ensure_colors(self.pcd)
        elif kind == "replace_pcd":
            self.pcd = payload
            self._ensure_colors(self.pcd)

        self.preview_result = None
        self.preview_pcd = None
        self._refresh_view()
        self.reset_camera()
        self.lbl.text = "Applicato."

    def on_cancel_preview(self):
        if self.pcd is None:
            return
        self.preview_result = None
        self.preview_pcd = None
        self._refresh_view()
        self.lbl.text = "Preview cancellato."


def main():
    PointAIViewerApp()
    gui.Application.instance.run()


if __name__ == "__main__":
    main()
