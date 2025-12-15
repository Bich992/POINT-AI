from __future__ import annotations
import numpy as np
import open3d as o3d
from core.oc_store import PointStore
from core.oc_ops import apply_ops

def export_filtered_ply(store_dir: str, out_path: str, lod: int = 0, max_points: int | None = None, progress_cb=None):
    ps = PointStore(store_dir)
    ps.ensure_ops()
    ops = ps.read_ops()

    tiles = ps.list_tiles(lod)
    pts_all = []
    cols_all = []

    for i, key in enumerate(tiles):
        g = ps.z[f"lod{lod}/tiles/{key}"]
        pts = np.asarray(g["points"])
        cols = np.asarray(g["colors"]) if "colors" in g else None

        keep = apply_ops(pts.astype(np.float64), ops)
        pts = pts[keep]
        if cols is not None:
            cols = cols[keep]

        if pts.size:
            pts_all.append(pts)
            if cols is not None:
                cols_all.append(cols)

        if progress_cb and len(tiles):
            progress_cb((i+1)/len(tiles)*100.0, f"Export: tile {i+1}/{len(tiles)}")

    if not pts_all:
        raise ValueError("Nessun punto da esportare (dopo filtri).")

    P = np.concatenate(pts_all, axis=0)
    C = np.concatenate(cols_all, axis=0) if cols_all else None

    if max_points is not None and P.shape[0] > int(max_points):
        step = int(np.ceil(P.shape[0]/int(max_points)))
        idx = np.arange(0, P.shape[0], step)[:int(max_points)]
        P = P[idx]
        if C is not None:
            C = C[idx]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(P.astype(np.float64))
    if C is not None:
        pcd.colors = o3d.utility.Vector3dVector(C.astype(np.float64))

    if not o3d.io.write_point_cloud(out_path, pcd):
        raise RuntimeError("write_point_cloud failed")

    if progress_cb:
        progress_cb(100.0, f"Export completato: {out_path}")
    return out_path
