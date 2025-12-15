from __future__ import annotations
import numpy as np
from core.oc_store import PointStore
from core.oc_ops import apply_ops

def pick_lod(meta, max_points: int) -> int:
    if max_points <= 300_000:
        return len(meta.lod_voxel_sizes) - 1
    if max_points <= 1_000_000:
        return max(0, len(meta.lod_voxel_sizes) - 2)
    return 0

def load_roi(ps: PointStore, lod: int, center: np.ndarray, radius: float, max_points: int = 2_000_000):
    meta = ps.read_meta()
    bmin = np.array(meta.bounds_min, dtype=np.float64)
    tile = meta.tile_size
    ops = ps.read_ops()

    c = center
    r = float(radius)
    mn = c - r
    mx = c + r

    i0 = np.floor((mn - bmin)/tile).astype(int)
    i1 = np.floor((mx - bmin)/tile).astype(int)

    pts_list = []
    col_list = []

    for ix in range(i0[0], i1[0]+1):
        for iy in range(i0[1], i1[1]+1):
            for iz in range(i0[2], i1[2]+1):
                if not ps.tile_exists(lod, ix, iy, iz):
                    continue
                pts, cols = ps.read_tile(lod, ix, iy, iz)

                m = (
                    (pts[:,0] >= mn[0]) & (pts[:,0] <= mx[0]) &
                    (pts[:,1] >= mn[1]) & (pts[:,1] <= mx[1]) &
                    (pts[:,2] >= mn[2]) & (pts[:,2] <= mx[2])
                )
                pts = pts[m]
                if cols is not None:
                    cols = cols[m]
                if pts.size == 0:
                    continue

                keep = apply_ops(pts.astype(np.float64), ops)
                pts = pts[keep]
                if cols is not None:
                    cols = cols[keep]
                if pts.size == 0:
                    continue

                pts_list.append(pts)
                if cols is not None:
                    col_list.append(cols)

    if not pts_list:
        return np.empty((0,3), dtype=np.float32), None

    P = np.concatenate(pts_list, axis=0)
    C = np.concatenate(col_list, axis=0) if col_list else None

    if P.shape[0] > max_points:
        step = int(np.ceil(P.shape[0]/max_points))
        idx = np.arange(0, P.shape[0], step)[:max_points]
        P = P[idx]
        if C is not None:
            C = C[idx]
    return P, C
