from __future__ import annotations
import os, math
import numpy as np
from collections import defaultdict

from core.oc_store import PointStore, StoreMeta
from core.stream_loaders import load_las_laz_reservoir, load_e57_sample

def _tile_indices(points: np.ndarray, tile_size: float, bmin: np.ndarray):
    rel = (points - bmin) / tile_size
    return np.floor(rel).astype(np.int32)

def _voxel_down(points: np.ndarray, colors: np.ndarray | None, voxel: float):
    q = np.floor(points / voxel).astype(np.int64)
    key = q[:,0]*73856093 ^ q[:,1]*19349663 ^ q[:,2]*83492791
    _, first = np.unique(key, return_index=True)
    pts = points[first]
    cols = colors[first] if colors is not None else None
    return pts, cols

def build_store_from_source(
    source_path: str,
    store_dir: str,
    tile_size: float = 50.0,
    lod_voxels: list[float] = [0.10, 0.25, 0.50, 1.0],
    max_points_ingest: int = 10_000_000,
    progress_cb=None
):
    os.makedirs(store_dir, exist_ok=True)
    ps = PointStore(store_dir)

    ext = os.path.splitext(source_path)[1].lower()

    def cb(p, m):
        if progress_cb:
            progress_cb(float(p), str(m))

    cb(1.0, "Ingest: caricamento campione ...")
    if ext in (".las", ".laz"):
        pts, cols = load_las_laz_reservoir(
            source_path,
            target_points=max_points_ingest,
            progress_cb=lambda p,m: cb(min(20.0, p*0.2), m)
        )
    elif ext == ".e57":
        pts, cols = load_e57_sample(
            source_path,
            target_points=max_points_ingest,
            progress_cb=lambda p,m: cb(min(20.0, p*0.2), m)
        )
    else:
        import open3d as o3d
        pcd = o3d.io.read_point_cloud(source_path)
        pts = np.asarray(pcd.points).astype(np.float64)
        cols = np.asarray(pcd.colors).astype(np.float64) if pcd.has_colors() else None
        if pts.shape[0] > max_points_ingest:
            step = int(math.ceil(pts.shape[0]/max_points_ingest))
            idx = np.arange(0, pts.shape[0], step)[:max_points_ingest]
            pts = pts[idx]
            if cols is not None:
                cols = cols[idx]
        cb(20.0, f"Ingest (open3d) completato: {pts.shape[0]:,} punti")

    bmin = pts.min(axis=0)
    bmax = pts.max(axis=0)

    meta = StoreMeta(
        version=5,
        crs=None,
        bounds_min=bmin.tolist(),
        bounds_max=bmax.tolist(),
        tile_size=float(tile_size),
        lod_voxel_sizes=[float(v) for v in lod_voxels],
        has_rgb=(cols is not None),
    )
    ps.write_meta(meta)
    ps.ensure_ops()

    for li, voxel in enumerate(lod_voxels):
        cb(20.0 + li*(70.0/len(lod_voxels)), f"LOD{li}: voxel {voxel} ...")
        lod_pts, lod_cols = _voxel_down(pts, cols, voxel)

        idx = _tile_indices(lod_pts, tile_size, bmin)
        buckets = defaultdict(list)
        for i in range(lod_pts.shape[0]):
            key = (int(idx[i,0]), int(idx[i,1]), int(idx[i,2]))
            buckets[key].append(i)

        total_tiles = len(buckets) if buckets else 1
        for ti, (k, inds) in enumerate(buckets.items()):
            tile_pts = lod_pts[inds].astype(np.float32)
            tile_cols = lod_cols[inds].astype(np.float32) if lod_cols is not None else None
            ps.write_tile(li, k[0], k[1], k[2], tile_pts, tile_cols)
            base = 20.0 + li*(70.0/len(lod_voxels))
            span = (70.0/len(lod_voxels))
            cb(base + (ti/total_tiles)*span, f"LOD{li}: tile {ti+1}/{total_tiles}")

        cb(20.0 + (li+1)*(70.0/len(lod_voxels)), f"LOD{li}: scritto {len(buckets)} tiles ({lod_pts.shape[0]:,} punti)")

    cb(100.0, f"Store creato in: {store_dir}")
    return store_dir
