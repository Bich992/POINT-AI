from __future__ import annotations
import numpy as np

def reservoir_update(res_points: np.ndarray, res_colors: np.ndarray|None, seen: int,
                     chunk_points: np.ndarray, chunk_colors: np.ndarray|None,
                     rng: np.random.Generator) -> int:
    """Reservoir sampling update."""
    K = res_points.shape[0]
    for i in range(chunk_points.shape[0]):
        seen += 1
        if seen <= K:
            res_points[seen-1] = chunk_points[i]
            if res_colors is not None and chunk_colors is not None:
                res_colors[seen-1] = chunk_colors[i]
        else:
            j = int(rng.integers(1, seen+1))  # 1..seen
            if j <= K:
                res_points[j-1] = chunk_points[i]
                if res_colors is not None and chunk_colors is not None:
                    res_colors[j-1] = chunk_colors[i]
    return seen

def load_las_laz_reservoir(path: str, target_points: int = 2_000_000, seed: int = 7, progress_cb=None):
    """Chunk LAS/LAZ + reservoir sampling (uniforme) fino a target_points."""
    import laspy

    target_points = int(max(10_000, target_points))
    rng = np.random.default_rng(seed)

    with laspy.open(path) as reader:
        total = int(reader.header.point_count)
        dims = set(reader.point_format.dimension_names)
        has_rgb = {"red","green","blue"}.issubset(dims)

        res_pts = np.empty((target_points, 3), dtype=np.float64)
        res_cols = np.empty((target_points, 3), dtype=np.float64) if has_rgb else None

        seen = 0
        read = 0

        for chunk in reader.chunk_iterator(250_000):
            pts = np.vstack((chunk.x, chunk.y, chunk.z)).T.astype(np.float64)
            cols = None
            if has_rgb:
                cols = np.vstack((chunk.red, chunk.green, chunk.blue)).T.astype(np.float64)
                if cols.max() > 255:
                    cols = (cols / 65535.0) * 255.0
                cols = cols / 255.0
            seen = reservoir_update(res_pts, res_cols, seen, pts, cols, rng)
            read += pts.shape[0]
            if progress_cb and total:
                pct = min(99.0, (read / total) * 100.0)
                progress_cb(pct, f"LAS/LAZ: letti {read:,}/{total:,} punti | campione {min(seen, target_points):,}")
        n = min(seen, target_points)
        res_pts = res_pts[:n].copy()
        if res_cols is not None:
            res_cols = res_cols[:n].copy()
        if progress_cb:
            progress_cb(100.0, f"LAS/LAZ: completato (campione {n:,} punti)")
        return res_pts, res_cols

def load_e57_sample(path: str, target_points: int = 2_000_000, progress_cb=None):
    """E57: pye57 non supporta chunk; leggi scan 0 e campiona con stride."""
    import pye57

    target_points = int(max(10_000, target_points))
    if progress_cb:
        progress_cb(5.0, "E57: lettura scan 0 ...")

    e57 = pye57.E57(path)
    try:
        data = e57.read_scan(0, colors=True, intensity=False, row_column=False, ignore_missing_fields=True)
    except TypeError:
        data = e57.read_scan(0, colors=True, intensity=False, row_column=False)

    x = np.asarray(data["cartesianX"])
    y = np.asarray(data["cartesianY"])
    z = np.asarray(data["cartesianZ"])
    n = int(x.shape[0])

    if progress_cb:
        progress_cb(40.0, f"E57: letti {n:,} punti | campionamento...")

    if n <= target_points:
        idx = np.arange(n)
    else:
        step = int(np.ceil(n / target_points))
        idx = np.arange(0, n, step)
        if idx.shape[0] > target_points:
            idx = idx[:target_points]

    pts = np.vstack((x[idx], y[idx], z[idx])).T.astype(np.float64)

    cols = None
    if all(k in data for k in ("colorRed","colorGreen","colorBlue")):
        r = np.asarray(data["colorRed"])[idx].astype(np.float64)
        g = np.asarray(data["colorGreen"])[idx].astype(np.float64)
        b = np.asarray(data["colorBlue"])[idx].astype(np.float64)
        cols = np.vstack((r,g,b)).T
        if cols.max() > 255:
            cols = (cols / 65535.0) * 255.0
        cols = cols / 255.0

    if progress_cb:
        progress_cb(100.0, f"E57: completato (campione {pts.shape[0]:,} punti)")
    return pts, cols
