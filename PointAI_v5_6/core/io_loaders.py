from __future__ import annotations
import os
import numpy as np
import open3d as o3d

def _pcd_from_numpy(points: np.ndarray, colors: np.ndarray | None = None) -> o3d.geometry.PointCloud:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    if colors is not None:
        c = colors.astype(np.float64)
        if c.max() > 1.0:
            c = c / 255.0
        c = np.clip(c, 0.0, 1.0)
        pcd.colors = o3d.utility.Vector3dVector(c)
    return pcd

def load_pointcloud(path: str) -> o3d.geometry.PointCloud:
    ext = os.path.splitext(path)[1].lower()

    if ext in (".ply", ".pcd", ".xyz", ".xyzn", ".xyzrgb"):
        pcd = o3d.io.read_point_cloud(path)
        if pcd.is_empty():
            raise ValueError("Nuvola punti vuota o formato non supportato.")
        return pcd

    if ext in (".las", ".laz"):
        return load_las_laz(path)

    if ext in (".e57",):
        return load_e57(path)

    raise ValueError(f"Estensione non supportata: {ext}")

def load_las_laz(path: str) -> o3d.geometry.PointCloud:
    import laspy
    las = laspy.read(path)
    points = np.vstack((las.x, las.y, las.z)).T

    colors = None
    has_rgb = all(hasattr(las, k) for k in ("red", "green", "blue"))
    if has_rgb:
        r = np.asarray(las.red)
        g = np.asarray(las.green)
        b = np.asarray(las.blue)
        colors = np.vstack((r, g, b)).T
        if colors.max() > 255:
            colors = (colors / 65535.0) * 255.0

    return _pcd_from_numpy(points, colors)

def load_e57(path: str) -> o3d.geometry.PointCloud:
    import pye57
    e57 = pye57.E57(path)
    try:
        data = e57.read_scan(0, colors=True, intensity=False, row_column=False, ignore_missing_fields=True)
    except TypeError:
        data = e57.read_scan(0, colors=True, intensity=False, row_column=False)

    x = np.asarray(data["cartesianX"])
    y = np.asarray(data["cartesianY"])
    z = np.asarray(data["cartesianZ"])
    points = np.vstack((x, y, z)).T

    colors = None
    if all(k in data for k in ("colorRed", "colorGreen", "colorBlue")):
        colors = np.vstack((data["colorRed"], data["colorGreen"], data["colorBlue"])).T
        if colors.max() > 255:
            colors = (colors / 65535.0) * 255.0

    return _pcd_from_numpy(points, colors)
