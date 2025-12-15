from __future__ import annotations
import numpy as np
import open3d as o3d

def get_bounds_info(pcd: o3d.geometry.PointCloud) -> dict:
    pts = np.asarray(pcd.points)
    mn = pts.min(axis=0)
    mx = pts.max(axis=0)
    return {"count": int(len(pts)), "min": mn.tolist(), "max": mx.tolist()}

def lowest_points(pcd: o3d.geometry.PointCloud, percentile: float = 1.0) -> o3d.geometry.PointCloud:
    pts = np.asarray(pcd.points)
    z = pts[:, 2]
    percentile = float(np.clip(percentile, 0.0, 100.0))
    thr = np.percentile(z, percentile)
    idx = np.where(z <= thr)[0]
    return pcd.select_by_index(idx)

def voxel_downsample(pcd: o3d.geometry.PointCloud, voxel_size: float = 0.05) -> o3d.geometry.PointCloud:
    voxel_size = max(1e-6, float(voxel_size))
    return pcd.voxel_down_sample(voxel_size)

def denoise_statistical(pcd: o3d.geometry.PointCloud, nb_neighbors: int = 20, std_ratio: float = 2.0) -> o3d.geometry.PointCloud:
    nb_neighbors = max(1, int(nb_neighbors))
    std_ratio = max(0.1, float(std_ratio))
    _, ind = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    return pcd.select_by_index(ind)

def dbscan_clusters(pcd: o3d.geometry.PointCloud, eps: float = 0.10, min_points: int = 20) -> tuple[o3d.geometry.PointCloud, int]:
    labels = np.array(pcd.cluster_dbscan(eps=float(eps), min_points=int(min_points), print_progress=False))
    if labels.size == 0:
        return pcd, 0

    max_label = int(labels.max())
    colors = np.zeros((len(labels), 3), dtype=np.float64)

    if max_label >= 0:
        for k in range(max_label + 1):
            mask = labels == k
            rgb = np.array([((k * 53) % 255), ((k * 97) % 255), ((k * 193) % 255)]) / 255.0
            colors[mask] = rgb

    pcd_out = pcd.clone()
    pcd_out.colors = o3d.utility.Vector3dVector(colors)
    return pcd_out, (max_label + 1) if max_label >= 0 else 0

def extract_ground_ransac(
    pcd: o3d.geometry.PointCloud,
    distance_threshold: float = 0.05,
    ransac_n: int = 3,
    num_iterations: int = 2000
) -> tuple[o3d.geometry.PointCloud, o3d.geometry.PointCloud, list[float]]:
    distance_threshold = max(1e-6, float(distance_threshold))
    ransac_n = max(3, int(ransac_n))
    num_iterations = max(100, int(num_iterations))

    plane_model, inliers = pcd.segment_plane(
        distance_threshold=distance_threshold,
        ransac_n=ransac_n,
        num_iterations=num_iterations
    )
    ground = pcd.select_by_index(inliers)
    non_ground = pcd.select_by_index(inliers, invert=True)
    return ground, non_ground, plane_model
