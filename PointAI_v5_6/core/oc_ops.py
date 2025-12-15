from __future__ import annotations
import numpy as np

def apply_ops(points: np.ndarray, ops: list[dict]) -> np.ndarray:
    """Return boolean keep mask for points."""
    if not ops:
        return np.ones((points.shape[0],), dtype=bool)
    keep = np.ones((points.shape[0],), dtype=bool)
    for op in ops:
        t = op.get("type")
        if t == "zrange":
            zmin = float(op["zmin"]); zmax = float(op["zmax"])
            keep &= (points[:,2] >= zmin) & (points[:,2] <= zmax)
        elif t == "bbox":
            xmin,xmax,ymin,ymax,zmin,zmax = map(float, (op["xmin"],op["xmax"],op["ymin"],op["ymax"],op["zmin"],op["zmax"]))
            keep &= (
                (points[:,0] >= xmin) & (points[:,0] <= xmax) &
                (points[:,1] >= ymin) & (points[:,1] <= ymax) &
                (points[:,2] >= zmin) & (points[:,2] <= zmax)
            )
        elif t == "remove_bbox":
            xmin,xmax,ymin,ymax,zmin,zmax = map(float, (op["xmin"],op["xmax"],op["ymin"],op["ymax"],op["zmin"],op["zmax"]))
            rem = (
                (points[:,0] >= xmin) & (points[:,0] <= xmax) &
                (points[:,1] >= ymin) & (points[:,1] <= ymax) &
                (points[:,2] >= zmin) & (points[:,2] <= zmax)
            )
            keep &= ~rem
    return keep
