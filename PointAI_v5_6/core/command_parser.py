from __future__ import annotations
from dataclasses import dataclass

@dataclass
class Command:
    name: str
    args: dict

def parse_command(text: str) -> Command:
    t = (text or "").strip().lower()
    if not t:
        return Command("noop", {})
    parts = t.split()
    cmd = parts[0]
    vals = parts[1:]

    if cmd in ("help", "?"):
        return Command("help", {})
    if cmd == "info":
        return Command("info", {})
    if cmd == "reset":
        return Command("reset", {})

    if cmd in ("lowest", "minz"):
        perc = float(vals[0]) if len(vals) >= 1 else 1.0
        return Command("lowest", {"percentile": perc})

    if cmd in ("downsample", "voxel"):
        v = float(vals[0]) if len(vals) >= 1 else 0.05
        return Command("downsample", {"voxel": v})

    if cmd in ("denoise", "sor"):
        nn = int(vals[0]) if len(vals) >= 1 else 20
        sr = float(vals[1]) if len(vals) >= 2 else 2.0
        return Command("denoise", {"nb_neighbors": nn, "std_ratio": sr})

    if cmd in ("cluster", "dbscan"):
        eps = float(vals[0]) if len(vals) >= 1 else 0.10
        mp = int(vals[1]) if len(vals) >= 2 else 20
        return Command("cluster", {"eps": eps, "min_points": mp})

    if cmd in ("ground", "terrain", "plane"):
        dist = float(vals[0]) if len(vals) >= 1 else 0.05
        rn = int(vals[1]) if len(vals) >= 2 else 3
        it = int(vals[2]) if len(vals) >= 3 else 2000
        return Command("ground", {"distance_threshold": dist, "ransac_n": rn, "num_iterations": it})

    return Command("unknown", {"raw": text})
