from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import json
import numpy as np
import zarr

# Dual-path compression:
# - Zarr v2: uses numcodecs compressors (e.g., numcodecs.Blosc) via `compressor=`
# - Zarr v3: uses zarr.codecs via `compressors=` and BytesBytesCodec requirements
try:
    from numcodecs import Blosc as _NCBlosc
except Exception:  # pragma: no cover
    _NCBlosc = None

try:
    from zarr.codecs import BloscCodec as _ZBloscCodec  # zarr>=3
except Exception:  # pragma: no cover
    _ZBloscCodec = None


def _zarr_major() -> int:
    try:
        v = getattr(zarr, "__version__", "2.0.0")
        return int(v.split(".")[0])
    except Exception:
        return 2


def _make_compressor():
    maj = _zarr_major()
    if maj >= 3 and _ZBloscCodec is not None:
        # bytes-bytes codec
        return _ZBloscCodec(cname="zstd", clevel=3, shuffle="bitshuffle")
    if _NCBlosc is None:
        return None
    return _NCBlosc(cname="zstd", clevel=3, shuffle=_NCBlosc.BITSHUFFLE)


COMP = _make_compressor()


@dataclass
class StoreMeta:
    version: int
    crs: str | None
    bounds_min: list[float]
    bounds_max: list[float]
    tile_size: float
    lod_voxel_sizes: list[float]
    has_rgb: bool


class PointStore:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.z = zarr.open_group(str(self.root), mode="a")
        self.meta: StoreMeta | None = None

    def write_meta(self, meta: StoreMeta):
        self.meta = meta
        (self.root / "meta.json").write_text(json.dumps(meta.__dict__, indent=2), encoding="utf-8")

    def read_meta(self) -> StoreMeta:
        data = json.loads((self.root / "meta.json").read_text(encoding="utf-8"))
        self.meta = StoreMeta(**data)
        return self.meta

    def ensure_ops(self):
        p = self.root / "ops.json"
        if not p.exists():
            p.write_text(json.dumps({"ops": []}, indent=2), encoding="utf-8")

    def append_op(self, op: dict):
        self.ensure_ops()
        p = self.root / "ops.json"
        data = json.loads(p.read_text(encoding="utf-8"))
        data["ops"].append(op)
        p.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def read_ops(self) -> list[dict]:
        self.ensure_ops()
        return json.loads((self.root / "ops.json").read_text(encoding="utf-8")).get("ops", [])

    def _tile_key(self, ix: int, iy: int, iz: int) -> str:
        return f"{ix}_{iy}_{iz}"

    def tile_group(self, lod: int, ix: int, iy: int, iz: int, create: bool = True):
        g = self.z.require_group(f"lod{lod}").require_group("tiles")
        key = self._tile_key(ix, iy, iz)
        return g.require_group(key) if create else g[key]

    def tile_exists(self, lod: int, ix: int, iy: int, iz: int) -> bool:
        try:
            tiles = self.z[f"lod{lod}/tiles"]
            return self._tile_key(ix, iy, iz) in tiles
        except Exception:
            return False

    def _create_array(self, tg, name: str, data: np.ndarray):
        maj = _zarr_major()
        n = int(len(data)) or 1
        chunks = (min(200_000, n), data.shape[1])

        if name in tg:
            del tg[name]

        if maj >= 3:
            # Zarr v3 requires shape/dtype and uses `compressors=`
            kwargs = dict(shape=data.shape, dtype=data.dtype, chunks=chunks, overwrite=True)
            if COMP is not None:
                kwargs["compressors"] = [COMP]
            arr = tg.create_dataset(name, **kwargs)
            arr[:] = data
            return arr

        # Zarr v2 path: accepts data= and `compressor=`
        kwargs = dict(data=data, chunks=chunks, overwrite=True)
        if COMP is not None:
            kwargs["compressor"] = COMP
        return tg.create_dataset(name, **kwargs)

    def write_tile(self, lod: int, ix: int, iy: int, iz: int, points: np.ndarray, colors: np.ndarray | None):
        tg = self.tile_group(lod, ix, iy, iz, create=True)
        pts = points.astype(np.float32, copy=False)
        self._create_array(tg, "points", pts)
        if colors is not None:
            cols = colors.astype(np.float32, copy=False)
            self._create_array(tg, "colors", cols)
        else:
            if "colors" in tg:
                del tg["colors"]

    def read_tile(self, lod: int, ix: int, iy: int, iz: int):
        tg = self.tile_group(lod, ix, iy, iz, create=False)
        pts = np.asarray(tg["points"])
        cols = np.asarray(tg["colors"]) if "colors" in tg else None
        return pts, cols

    def list_tiles(self, lod: int):
        try:
            tiles = self.z[f"lod{lod}/tiles"]
            return list(tiles.group_keys())
        except Exception:
            return []
