"""Convert PASTIS (France, Sentinel-2 time series, 18 crop types) into per-parcel patches.

PASTIS ships 2,433 tiles of 128x128 px (10 m), each a Sentinel-2 L2A time series of
38-61 dates (2018-09 .. 2019-10) with semantic + instance (parcel) labels. This script cuts one small patch
around every labelled parcel and writes it as a GeoTIFF into the folder layout that
``cropcls.train --dataset folder`` reads::

    <out>/fold{1..5}/<crop>/<patch>_<parcel>.tif
    <out>/cropcls_meta.json        # bands, timesteps, dates -> picked up by train.py

Two composite modes:

* ``median``  - one median composite over ``--start..--end`` (10 bands). Works with
  the plain ResNet-18 classifier.
* ``monthly`` - one median composite per month in ``--start..--end`` stacked in time
  (months x 10 bands). Use with ``--timesteps`` / the temporal model; phenology is
  what separates e.g. winter wheat from winter barley.

Parcels are stored per official PASTIS fold (spatially disjoint). ``train.py --cv-fold k``
picks the official rotation k (k=1: train 1,2,3 / val 4 / test 5), and ``cropcls.cv``
runs all 5 and pools the test confusion matrices, which is how the benchmark reports
parcel classification (PASTIS-R paper, Table 1: S2 PSE+LTAE OA 91.7 / mIoU 73.9).
Like the benchmark, background (0) and void (19) are excluded, leaving 18 crop classes.

Download PASTIS (S2 only, ~29 GB) from https://zenodo.org/records/5012942 and unzip,
then::

    python -m cropcls.prepare_pastis --root data/PASTIS --out data/pastis_parcels \
        --composite monthly
"""

from __future__ import annotations

import argparse
import json
import warnings
from collections import Counter
from datetime import date
from pathlib import Path

import numpy as np

PASTIS_S2_BANDS = ("B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12")

PASTIS_CLASSES = (
    "background", "meadow", "soft_winter_wheat", "corn", "winter_barley",
    "winter_rapeseed", "spring_barley", "sunflower", "grapevine", "beet",
    "winter_triticale", "winter_durum_wheat", "fruits_vegetables_flowers", "potatoes",
    "leguminous_fodder", "soybeans", "orchard", "mixed_cereal", "sorghum", "void_label",
)
IGNORE_LABELS = (0, 19)


# ------------------------------------------------------------------------- reading


def find_root(root: Path) -> Path:
    """Accept the unzip dir or its PASTIS / PASTIS-R child."""
    for cand in (root, root / "PASTIS", root / "PASTIS-R"):
        if (cand / "metadata.geojson").exists():
            return cand
    raise FileNotFoundError(f"metadata.geojson not found under {root}")


def read_metadata(root: Path) -> list[dict]:
    gj = json.loads((root / "metadata.geojson").read_text())
    out = []
    for feat in gj["features"]:
        p = feat["properties"]
        dates = p["dates-S2"]
        if isinstance(dates, str):
            dates = json.loads(dates)
        ordered = [dates[k] for k in sorted(dates, key=lambda k: int(k))]
        out.append({
            "id": int(p["ID_PATCH"]),
            "fold": int(p["Fold"]),
            "dates": [_to_date(d) for d in ordered],
        })
    return out


def _to_date(d) -> date:
    s = str(int(d))
    return date(int(s[:4]), int(s[4:6]), int(s[6:8]))


def load_instances(root: Path, pid: int) -> np.ndarray:
    for rel in (f"INSTANCE_ANNOTATIONS/INSTANCES_{pid}.npy", f"ANNOTATIONS/ParcelIDs_{pid}.npy"):
        f = root / rel
        if f.exists():
            return np.load(f)
    raise FileNotFoundError(f"no instance annotation for patch {pid}")


# ----------------------------------------------------------------------- compositing


def month_windows(start: date, end: date) -> list[tuple[date, date]]:
    wins = []
    y, m = start.year, start.month
    while (y, m) <= (end.year, end.month):
        ny, nm = (y + 1, 1) if m == 12 else (y, m + 1)
        wins.append((date(y, m, 1), date(ny, nm, 1)))
        y, m = ny, nm
    return wins


def composite(series: np.ndarray, dates: list[date], windows: list[tuple[date, date]]
              ) -> np.ndarray:
    """Median per window -> (len(windows), C, H, W). Empty windows reuse the nearest one."""
    d = np.array([x.toordinal() for x in dates])
    out = np.empty((len(windows), *series.shape[1:]), dtype=np.float32)
    have = []
    for i, (a, b) in enumerate(windows):
        sel = (d >= a.toordinal()) & (d < b.toordinal())
        if sel.any():
            out[i] = np.median(series[sel].astype(np.float32), axis=0)
            have.append(i)
    if not have:
        raise ValueError("no acquisitions inside the requested date range")
    for i in range(len(windows)):
        if i not in have:
            out[i] = out[min(have, key=lambda j: abs(j - i))]
    return out


def crop_box(mask: np.ndarray, size: int) -> tuple[int, int]:
    """Top-left corner of a size x size window centred on the parcel, kept in bounds."""
    ys, xs = np.nonzero(mask)
    cy, cx = int(round(ys.mean())), int(round(xs.mean()))
    h, w = mask.shape
    y0 = min(max(cy - size // 2, 0), h - size)
    x0 = min(max(cx - size // 2, 0), w - size)
    return y0, x0


# ----------------------------------------------------------------------------- main


def write_tif(path: Path, arr: np.ndarray) -> None:
    import rasterio

    path.parent.mkdir(parents=True, exist_ok=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with rasterio.open(path, "w", driver="GTiff", height=arr.shape[1], width=arr.shape[2],
                           count=arr.shape[0], dtype="int16", compress="deflate") as f:
            f.write(arr)


def main(argv: list[str] | None = None) -> dict:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", required=True, help="unzipped PASTIS directory")
    p.add_argument("--out", default="data/pastis_parcels")
    p.add_argument("--composite", default="monthly", choices=["median", "monthly"])
    p.add_argument("--start", default="2018-09-01",
                   help="first day (YYYY-MM-DD); default covers the full PASTIS series so "
                        "autumn sowing of winter crops is seen")
    p.add_argument("--end", default="2019-10-31", help="last day")
    p.add_argument("--patch-size", type=int, default=32, help="pixels (10 m) around each parcel")
    p.add_argument("--min-pixels", type=int, default=20, help="skip parcels smaller than this")
    p.add_argument("--mask-outside", action=argparse.BooleanOptionalAction, default=True,
                   help="zero pixels outside the parcel (default on)")
    p.add_argument("--limit-patches", type=int, default=None, help="debug: first N tiles only")
    args = p.parse_args(argv)

    root = find_root(Path(args.root))
    out = Path(args.out)
    start, end = date.fromisoformat(args.start), date.fromisoformat(args.end)
    if args.composite == "median":
        windows = [(start, date.fromordinal(end.toordinal() + 1))]
    else:
        windows = month_windows(start, end)

    meta = read_metadata(root)
    if args.limit_patches:
        meta = meta[: args.limit_patches]
    counts: dict[str, Counter] = {f"fold{f}": Counter() for f in range(1, 6)}
    size = args.patch_size
    missing = 0
    for n, m in enumerate(meta, 1):
        split = f"fold{m['fold']}"
        pid = m["id"]
        if not (root / "DATA_S2" / f"S2_{pid}.npy").exists():
            missing += 1
            continue
        s2 = np.load(root / "DATA_S2" / f"S2_{pid}.npy")             # (T, 10, H, W)
        sem = np.load(root / "ANNOTATIONS" / f"TARGET_{pid}.npy")[0]  # (H, W)
        inst = load_instances(root, pid)
        comp = composite(s2, m["dates"], windows)                     # (W, 10, H, W)
        for iid in np.unique(inst):
            if iid == 0:
                continue
            mask = inst == iid
            if mask.sum() < args.min_pixels:
                continue
            label = int(np.bincount(sem[mask].astype(np.int64)).argmax())
            if label in IGNORE_LABELS:
                continue
            y0, x0 = crop_box(mask, size)
            patch = comp[:, :, y0:y0 + size, x0:x0 + size]
            if args.mask_outside:
                patch = patch * mask[None, None, y0:y0 + size, x0:x0 + size]
            arr = np.clip(np.round(patch), -32768, 32767).astype(np.int16)
            arr = arr.reshape(-1, size, size)                         # time-major bands
            name = PASTIS_CLASSES[label]
            write_tif(out / split / name / f"{pid}_{int(iid)}.tif", arr)
            counts[split][name] += 1
        if n % 100 == 0 or n == len(meta):
            print(f"[{n}/{len(meta)}] parcels so far: "
                  f"{ {k: sum(v.values()) for k, v in counts.items()} }")

    if missing:
        print(f"warning: {missing} tiles listed in metadata.geojson had no S2 file (skipped)")
    info = {
        "source": "PASTIS",
        "layout": "folds",
        "classes": [c for i, c in enumerate(PASTIS_CLASSES) if i not in IGNORE_LABELS],
        "file_bands": list(PASTIS_S2_BANDS),
        "timesteps": len(windows),
        "windows": [[a.isoformat(), b.isoformat()] for a, b in windows],
        "composite": args.composite,
        "patch_size": size,
        "mask_outside": args.mask_outside,
        "counts": {k: dict(sorted(v.items())) for k, v in counts.items()},
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "cropcls_meta.json").write_text(json.dumps(info, indent=2))
    print(json.dumps(info["counts"], indent=2))
    return info


if __name__ == "__main__":
    main()
