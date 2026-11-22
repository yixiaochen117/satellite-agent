"""Classify GeoTIFF patches with a trained checkpoint and write a CSV.

    python -m cropcls.predict runs/<run>/checkpoints/best-XX.ckpt path/to/tifs \
        --file-bands l2a --out predictions.csv

Each file must contain the bands listed in ``--file-bands`` (default: the bands the
model was trained on), as Sentinel-2 DN (reflectance * 10000) unless ``--scale-to-dn``
is set.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .bands import parse_bands
from .data import TIF_SUFFIXES, GeoTiffFolderDataset
from .module import CropClassifier


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("checkpoint")
    p.add_argument("inputs", nargs="+", help="GeoTIFF files or directories")
    p.add_argument("--file-bands", default=None)
    p.add_argument("--scale-to-dn", type=float, default=1.0)
    p.add_argument("--patch-size", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--top-k", type=int, default=3)
    p.add_argument("--out", default="predictions.csv")
    args = p.parse_args(argv)

    files: list[Path] = []
    for inp in args.inputs:
        q = Path(inp)
        files += (sorted(f for f in q.rglob("*") if f.suffix.lower() in TIF_SUFFIXES)
                  if q.is_dir() else [q])
    if not files:
        raise SystemExit("no GeoTIFFs found")

    model = CropClassifier.load_from_checkpoint(args.checkpoint, map_location="cpu",
                                                weights=None)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.eval().to(dev)
    bands = tuple(model.hparams.bands)
    classes = list(model.hparams.class_names)
    file_bands = parse_bands(args.file_bands) if args.file_bands else bands

    ds = GeoTiffFolderDataset([(f, -1) for f in files], classes, file_bands, bands,
                              args.patch_size, args.scale_to_dn)
    dl = DataLoader(ds, batch_size=args.batch_size, num_workers=args.num_workers)
    k = min(args.top_k, len(classes))
    rows = []
    i = 0
    with torch.inference_mode(), torch.autocast(dev, dtype=torch.bfloat16,
                                                enabled=dev == "cuda"):
        for x, _ in dl:
            prob = model(x.to(dev)).float().softmax(-1).cpu()
            top = prob.topk(k, dim=-1)
            for j in range(prob.shape[0]):
                row = {"file": str(files[i])}
                for r in range(k):
                    row[f"top{r + 1}"] = classes[top.indices[j, r]]
                    row[f"p{r + 1}"] = round(top.values[j, r].item(), 4)
                rows.append(row)
                i += 1
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} predictions to {args.out}")


if __name__ == "__main__":
    main()
