"""Evaluate a trained checkpoint on the test split.

    python -m cropcls.evaluate runs/<run>/checkpoints/best-XX.ckpt

Dataset settings are read from the run's ``args.json`` unless overridden.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import lightning as L
import torch

from .bands import parse_bands
from .data import CropDataModule
from .module import CropClassifier
from .reports import print_report, write_reports


def load_run_args(ckpt: Path) -> dict:
    for d in (ckpt.parent.parent, ckpt.parent):
        f = d / "args.json"
        if f.exists():
            return json.loads(f.read_text())
    return {}


def main(argv: list[str] | None = None) -> dict:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("checkpoint")
    p.add_argument("--data-root", default=None)
    p.add_argument("--split", default="test", choices=["val", "test"])
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--out-dir", default=None, help="default: <run>/eval-<split>")
    args = p.parse_args(argv)

    ckpt = Path(args.checkpoint)
    run = load_run_args(ckpt)
    model = CropClassifier.load_from_checkpoint(ckpt, map_location="cpu", weights=None)
    hp = model.hparams
    dataset = run.get("dataset", "eurosat")
    dm = CropDataModule(
        dataset=dataset,
        data_root=args.data_root or run.get("data_root") or f"data/{dataset}",
        bands=tuple(hp.bands), classes=run.get("classes"),
        file_bands=parse_bands(run["file_bands"]) if run.get("file_bands") else None,
        patch_size=run.get("patch_size") or 64, scale_to_dn=run.get("scale_to_dn", 1.0),
        timesteps=int(hp.get("timesteps", 1)),
        val_frac=run.get("val_frac", 0.15), test_frac=run.get("test_frac", 0.15),
        seed=run.get("seed", 42), batch_size=args.batch_size, num_workers=args.num_workers,
    )
    if dm.classes != list(hp.class_names):
        raise SystemExit(f"Class mismatch: checkpoint {list(hp.class_names)} vs data {dm.classes}")
    dm.build()
    if args.split == "val":
        dm.splits["test"] = dm.splits["val"]

    precision = "bf16-mixed" if torch.cuda.is_available() else "32-true"
    trainer = L.Trainer(accelerator="auto", devices=1, precision=precision, logger=False)
    results = trainer.test(model, datamodule=dm)[0]
    out = Path(args.out_dir) if args.out_dir else ckpt.parent.parent / f"eval-{args.split}"
    report = write_reports(out, list(hp.class_names), results, model.last_test_per_class,
                           model.last_test_confmat)
    print_report(report)
    print(f"\nreports written to {out}")
    return report


if __name__ == "__main__":
    main()
