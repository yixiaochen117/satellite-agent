"""Train a crop / land-cover classifier.

Examples::

    # EuroSAT, 13 bands, ResNet-18 SSL4EO MoCo weights (default)
    python -m cropcls.train --dataset eurosat --data-root data/eurosat --download

    # Only the agricultural EuroSAT classes
    python -m cropcls.train --dataset eurosat --classes agri

    # Your own labelled patches: data/mycrops/<class>/*.tif with 12 L2A bands
    python -m cropcls.train --dataset folder --data-root data/mycrops \
        --file-bands l2a --bands l2a --class-weights
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import lightning as L
import torch
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from lightning.pytorch.loggers import CSVLogger

from .bands import parse_bands
from .data import CropDataModule, save_summary
from .model import ARCHS
from .module import CropClassifier
from .reports import print_report, write_reports


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    d = p.add_argument_group("data")
    d.add_argument("--dataset", default="eurosat", choices=["eurosat", "eurosat100", "folder"])
    d.add_argument("--data-root", default=None,
                   help="dataset directory (default: data/<dataset>)")
    d.add_argument("--download", action="store_true", help="download EuroSAT if missing")
    d.add_argument("--bands", default="all",
                   help="bands fed to the model: all | l2a | rgb | comma list (e.g. B02,B03,B04,B08)")
    d.add_argument("--file-bands", default=None,
                   help="[folder] bands stored in each GeoTIFF, in file order (default = --bands)")
    d.add_argument("--scale-to-dn", type=float, default=1.0,
                   help="[folder] multiply file values by this; use 10000 if files hold 0-1 reflectance")
    d.add_argument("--patch-size", type=int, default=64, help="[folder] patches resized to this")
    d.add_argument("--classes", nargs="+", default=None,
                   help="keep only these classes ('agri' = EuroSAT crop-related classes)")
    d.add_argument("--val-frac", type=float, default=0.15)
    d.add_argument("--test-frac", type=float, default=0.15)
    d.add_argument("--batch-size", type=int, default=128)
    d.add_argument("--num-workers", type=int, default=8)

    m = p.add_argument_group("model")
    m.add_argument("--arch", default="resnet18", choices=ARCHS)
    m.add_argument("--weights", default="SENTINEL2_ALL_MOCO",
                   help="TorchGeo weight name (SENTINEL2_ALL_MOCO, SENTINEL2_ALL_DINO, ...) or 'none'")
    m.add_argument("--img-size", type=int, default=224,
                   help="input size on GPU; 224 matches pretraining, 128 is ~3x faster")
    m.add_argument("--drop-rate", type=float, default=0.2)

    o = p.add_argument_group("optimisation")
    o.add_argument("--epochs", type=int, default=30)
    o.add_argument("--freeze-epochs", type=int, default=5,
                   help="epochs training only the head before unfreezing the backbone")
    o.add_argument("--lr", type=float, default=1e-3, help="head learning rate")
    o.add_argument("--backbone-lr-mult", type=float, default=0.1,
                   help="backbone lr = lr * this, after unfreezing")
    o.add_argument("--weight-decay", type=float, default=1e-4)
    o.add_argument("--warmup-epochs", type=float, default=1.0)
    o.add_argument("--label-smoothing", type=float, default=0.1)
    o.add_argument("--class-weights", action="store_true",
                   help="inverse-frequency class weights in the loss (imbalanced crop data)")
    o.add_argument("--patience", type=int, default=0,
                   help="early-stop after this many epochs without val/f1 gain (0 = off)")

    r = p.add_argument_group("runtime")
    r.add_argument("--precision", default=None,
                   help="default: bf16-mixed on CUDA, 32 on CPU")
    r.add_argument("--accelerator", default="auto")
    r.add_argument("--devices", default="auto")
    r.add_argument("--compile", action="store_true", help="torch.compile the model")
    r.add_argument("--seed", type=int, default=42)
    r.add_argument("--out-dir", default="runs")
    r.add_argument("--name", default=None, help="run name (default: <dataset>-<arch>-<time>)")
    r.add_argument("--limit-batches", type=float, default=None,
                   help="debug: limit train/val/test batches per epoch")
    return p


def main(argv: list[str] | None = None) -> dict:
    args = build_parser().parse_args(argv)
    L.seed_everything(args.seed, workers=True)
    torch.set_float32_matmul_precision("high")

    bands = parse_bands(args.bands)
    file_bands = parse_bands(args.file_bands) if args.file_bands else None
    data_root = args.data_root or f"data/{args.dataset}"
    weights = None if str(args.weights).lower() == "none" else args.weights

    dm = CropDataModule(
        dataset=args.dataset, data_root=data_root, bands=bands, classes=args.classes,
        file_bands=file_bands, patch_size=args.patch_size, scale_to_dn=args.scale_to_dn,
        val_frac=args.val_frac, test_frac=args.test_frac, batch_size=args.batch_size,
        num_workers=args.num_workers, seed=args.seed, download=args.download,
    )
    classes = dm.classes
    name = args.name or f"{args.dataset}-{args.arch}-{time.strftime('%Y%m%d-%H%M%S')}"
    run_dir = Path(args.out_dir) / name
    run_dir.mkdir(parents=True, exist_ok=True)
    save_summary(dm, run_dir / "data_summary.json")
    (run_dir / "args.json").write_text(json.dumps(vars(args), indent=2))
    print(json.dumps(dm.summary(), indent=2, ensure_ascii=False))

    class_weights = None
    if args.class_weights:
        counts = torch.tensor(dm.class_counts(), dtype=torch.float32).clamp(min=1)
        w = counts.sum() / (len(counts) * counts)
        class_weights = (w / w.mean()).tolist()

    model = CropClassifier(
        num_classes=len(classes), class_names=classes, bands=bands, arch=args.arch,
        weights=weights, img_size=args.img_size, lr=args.lr,
        backbone_lr_mult=args.backbone_lr_mult, weight_decay=args.weight_decay,
        freeze_epochs=args.freeze_epochs, warmup_epochs=args.warmup_epochs,
        label_smoothing=args.label_smoothing, drop_rate=args.drop_rate,
        class_weights=class_weights,
    )
    if args.compile:
        model.model.compile()  # in-place: keeps checkpoint keys unchanged

    precision = args.precision or ("bf16-mixed" if torch.cuda.is_available() else "32-true")
    ckpt = ModelCheckpoint(dirpath=run_dir / "checkpoints", filename="best-{epoch:02d}",
                           monitor="val/f1", mode="max", save_last=True,
                           auto_insert_metric_name=False)
    callbacks = [ckpt, LearningRateMonitor("step")]
    if args.patience > 0:
        callbacks.append(EarlyStopping("val/f1", mode="max", patience=args.patience))

    limit = args.limit_batches
    trainer = L.Trainer(
        max_epochs=args.epochs, accelerator=args.accelerator, devices=args.devices,
        precision=precision, callbacks=callbacks,
        logger=CSVLogger(run_dir, name="", version=""), default_root_dir=run_dir,
        log_every_n_steps=20, deterministic=False,
        limit_train_batches=limit, limit_val_batches=limit, limit_test_batches=limit,
    )
    trainer.fit(model, datamodule=dm)

    best = ckpt.best_model_path or None
    print(f"\nbest checkpoint: {best}")
    results = trainer.test(model, datamodule=dm, ckpt_path=best or "last")[0]
    report = write_reports(run_dir, classes, results, model.last_test_per_class,
                           model.last_test_confmat)
    print_report(report)
    print(f"\nreports written to {run_dir}")
    return report


if __name__ == "__main__":
    main()
