"""Datasets and the Lightning DataModule.

Every dataset returns ``(image, label)`` where ``image`` is a float32 tensor of shape
``(C, H, W)`` holding raw Sentinel-2 digital numbers (reflectance * 10000) in the band
order given by ``bands``. Scaling, augmentation and resizing happen on the GPU inside
the model (see ``cropcls.module``), which keeps the data loaders cheap.

Supported sources:

* ``eurosat``    - EuroSAT multispectral (27k 64x64 patches, 10 classes, 13 bands).
* ``eurosat100`` - 100-image EuroSAT subset, for smoke tests only.
* ``folder``     - your own labelled GeoTIFF patches, laid out as
  ``<root>/<class>/*.tif`` or ``<root>/{train,val,test}/<class>/*.tif``.
"""

from __future__ import annotations

import json
import os
import random
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

import lightning as L
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from .bands import S2_ALL_BANDS

TIF_SUFFIXES = (".tif", ".tiff")

EUROSAT_CLASSES = (
    "AnnualCrop", "Forest", "HerbaceousVegetation", "Highway", "Industrial",
    "Pasture", "PermanentCrop", "Residential", "River", "SeaLake",
)

# Classes most relevant for agriculture; handy with --classes agri
EUROSAT_AGRI_CLASSES = ("AnnualCrop", "PermanentCrop", "Pasture", "HerbaceousVegetation")


# --------------------------------------------------------------------------- EuroSAT


class EuroSATDataset(Dataset):
    """Thin wrapper over ``torchgeo.datasets.EuroSAT`` returning ``(image, label)``.

    TorchGeo names EuroSAT's bands with B8A last, which is how they are stored in the
    GeoTIFFs. Passing ``bands`` in the SSL4EO order makes TorchGeo reorder them so they
    line up with the pretrained weights.
    """

    def __init__(
        self,
        root: str | os.PathLike,
        split: str,
        bands: Sequence[str] = S2_ALL_BANDS,
        small: bool = False,
        download: bool = False,
    ) -> None:
        from torchgeo.datasets import EuroSAT, EuroSAT100

        cls = EuroSAT100 if small else EuroSAT
        self.ds = cls(root=str(root), split=split, bands=tuple(bands), download=download)
        self.classes = list(self.ds.classes)
        self.targets = [int(t) for t in self.ds.targets]

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        sample = self.ds[idx]
        return sample["image"].float(), int(sample["label"])


# --------------------------------------------------------------------- GeoTIFF folder


def _list_class_dirs(root: Path) -> list[str]:
    return sorted(p.name for p in root.iterdir() if p.is_dir() and not p.name.startswith("."))


def _list_tifs(d: Path) -> list[Path]:
    return sorted(p for p in d.rglob("*") if p.suffix.lower() in TIF_SUFFIXES)


def scan_folder(root: str | os.PathLike, classes: Sequence[str] | None = None
                ) -> tuple[list[str], list[tuple[Path, int]]]:
    """Scan ``<root>/<class>/*.tif`` into a sorted class list and (path, label) pairs."""
    root = Path(root)
    classes = list(classes) if classes else _list_class_dirs(root)
    samples: list[tuple[Path, int]] = []
    for label, name in enumerate(classes):
        d = root / name
        if not d.is_dir():
            continue
        samples += [(p, label) for p in _list_tifs(d)]
    if not samples:
        raise FileNotFoundError(f"No .tif files found under {root}/<class>/")
    return classes, samples


def stratified_split(samples: list[tuple[Path, int]], val_frac: float, test_frac: float,
                     seed: int) -> dict[str, list[tuple[Path, int]]]:
    """Split per class so small crop classes appear in every split."""
    rng = random.Random(seed)
    by_class: dict[int, list[tuple[Path, int]]] = {}
    for s in samples:
        by_class.setdefault(s[1], []).append(s)
    out: dict[str, list[tuple[Path, int]]] = {"train": [], "val": [], "test": []}
    for items in by_class.values():
        items = items[:]
        rng.shuffle(items)
        n = len(items)
        n_test = int(round(n * test_frac))
        n_val = int(round(n * val_frac))
        if n >= 3:  # keep at least one sample of each class in val and test
            n_test, n_val = max(n_test, 1), max(n_val, 1)
        out["test"] += items[:n_test]
        out["val"] += items[n_test:n_test + n_val]
        out["train"] += items[n_test + n_val:]
    return out


class GeoTiffFolderDataset(Dataset):
    """Labelled GeoTIFF patches.

    Args:
        samples: list of ``(path, label)``.
        file_bands: the band names stored in each file, in file order.
        bands: the bands to feed the model (subset/reorder of ``file_bands``).
        patch_size: every patch is resized to this size so they can be batched.
        scale_to_dn: multiply file values by this (e.g. 10000 if your files hold
            0-1 reflectance) so they match the 0-10000 range the weights expect.
    """

    def __init__(self, samples: list[tuple[Path, int]], classes: Sequence[str],
                 file_bands: Sequence[str], bands: Sequence[str], patch_size: int = 64,
                 scale_to_dn: float = 1.0) -> None:
        missing = [b for b in bands if b not in file_bands]
        if missing:
            raise ValueError(f"Bands {missing} requested but files only have {list(file_bands)}")
        self.samples = samples
        self.classes = list(classes)
        self.targets = [s[1] for s in samples]
        self.index = [list(file_bands).index(b) for b in bands]
        self.n_file_bands = len(file_bands)
        self.patch_size = patch_size
        self.scale_to_dn = scale_to_dn

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        import rasterio

        path, label = self.samples[idx]
        with rasterio.open(path) as f:
            if f.count != self.n_file_bands:
                raise ValueError(f"{path} has {f.count} bands, expected {self.n_file_bands} "
                                 "(set --file-bands to match your files)")
            arr = f.read(indexes=[i + 1 for i in self.index]).astype(np.float32)
        x = torch.from_numpy(arr)
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0) * self.scale_to_dn
        if x.shape[-2:] != (self.patch_size, self.patch_size):
            x = F.interpolate(x[None], size=(self.patch_size, self.patch_size),
                              mode="bilinear", align_corners=False)[0]
        return x, label


# ------------------------------------------------------------------- class filtering


class ClassSubset(Dataset):
    """Keep only some classes and remap labels to 0..K-1."""

    def __init__(self, ds: Dataset, keep: Sequence[str]) -> None:
        all_classes = list(ds.classes)
        unknown = [c for c in keep if c not in all_classes]
        if unknown:
            raise ValueError(f"Unknown classes {unknown}; available: {all_classes}")
        self.ds = ds
        self.classes = list(keep)
        old_to_new = {all_classes.index(c): i for i, c in enumerate(keep)}
        self.indices = [i for i, t in enumerate(ds.targets) if t in old_to_new]
        self.map = old_to_new
        self.targets = [old_to_new[ds.targets[i]] for i in self.indices]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        x, y = self.ds[self.indices[idx]]
        return x, self.map[y]


# ------------------------------------------------------------------------ DataModule


class CropDataModule(L.LightningDataModule):
    def __init__(
        self,
        dataset: str = "eurosat",
        data_root: str = "data/eurosat",
        bands: Sequence[str] = S2_ALL_BANDS,
        classes: Sequence[str] | None = None,
        file_bands: Sequence[str] | None = None,
        patch_size: int = 64,
        scale_to_dn: float = 1.0,
        val_frac: float = 0.15,
        test_frac: float = 0.15,
        batch_size: int = 128,
        num_workers: int = 8,
        seed: int = 42,
        download: bool = False,
    ) -> None:
        super().__init__()
        self.dataset = dataset
        self.data_root = data_root
        self.bands = tuple(bands)
        self.keep_classes = list(classes) if classes else None
        self.file_bands = tuple(file_bands) if file_bands else self.bands
        self.patch_size = patch_size
        self.scale_to_dn = scale_to_dn
        self.val_frac, self.test_frac = val_frac, test_frac
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed
        self.download = download
        self.splits: dict[str, Dataset] = {}

    # Build all three splits eagerly so class names are known before the model exists.
    def build(self) -> None:
        if self.splits:
            return
        if self.dataset in ("eurosat", "eurosat100"):
            small = self.dataset == "eurosat100"
            for split in ("train", "val", "test"):
                self.splits[split] = EuroSATDataset(self.data_root, split, self.bands,
                                                    small=small, download=self.download)
        elif self.dataset == "folder":
            root = Path(self.data_root)
            kw = dict(file_bands=self.file_bands, bands=self.bands,
                      patch_size=self.patch_size, scale_to_dn=self.scale_to_dn)
            if (root / "train").is_dir():
                classes, train = scan_folder(root / "train")
                parts = {"train": train}
                for split in ("val", "test"):
                    if (root / split).is_dir():
                        parts[split] = scan_folder(root / split, classes)[1]
                if "val" not in parts:
                    sub = stratified_split(train, self.val_frac, 0.0, self.seed)
                    parts["train"], parts["val"] = sub["train"], sub["val"]
                parts.setdefault("test", parts["val"])
            else:
                classes, samples = scan_folder(root)
                parts = stratified_split(samples, self.val_frac, self.test_frac, self.seed)
            for split, s in parts.items():
                self.splits[split] = GeoTiffFolderDataset(s, classes, **kw)
        else:
            raise ValueError(f"Unknown dataset {self.dataset!r}")

        if self.keep_classes:
            keep = self.keep_classes
            if keep == ["agri"]:
                keep = list(EUROSAT_AGRI_CLASSES)
            self.splits = {k: ClassSubset(v, keep) for k, v in self.splits.items()}

    @property
    def classes(self) -> list[str]:
        self.build()
        return list(self.splits["train"].classes)

    def class_counts(self) -> list[int]:
        self.build()
        c = Counter(self.splits["train"].targets)
        return [c.get(i, 0) for i in range(len(self.classes))]

    def summary(self) -> dict:
        self.build()
        return {
            "dataset": self.dataset,
            "bands": list(self.bands),
            "classes": self.classes,
            "sizes": {k: len(v) for k, v in self.splits.items()},
            "train_class_counts": dict(zip(self.classes, self.class_counts())),
        }

    def setup(self, stage: str | None = None) -> None:
        self.build()

    def _loader(self, split: str, shuffle: bool) -> DataLoader:
        return DataLoader(
            self.splits[split],
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=self.num_workers > 0,
            drop_last=shuffle and len(self.splits[split]) > self.batch_size,
        )

    def train_dataloader(self) -> DataLoader:
        return self._loader("train", True)

    def val_dataloader(self) -> DataLoader:
        return self._loader("val", False)

    def test_dataloader(self) -> DataLoader:
        return self._loader("test", False)


def save_summary(dm: CropDataModule, path: str | os.PathLike) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(dm.summary(), indent=2, ensure_ascii=False))
