"""Fast CPU tests on synthetic GeoTIFFs (no downloads)."""

import warnings

import numpy as np
import pytest
import torch

rasterio = pytest.importorskip("rasterio")

from cropcls.bands import S2_ALL_BANDS, S2_L2A_BANDS, parse_bands
from cropcls.data import CropDataModule, stratified_split
from cropcls.model import build_model
from cropcls.module import CropClassifier, random_dihedral


def _write_fake(root, classes=("maize", "rice", "wheat"), n=12, bands=12, size=24):
    from rasterio.transform import from_origin

    rng = np.random.default_rng(0)
    for ci, c in enumerate(classes):
        (root / c).mkdir(parents=True)
        for i in range(n):
            a = (rng.random((bands, size, size)) * 1000 + ci * 1500).astype("uint16")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                with rasterio.open(root / c / f"{i}.tif", "w", driver="GTiff", height=size,
                                   width=size, count=bands, dtype="uint16", crs="EPSG:4326",
                                   transform=from_origin(0, 0, 1, 1)) as f:
                    f.write(a)


def test_parse_bands():
    assert parse_bands("all") == S2_ALL_BANDS
    assert parse_bands("b2,B3,b8a") == ("B02", "B03", "B8A")
    with pytest.raises(ValueError):
        parse_bands("B13")


def test_stratified_split_keeps_every_class():
    samples = [(f"{c}{i}", c) for c in range(4) for i in range(10)]
    s = stratified_split(samples, 0.15, 0.15, 0)
    for split in ("train", "val", "test"):
        assert {y for _, y in s[split]} == {0, 1, 2, 3}
    assert sum(len(v) for v in s.values()) == 40


def test_band_subset_model_shapes():
    m = build_model(5, bands=("B02", "B03", "B04", "B08"), weights=None)
    assert m(torch.zeros(2, 4, 64, 64)).shape == (2, 5)


def test_dihedral_preserves_values():
    x = torch.arange(2 * 3 * 8 * 8, dtype=torch.float32).reshape(2, 3, 8, 8)
    y = random_dihedral(x)
    assert y.shape == x.shape
    assert torch.equal(y.flatten(1).sort(1).values, x.flatten(1).sort(1).values)


def test_folder_datamodule_and_forward(tmp_path):
    _write_fake(tmp_path)
    dm = CropDataModule("folder", str(tmp_path), bands=parse_bands("B02,B03,B04,B08"),
                        file_bands=S2_L2A_BANDS, patch_size=32, batch_size=4, num_workers=0)
    assert dm.classes == ["maize", "rice", "wheat"]
    x, y = next(iter(dm.train_dataloader()))
    assert x.shape == (4, 4, 32, 32)
    model = CropClassifier(3, dm.classes, bands=dm.bands, weights=None, img_size=48)
    assert model(x).shape == (4, 3)


def test_train_end_to_end(tmp_path):
    from cropcls.train import main

    data = tmp_path / "data"
    _write_fake(data)
    report = main(["--dataset", "folder", "--data-root", str(data), "--file-bands", "l2a",
                   "--bands", "l2a", "--weights", "none", "--epochs", "2",
                   "--freeze-epochs", "1", "--batch-size", "8", "--num-workers", "0",
                   "--img-size", "32", "--patch-size", "24", "--class-weights",
                   "--out-dir", str(tmp_path / "runs"), "--name", "t",
                   "--accelerator", "cpu"])
    assert set(report["per_class"]) == {"maize", "rice", "wheat"}
    assert (tmp_path / "runs" / "t" / "confusion_matrix.csv").exists()
