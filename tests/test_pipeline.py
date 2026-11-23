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


def _write_fake_pastis(root, n_patches=5, size=64, n_dates=12):
    import json as _json

    rng = np.random.default_rng(1)
    for d in ("DATA_S2", "ANNOTATIONS", "INSTANCE_ANNOTATIONS"):
        (root / d).mkdir(parents=True)
    feats = []
    for pid in range(10, 10 + n_patches):
        # 4 square parcels per tile with labels 1..4, one tiny parcel, one void parcel
        inst = np.zeros((size, size), dtype=np.int64)
        sem = np.zeros((size, size), dtype=np.uint8)
        for k, (y, x) in enumerate([(4, 4), (4, 36), (36, 4), (36, 36)]):
            inst[y:y + 20, x:x + 20] = k + 1
            sem[y:y + 20, x:x + 20] = k + 1
        inst[0:2, 60:62], sem[0:2, 60:62] = 9, 1       # too small
        inst[60:64, 0:10], sem[60:64, 0:10] = 8, 19    # void
        s2 = (rng.random((n_dates, 10, size, size)) * 500
              + sem[None, None].astype(np.float32) * 300).astype(np.int16)
        np.save(root / "DATA_S2" / f"S2_{pid}.npy", s2)
        np.save(root / "ANNOTATIONS" / f"TARGET_{pid}.npy", np.stack([sem] * 3))
        np.save(root / "INSTANCE_ANNOTATIONS" / f"INSTANCES_{pid}.npy", inst)
        dates = {str(i): int(f"2019{(i % 12) + 1:02d}15") for i in range(n_dates)}
        feats.append({"type": "Feature", "geometry": None,
                      "properties": {"ID_PATCH": pid, "Fold": pid - 9,
                                     "dates-S2": dates}})
    (root / "metadata.geojson").write_text(_json.dumps({"type": "FeatureCollection",
                                                         "features": feats}))


@pytest.mark.parametrize("mode,timesteps", [("median", 1), ("monthly", 14)])
def test_pastis_convert_and_train(tmp_path, mode, timesteps):
    from cropcls.prepare_pastis import main as prep
    from cropcls.train import main as train

    _write_fake_pastis(tmp_path / "PASTIS")
    info = prep(["--root", str(tmp_path), "--out", str(tmp_path / "parcels"),
                 "--composite", mode, "--patch-size", "24"])
    assert info["timesteps"] == timesteps
    assert len(info["classes"]) == 18
    # 4 valid parcels per tile, one tile per fold; tiny and void parcels dropped
    for f in range(1, 6):
        assert sum(info["counts"][f"fold{f}"].values()) == 4
    assert "void_label" not in info["counts"]["fold1"]

    report = train(["--dataset", "folder", "--data-root", str(tmp_path / "parcels"),
                    "--cv-fold", "2", "--weights", "none", "--epochs", "2",
                    "--freeze-epochs", "1", "--batch-size", "4", "--num-workers", "0",
                    "--img-size", "32", "--out-dir", str(tmp_path / "runs"), "--name", "p",
                    "--accelerator", "cpu"])
    assert len(report["per_class"]) == 18
    assert report["confmat_summary"]["n"] == 4  # rotation 2 tests on fold 1
    assert 0.0 <= report["confmat_summary"]["mIoU"] <= 1.0


def test_pastis_cv_pools_folds(tmp_path):
    from cropcls.cv import main as cv
    from cropcls.prepare_pastis import main as prep

    _write_fake_pastis(tmp_path / "PASTIS")
    prep(["--root", str(tmp_path), "--out", str(tmp_path / "parcels"),
          "--composite", "median", "--patch-size", "24"])
    report = cv(["--data-root", str(tmp_path / "parcels"), "--name", "cv",
                 "--out-dir", str(tmp_path / "runs"), "--",
                 "--weights", "none", "--epochs", "1", "--freeze-epochs", "0",
                 "--batch-size", "4", "--num-workers", "0", "--img-size", "32",
                 "--accelerator", "cpu"])
    assert report["confmat_summary"]["n"] == 20  # every parcel tested exactly once
    assert set(report["per_fold"]) == {1, 2, 3, 4, 5}


def test_confmat_summary_matches_definition():
    from cropcls.reports import confmat_summary

    cm = torch.tensor([[5, 1, 0], [2, 2, 0], [0, 0, 0]])
    s = confmat_summary(cm)
    assert abs(s["OA"] - 7 / 10) < 1e-9
    # class 2 absent from truth and prediction -> skipped
    assert abs(s["mIoU"] - ((5 / 8) + (2 / 5)) / 2) < 1e-9
