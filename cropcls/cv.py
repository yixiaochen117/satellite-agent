"""Official 5-fold cross-validation on a fold-organised folder (e.g. PASTIS parcels).

Trains one model per official rotation (train 3 folds / val 1 / test 1), then pools
the five test confusion matrices and reports OA and mIoU, the way the PASTIS benchmark
reports parcel classification::

    python -m cropcls.cv --data-root data/pastis_parcels --name pastis-cv -- --class-weights

Everything after ``--`` is passed to ``cropcls.train``.

Reference (PASTIS-R paper, Table 1, S2 only, 5-fold, 18 classes): PSE+LTAE
OA 91.7 / mIoU 73.9 (114k params).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import torch

from .reports import print_report, write_reports
from .train import main as train_main

BASELINE = {"model": "PSE+LTAE (S2, PASTIS-R paper Table 1)", "OA": 0.917, "mIoU": 0.739}


def read_confmat(path: Path) -> tuple[list[str], torch.Tensor]:
    with open(path) as f:
        rows = list(csv.reader(f))
    classes = rows[0][1:]
    cm = torch.tensor([[int(v) for v in r[1:]] for r in rows[1:]], dtype=torch.long)
    return classes, cm


def main(argv: list[str] | None = None) -> dict:
    argv = list(sys.argv[1:] if argv is None else argv)
    extra: list[str] = []
    if "--" in argv:
        i = argv.index("--")
        argv, extra = argv[:i], argv[i + 1:]
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", required=True)
    p.add_argument("--name", default="cv")
    p.add_argument("--out-dir", default="runs")
    p.add_argument("--folds", default="1,2,3,4,5", help="which rotations to run")
    args = p.parse_args(argv)

    out = Path(args.out_dir) / args.name
    folds = [int(f) for f in args.folds.split(",") if f.strip()]
    total, classes, per_fold = None, None, {}
    for k in folds:
        print(f"\n===== CV rotation {k}/5 =====")
        run_name = f"{args.name}/fold{k}"
        rep = train_main([*extra, "--dataset", "folder", "--data-root", args.data_root,
                          "--cv-fold", str(k), "--out-dir", args.out_dir, "--name", run_name])
        c, cm = read_confmat(Path(args.out_dir) / run_name / "confusion_matrix.csv")
        if classes is None:
            classes, total = c, torch.zeros_like(cm)
        total += cm
        per_fold[k] = rep["confmat_summary"]

    report = write_reports(out, classes, {}, torch.zeros(len(classes)), total)
    report["per_fold"] = per_fold
    report["baseline"] = BASELINE
    (out / "cv_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print_report(report)
    s = report["confmat_summary"]
    print(f"\npooled over folds {folds}: OA {s['OA']:.3f}  mIoU {s['mIoU']:.3f}   "
          f"(baseline {BASELINE['model']}: OA {BASELINE['OA']:.3f}  mIoU {BASELINE['mIoU']:.3f})")
    return report


if __name__ == "__main__":
    main()
