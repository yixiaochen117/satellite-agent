"""Write evaluation reports (JSON, CSV confusion matrix, PNG)."""

from __future__ import annotations

import csv
import json
from collections.abc import Sequence
from pathlib import Path

import torch


def write_reports(out_dir: str | Path, classes: Sequence[str], metrics: dict,
                  per_class_acc: torch.Tensor, confmat: torch.Tensor) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cm = confmat.long()
    support = cm.sum(1)
    predicted = cm.sum(0)
    tp = cm.diag()
    per_class = {}
    for i, name in enumerate(classes):
        recall = (tp[i] / support[i]).item() if support[i] else float("nan")
        precision = (tp[i] / predicted[i]).item() if predicted[i] else float("nan")
        f1 = (2 * precision * recall / (precision + recall)
              if precision == precision and recall == recall and precision + recall > 0
              else float("nan"))
        per_class[name] = {"precision": precision, "recall": recall, "f1": f1,
                           "support": int(support[i])}
    report = {"metrics": {k: float(v) for k, v in metrics.items()}, "per_class": per_class}
    (out / "test_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))

    with open(out / "confusion_matrix.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["true\\pred", *classes])
        for i, name in enumerate(classes):
            w.writerow([name, *cm[i].tolist()])

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        norm = cm.float() / cm.sum(1, keepdim=True).clamp(min=1)
        n = len(classes)
        fig, ax = plt.subplots(figsize=(1.0 + 0.7 * n, 0.8 + 0.6 * n))
        im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(n), classes, rotation=45, ha="right")
        ax.set_yticks(range(n), classes)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        for i in range(n):
            for j in range(n):
                v = norm[i, j].item()
                if v >= 0.005:
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8,
                            color="white" if v > 0.5 else "black")
        fig.colorbar(im, ax=ax, fraction=0.046)
        ax.set_title("Test confusion matrix (row-normalised)")
        fig.tight_layout()
        fig.savefig(out / "confusion_matrix.png", dpi=150)
        plt.close(fig)
    except Exception as e:  # plotting is optional
        print(f"(skipped confusion matrix plot: {e})")

    return report


def print_report(report: dict) -> None:
    m = report["metrics"]
    print("\n== Test metrics ==")
    for k, v in m.items():
        print(f"  {k:16s} {v:.4f}")
    print("\n  {:24s} {:>9s} {:>7s} {:>7s} {:>8s}".format("class", "precision", "recall",
                                                       "f1", "support"))
    for name, r in report["per_class"].items():
        print(f"  {name:24s} {r['precision']:9.3f} {r['recall']:7.3f} {r['f1']:7.3f} "
              f"{r['support']:8d}")
