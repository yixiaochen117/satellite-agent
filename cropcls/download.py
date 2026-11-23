"""Pre-download datasets and pretrained weights.

    python -m cropcls.download --dataset eurosat --root data/eurosat
"""

from __future__ import annotations

import argparse

from .data import EuroSATDataset
from .model import _weights_enum


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", default="eurosat", choices=["eurosat", "eurosat100", "none"])
    p.add_argument("--root", default=None)
    p.add_argument("--arch", default="resnet18")
    p.add_argument("--weights", default="SENTINEL2_ALL_MOCO")
    args = p.parse_args(argv)

    if args.dataset != "none":
        root = args.root or f"data/{args.dataset}"
        for split in ("train", "val", "test"):
            ds = EuroSATDataset(root, split, small=args.dataset == "eurosat100", download=True)
            print(f"{args.dataset} {split}: {len(ds)} images")

    if args.weights and args.weights.lower() != "none":
        w = getattr(_weights_enum(args.arch), args.weights)
        w.get_state_dict(progress=True, check_hash=True, weights_only=True)
        print(f"weights {args.arch}/{args.weights} cached")


if __name__ == "__main__":
    main()
