"""Stage 5 — create the frozen data splits.

Partitions the dataset into four disjoint pools (placement search,
hyper-parameter search, cross-validation, holdout) and freezes their indices.

Run this **once**.  Overwriting :data:`config.SPLITS_PKL` invalidates every
result produced with the previous version, because the pools would no longer be
the ones the models were selected on.  The script refuses to overwrite an
existing file unless ``--force`` is given.

Usage
-----
.. code-block:: bash

    python scripts/05_make_splits.py
    python scripts/05_make_splits.py --dataset data/processed/features_candidate_pmus.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config
from src.splits import (
    create_splits,
    describe_classes,
    save_splits,
    summarise,
    validate_splits,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--dataset", type=config.resolve_path, default=config.FEATURES_ALL_CSV)
    parser.add_argument("--out", type=config.resolve_path, default=config.SPLITS_PKL)
    parser.add_argument("--summary", type=config.resolve_path, default=config.SPLITS_SUMMARY_CSV)
    parser.add_argument("--seed", type=int, default=config.SEED)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing splits. Invalidates all previous results.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config.ensure_parent(args.out)
    config.ensure_parent(args.summary)

    if args.out.exists() and not args.force:
        raise SystemExit(
            f"{args.out} already exists.\n"
            "The splits are frozen on purpose: every result in the study was "
            "produced with them. Pass --force only if you intend to rerun the "
            "entire pipeline."
        )

    print(f"Reading {args.dataset}")
    df = pd.read_csv(args.dataset, usecols=[config.TARGET_COL])
    y = df[config.TARGET_COL].to_numpy()
    print(f"  samples: {len(y):,}")

    stats = describe_classes(y, config.EXPECTED_N_CLASSES)
    print(f"\nClasses: {stats['n_classes']}")
    print(f"  rarest : {stats['min_count']:>6,} samples (bus {stats['min_class']})")
    print(f"  most   : {stats['max_count']:>6,} samples (bus {stats['max_class']})")
    print(f"  imbalance ratio: {stats['imbalance_ratio']:.1f}x")
    for warning in stats["warnings"]:
        print(f"  WARNING: {warning}")

    fixed = sum(config.SPLIT_SIZES.values())
    if len(y) <= fixed:
        raise SystemExit(
            f"Dataset has {len(y):,} samples; the three fixed pools alone need "
            f"more than {fixed:,}."
        )

    min_per_class_needed = config.CV_N_SPLITS * 4
    if stats["min_count"] < min_per_class_needed:
        print(
            f"  WARNING: the rarest class has {stats['min_count']} samples. "
            f"Roughly {min_per_class_needed} are needed for it to survive four "
            f"pools and {config.CV_N_SPLITS} stratified folds; some folds may "
            "not contain it at all."
        )

    splits = create_splits(
        y,
        sizes=config.SPLIT_SIZES,
        fsnr_val_fraction=config.FSNR_VAL_FRACTION,
        optuna_val_fraction=config.OPTUNA_VAL_FRACTION,
        cv_n_splits=config.CV_N_SPLITS,
        seed=args.seed,
    )
    validate_splits(splits)
    print("\nAll split invariants hold "
          "(pools disjoint, pools complete, folds disjoint and complete).")

    summary = summarise(splits, y)
    print()
    print(summary.to_string(index=False))

    save_splits(splits, args.out)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.summary, index=False)
    print(f"\nSaved: {args.out}")
    print(f"Saved: {args.summary}")
    print(f"\nLabel fingerprint: {splits['target_fingerprint']}")
    print("Downstream stages verify this fingerprint before using the splits.")


if __name__ == "__main__":
    main()
