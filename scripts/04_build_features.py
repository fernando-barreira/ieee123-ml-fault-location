"""Stage 4 — derive the physics-informed feature matrix.

Turns the raw phasors written by stage 3 into the features the models consume:
superimposed phasors, per-unit sag and surge, superimposed apparent impedance,
symmetrical components and cross-PMU aggregates.  See :mod:`src.features` for
what each family measures and why it is there.

The matrix covers **every** candidate PMU and is built exactly once.  The
K-PMU scenario datasets are *not* produced here: they are extracted from this
matrix by stage 7, after the placement search, which copies the per-PMU columns
and recomputes the three subset-dependent aggregates.  Keeping a single
extraction path is deliberate — regenerating features per scenario from the raw
file would create a second code path whose PMU lists could silently drift from
the FSNR result.

Usage
-----
.. code-block:: bash

    python scripts/04_build_features.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config
from src.features import build_features
from src.pmu_candidates import load_candidates


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--raw", type=config.resolve_path, default=config.RAW_DATASET_CSV)
    parser.add_argument("--candidates", type=config.resolve_path, default=config.CANDIDATES_PKL)
    parser.add_argument("--out", type=config.resolve_path, default=config.FEATURES_ALL_CSV)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config.ensure_parent(args.out)

    if not args.raw.exists():
        raise SystemExit(
            f"Raw dataset not found: {args.raw}\n"
            "Run scripts/03_run_simulations.py first, or download the "
            "published dataset (see data/README.md)."
        )

    print(f"Reading {args.raw}")
    df = pd.read_csv(args.raw)
    print(f"  shape: {df.shape}")

    pmus = [str(b) for b in load_candidates(args.candidates)["candidates"]]

    features = build_features(
        df, pmus, eps=config.EPS, phases=config.PHASES, verbose=True
    )

    expected = config.expected_n_features(len(pmus))
    actual = features.shape[1] - 1  # the target column is not a feature
    if actual != expected:
        print(
            f"  WARNING: {actual} feature columns, expected {expected} for "
            f"{len(pmus)} PMU(s). Check that every measurement column is "
            "present in the raw dataset."
        )

    n_classes = features[config.TARGET_COL].nunique()
    print(f"  classes: {n_classes}")
    if n_classes != config.EXPECTED_N_CLASSES:
        print(
            f"  WARNING: expected {config.EXPECTED_N_CLASSES} classes. "
            "Some buses may never have been drawn as a fault location, or the "
            "fictitious-bus filter differs from the one in config.py."
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(args.out, index=False)
    print(f"\nSaved: {args.out}  {features.shape}")


if __name__ == "__main__":
    main()
