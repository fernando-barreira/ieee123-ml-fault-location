"""Frozen data splits shared by every phase of the pipeline.

The study runs four consecutive experiments on the same 100 000-sample dataset:
PMU placement search, hyper-parameter search, cross-validated training, and a
final sanity check.  If each one drew its own split, information would flow
from one to the next — a placement chosen on samples that later end up in a
cross-validation test fold inflates the reported accuracy.

This module creates **four disjoint pools once**, stores their indices, and
never touches them again:

===========  =========  ====================================================
Pool         Size       Purpose
===========  =========  ====================================================
``holdout``    5 000    Untouched until the final check.  Its only job is to
                        confirm that repeated use of the cross-validation pool
                        during model development did not overfit it.
``fsnr``       5 000    PMU placement search (70/30 train/validation, frozen).
``optuna``    15 000    Hyper-parameter search (70/30 train/validation, frozen).
``cv``       ~75 000    Stratified 5-fold cross-validation: the reported result.
===========  =========  ====================================================

Every split is stratified on the fault bus, so all 122 classes are represented
in every pool and every fold, including the rare ones.

Binding to the dataset
----------------------
The splits are **positional indices**.  They are only valid for the exact
dataset they were generated from, so :func:`create_splits` records the row
count and a hash of the target column, and :func:`load_splits` refuses to load
them against a dataset that does not match.  Regenerating the raw dataset
invalidates the splits and therefore every result computed with them.
"""

from __future__ import annotations

import hashlib
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split


def target_fingerprint(y: np.ndarray) -> str:
    """Stable hash of the label column, used to bind splits to a dataset."""
    payload = "\n".join(str(v) for v in y).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def describe_classes(y: np.ndarray, expected_n_classes: int | None = None) -> dict:
    """Class-count statistics, with warnings for the failure modes that matter.

    A stratified 5-fold split over four pools needs roughly 20 samples of the
    rarest class to keep every fold populated.  Below that, some folds lose the
    class entirely and per-class metrics become undefined rather than merely
    noisy.
    """
    counts = pd.Series(y).value_counts()
    stats = {
        "n_classes": int(counts.size),
        "min_count": int(counts.min()),
        "min_class": counts.idxmin(),
        "max_count": int(counts.max()),
        "max_class": counts.idxmax(),
        "imbalance_ratio": float(counts.max() / counts.min()),
        "warnings": [],
    }
    if expected_n_classes is not None and stats["n_classes"] != expected_n_classes:
        stats["warnings"].append(
            f"{stats['n_classes']} classes found, {expected_n_classes} "
            "expected; check the fictitious-bus filter used by the simulator."
        )
    return stats


def create_splits(
    y: np.ndarray,
    sizes: dict[str, int],
    fsnr_val_fraction: float,
    optuna_val_fraction: float,
    cv_n_splits: int,
    seed: int,
) -> dict:
    """Build the four pools and their frozen internal splits.

    The pools are carved out in a fixed order — holdout first, then the
    placement pool, then the hyper-parameter pool, with cross-validation taking
    the remainder — so that adding samples to the dataset later grows only the
    cross-validation pool and leaves the other three bit-identical.

    Parameters
    ----------
    y
        Label array, one entry per dataset row.
    sizes
        ``{"holdout": n, "fsnr": n, "optuna": n}``.
    fsnr_val_fraction, optuna_val_fraction
        Validation share inside those two pools.
    cv_n_splits
        Number of stratified folds.
    seed
        Random seed.

    Returns
    -------
    dict
        Pool indices, frozen internal splits, cross-validation folds
        (**relative to** ``cv_indices``), and metadata.
    """
    n_total = len(y)
    indices = np.arange(n_total)

    rest, holdout = train_test_split(
        indices, test_size=sizes["holdout"], stratify=y, random_state=seed
    )
    rest, fsnr = train_test_split(
        rest, test_size=sizes["fsnr"], stratify=y[rest], random_state=seed
    )
    cv_indices, optuna = train_test_split(
        rest, test_size=sizes["optuna"], stratify=y[rest], random_state=seed
    )

    fsnr_tr, fsnr_va = train_test_split(
        fsnr, test_size=fsnr_val_fraction, stratify=y[fsnr], random_state=seed
    )
    optuna_tr, optuna_va = train_test_split(
        optuna, test_size=optuna_val_fraction, stratify=y[optuna], random_state=seed
    )

    skf = StratifiedKFold(n_splits=cv_n_splits, shuffle=True, random_state=seed)
    cv_folds = [(tr, te) for tr, te in skf.split(cv_indices, y[cv_indices])]

    return {
        "holdout": holdout,
        "fsnr": fsnr,
        "optuna": optuna,
        "cv_indices": cv_indices,
        "fsnr_tr": fsnr_tr,
        "fsnr_va": fsnr_va,
        "optuna_tr": optuna_tr,
        "optuna_va": optuna_va,
        # Fold indices are LOCAL to cv_indices: use cv_indices[tr] to index the
        # dataset.  Storing them locally keeps them valid regardless of how the
        # cross-validation pool is later materialised.
        "cv_folds": cv_folds,
        "seed": seed,
        "target_fingerprint": target_fingerprint(y),
        "sizes": {
            "total": int(n_total),
            "holdout": int(len(holdout)),
            "fsnr": int(len(fsnr)),
            "fsnr_tr": int(len(fsnr_tr)),
            "fsnr_va": int(len(fsnr_va)),
            "optuna": int(len(optuna)),
            "optuna_tr": int(len(optuna_tr)),
            "optuna_va": int(len(optuna_va)),
            "cv": int(len(cv_indices)),
            "cv_n_splits": cv_n_splits,
        },
    }


def validate_splits(splits: dict) -> None:
    """Assert the invariants the whole study depends on.

    * The four pools are pairwise disjoint and together cover the dataset.
    * Each frozen internal split partitions its pool exactly.
    * The cross-validation test folds are disjoint and cover the pool.

    Any violation means results from different phases were computed on
    overlapping data and are not comparable, so these are hard errors.
    """
    pools = {
        "holdout": splits["holdout"],
        "fsnr": splits["fsnr"],
        "optuna": splits["optuna"],
        "cv": splits["cv_indices"],
    }

    names = list(pools)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            overlap = np.intersect1d(pools[a], pools[b])
            if overlap.size:
                raise AssertionError(
                    f"Pools {a} and {b} share {overlap.size} sample(s)."
                )

    union = sum(len(v) for v in pools.values())
    if union != splits["sizes"]["total"]:
        raise AssertionError(
            f"Pools cover {union} samples, dataset has "
            f"{splits['sizes']['total']}."
        )

    for pool in ("fsnr", "optuna"):
        tr, va, full = (
            set(splits[f"{pool}_tr"]),
            set(splits[f"{pool}_va"]),
            set(splits[pool]),
        )
        if tr | va != full or tr & va:
            raise AssertionError(f"Internal split of pool {pool!r} is invalid.")

    seen = np.zeros(len(splits["cv_indices"]), dtype=bool)
    for k, (tr, te) in enumerate(splits["cv_folds"]):
        if np.intersect1d(tr, te).size:
            raise AssertionError(f"Fold {k}: train and test overlap.")
        if seen[te].any():
            raise AssertionError(f"Fold {k}: test set overlaps another fold.")
        seen[te] = True
    if not seen.all():
        raise AssertionError("Cross-validation folds do not cover the pool.")


def summarise(splits: dict, y: np.ndarray) -> pd.DataFrame:
    """Per-pool class coverage, as a table suitable for the paper's appendix."""
    n_classes_total = len(np.unique(y))
    rows = []

    for name in (
        "fsnr", "fsnr_tr", "fsnr_va",
        "optuna", "optuna_tr", "optuna_va",
        "cv_indices", "holdout",
    ):
        counts = pd.Series(y[splits[name]]).value_counts()
        rows.append(
            {
                "pool": name,
                "n_samples": len(splits[name]),
                "n_classes": int(counts.size),
                "class_coverage_pct": round(100 * counts.size / n_classes_total, 1),
                "min_per_class": int(counts.min()),
                "max_per_class": int(counts.max()),
            }
        )

    for k, (_, te) in enumerate(splits["cv_folds"]):
        counts = pd.Series(y[splits["cv_indices"][te]]).value_counts()
        rows.append(
            {
                "pool": f"cv_fold{k}_test",
                "n_samples": int(te.size),
                "n_classes": int(counts.size),
                "class_coverage_pct": round(100 * counts.size / n_classes_total, 1),
                "min_per_class": int(counts.min()),
                "max_per_class": int(counts.max()),
            }
        )

    return pd.DataFrame(rows)


def save_splits(splits: dict, path: str | Path) -> None:
    """Persist the splits.  Overwriting this file invalidates every result."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fh:
        pickle.dump(splits, fh, protocol=pickle.HIGHEST_PROTOCOL)


def load_splits(path: str | Path, y: np.ndarray | None = None) -> dict:
    """Load the splits and, if ``y`` is given, verify they match the dataset.

    Raises
    ------
    ValueError
        If the row count or the label fingerprint differs.  This catches the
        most damaging silent failure in the pipeline: re-running the simulation
        campaign and then reusing stale indices, which would scramble the
        sample-to-label correspondence without raising anything.
    """
    with open(path, "rb") as fh:
        splits = pickle.load(fh)

    if y is None:
        return splits

    if len(y) != splits["sizes"]["total"]:
        raise ValueError(
            f"Dataset has {len(y)} rows, splits were built for "
            f"{splits['sizes']['total']}. Regenerate the splits."
        )

    stored = splits.get("target_fingerprint")
    if stored is not None and stored != target_fingerprint(y):
        raise ValueError(
            "Label fingerprint mismatch: these splits belong to a different "
            "dataset. Regenerate them, and regenerate everything downstream."
        )
    return splits
