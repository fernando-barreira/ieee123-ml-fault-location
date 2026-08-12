"""Stage 12 — accuracy conditioned on fault type and on fault resistance.

Aggregate accuracy hides the two variables that decide how hard a fault is to
locate, and both are physically meaningful.

**Fault type.**  A three-phase fault drives all three conductors and produces a
large, balanced disturbance at every PMU.  A single line-to-ground fault
disturbs one phase through a ground return path whose impedance is variable and
poorly known, and it is also the only type that can occur on a single-phase
lateral — the part of the feeder the sensors observe worst.  The expected
ordering is therefore LLL best, LG and LLG worst.

**Fault resistance.**  The superimposed current at a PMU scales roughly as

.. math:: |\\tilde{I}_F| \\\simeq \\\frac{\\kappa_\\tau V^{(n)}_\\text{pre}}{|\\tilde{Z}_\\text{eq} + mR_f|}

so once :math:`R_f` dominates the path impedance the location-dependent part of
the signature is compressed towards zero and every candidate bus starts to look
alike.  Accuracy within LG faults should fall monotonically with :math:`R_f`,
and how fast it falls is the honest statement of where a phasor-based locator
stops working.

Both conditioning variables are metadata of the simulation, not measurements,
so they are read from the raw dataset.  The row alignment between the raw file
and the feature file is asserted bus by bus rather than assumed: the feature
builder concatenates columns without reordering rows, but that is a property
worth checking rather than trusting, since a silent misalignment would produce
a plausible-looking table of nonsense.

Usage
-----
.. code-block:: bash

    python scripts/12_analyze_per_fault_type.py
    python scripts/12_analyze_per_fault_type.py --models "Deep Residual MLP,Random Forest"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config
from src.analysis import (
    Report,
    load_fold_indices,
    load_fold_results,
    load_preprocessing,
    mean_std,
)


def top1(preds: np.ndarray, truth: np.ndarray, mask: np.ndarray) -> float:
    return float((preds[mask] == truth[mask]).mean()) if mask.any() else np.nan


def top3(probs: np.ndarray, truth: np.ndarray, mask: np.ndarray) -> float:
    if not mask.any():
        return np.nan
    selected, target = probs[mask], truth[mask]
    top = np.argpartition(-selected, kth=2, axis=1)[:, :3]
    return float(np.any(top == target[:, None], axis=1).mean())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--raw", type=config.resolve_path,
                        default=config.RAW_DATASET_CSV)
    parser.add_argument("--dataset", type=config.resolve_path,
                        default=config.features_csv(5))
    parser.add_argument("--cv-dir", type=config.resolve_path, default=config.CV_DIR)
    parser.add_argument("--out-dir", type=config.resolve_path,
                        default=config.ANALYSIS_DIR / "per_fault_type")
    parser.add_argument(
        "--models", type=str, default="Deep Residual MLP,Random Forest",
        help="Comma-separated display names to tabulate.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config.ensure_dir(args.out_dir)
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    report = Report()

    report("=" * 78)
    report("  ACCURACY BY FAULT TYPE AND BY FAULT RESISTANCE")
    report("=" * 78)

    pre = load_preprocessing(args.cv_dir)
    encoder = pre["target_encoder"]
    fold_indices = load_fold_indices(args.cv_dir)
    n_folds = pre["n_folds"]

    if not args.raw.exists():
        raise SystemExit(
            f"{args.raw} not found. The fault type and resistance are metadata "
            "of the simulation and live only in the raw dataset."
        )

    # Only the metadata columns are read; none of the candidate PMU
    # measurements are touched, which keeps this cheap on a multi-GB file.
    meta = pd.read_csv(
        args.raw,
        usecols=["tipo_falta", "impedancia_falta", config.TARGET_COL],
    )
    labels = pd.read_csv(args.dataset, usecols=[config.TARGET_COL])[config.TARGET_COL]

    if len(meta) != len(labels):
        raise SystemExit(
            f"Raw dataset has {len(meta)} rows, feature dataset has "
            f"{len(labels)} — they cannot be aligned by position."
        )
    mismatch = (meta[config.TARGET_COL].astype(str).to_numpy()
                != labels.astype(str).to_numpy())
    if mismatch.any():
        raise SystemExit(
            f"The fault bus differs on {int(mismatch.sum())} row(s) between the "
            "raw and feature datasets, so their row order is not the same. "
            "Conditioning on the raw metadata would mislabel every sample."
        )

    fault_types = list(config.FAULT_TYPE_WEIGHTS)
    all_types = meta["tipo_falta"].astype(str).to_numpy()
    all_rf = meta["impedancia_falta"].astype(float).to_numpy()
    y_all = encoder.transform(labels.to_numpy())

    observed = sorted(set(all_types))
    unknown = [t for t in observed if t not in fault_types]
    if unknown:
        report(
            f"  NOTE: fault types {unknown} are present in the data but not in "
            f"config.FAULT_TYPE_WEIGHTS ({fault_types}). If this dataset was "
            "generated before the LG/LLG/LL/LLL rename, its labels use the old "
            "spelling and the table below will be empty."
        )

    # Display order: most severe first, then by decreasing frequency.
    order = [t for t in ("LLL", "LL", "LLG", "LG") if t in observed]
    order += [t for t in observed if t not in order]

    by_type = {m: {t: {"top1": [], "top3": []} for t in order} for m in models}
    by_rf = {m: {b: [] for b in config.RF_BIN_LABELS} for m in models}
    overall = {m: [] for m in models}

    for fold in range(n_folds):
        results = load_fold_results(args.cv_dir, fold)
        if not results:
            report(f"  fold {fold}: no results, skipped")
            continue

        test_idx = fold_indices[fold]["test"]
        types_fold, rf_fold = all_types[test_idx], all_rf[test_idx]

        for name in models:
            if name not in results:
                report(f"  fold {fold}: '{name}' absent from fold_results, skipped")
                continue
            entry = results[name]
            preds = np.asarray(entry["preds"])
            probs = np.asarray(entry["probs"])
            truth = np.asarray(entry["y_true"])

            if len(truth) != len(test_idx) or not np.array_equal(truth, y_all[test_idx]):
                raise SystemExit(
                    f"fold {fold} / {name}: the stored labels do not match the "
                    "encoded target at the fold's test indices. The results and "
                    "the splits belong to different runs."
                )

            overall[name].append(float((preds == truth).mean()))
            for fault_type in order:
                mask = types_fold == fault_type
                by_type[name][fault_type]["top1"].append(top1(preds, truth, mask))
                by_type[name][fault_type]["top3"].append(top3(probs, truth, mask))

            # Resistance is swept within LG only: it is the dominant type and
            # the one whose ground return path makes Rf physically variable.
            ground_fault = types_fold == "LG"
            for low, high, label in zip(config.RF_BINS[:-1], config.RF_BINS[1:],
                                        config.RF_BIN_LABELS):
                mask = ground_fault & (rf_fold >= low) & (rf_fold < high)
                by_rf[name][label].append(top1(preds, truth, mask))

        counts = " | ".join(
            f"{t}={int((types_fold == t).sum())}" for t in order
        )
        report(f"  fold {fold}: n_test={len(test_idx)} | {counts}")

    # ── report ────────────────────────────────────────────────────────────
    report("\n" + "=" * 78)
    report("  [1] TOP-1 / TOP-3 BY FAULT TYPE (mean +/- std over folds)")
    report("=" * 78)
    for name in models:
        mean, std = mean_std(overall[name])
        report(f"  {name:<22} overall Top-1 = {mean:.4f} +/- {std:.4f}")

    report("")
    report(f"  {'Type':<6}" + "".join(f"{m:>34}" for m in models))
    report(f"  {'':<6}" + "".join(f"{'Top-1':>17}{'Top-3':>17}" for _ in models))
    for fault_type in order:
        row = f"  {fault_type:<6}"
        for name in models:
            t1, s1 = mean_std(by_type[name][fault_type]["top1"])
            t3, s3 = mean_std(by_type[name][fault_type]["top3"])
            row += f"{t1:>10.4f}+/-{s1:<4.4f}{t3:>10.4f}+/-{s3:<4.4f}"
        report(row)

    report("\n  [2] TOP-1 WITHIN LG FAULTS, BY FAULT RESISTANCE (ohm)")
    report(f"  {'Bin':<10}" + "".join(f"{m:>28}" for m in models))
    for label in config.RF_BIN_LABELS:
        row = f"  {label:<10}"
        for name in models:
            mean, std = mean_std(by_rf[name][label])
            row += f"{mean:>18.4f} +/-{std:<6.4f}"
        report(row)

    report("\n  [3] LaTeX rows for the per-type table")
    for fault_type in order:
        cells = []
        for name in models:
            t1, _ = mean_std(by_type[name][fault_type]["top1"])
            t3, _ = mean_std(by_type[name][fault_type]["top3"])
            cells += [f"{t1:.3f}", f"{t3:.3f}"]
        report(f"    {fault_type:<4}& " + " & ".join(cells) + r" \\")

    if models:
        first = models[0]
        low, _ = mean_std(by_rf[first][config.RF_BIN_LABELS[0]])
        high, _ = mean_std(by_rf[first][config.RF_BIN_LABELS[-1]])
        report(
            f"\n  Resistance sentence ({first}): Top-1 within LG falls from "
            f"{low:.3f} at $R_f<5\\,\\Omega$ to {high:.3f} at "
            f"$R_f\\geq 50\\,\\Omega$."
        )

    rows = []
    for name in models:
        for fault_type in order:
            t1, s1 = mean_std(by_type[name][fault_type]["top1"])
            t3, s3 = mean_std(by_type[name][fault_type]["top3"])
            rows.append({"model": name, "fault_type": fault_type,
                         "top1_mean": t1, "top1_std": s1,
                         "top3_mean": t3, "top3_std": s3})
    pd.DataFrame(rows).to_csv(args.out_dir / "per_fault_type.csv", index=False)

    rows = [
        {"model": name, "rf_bin": label,
         "top1_mean": mean_std(by_rf[name][label])[0],
         "top1_std": mean_std(by_rf[name][label])[1]}
        for name in models for label in config.RF_BIN_LABELS
    ]
    pd.DataFrame(rows).to_csv(args.out_dir / "per_fault_resistance.csv", index=False)
    report.save(args.out_dir / "per_fault_type.txt")


if __name__ == "__main__":
    main()
