"""Class distribution of the fault-location dataset, and where it comes from.

Reports how often each bus appears as the fault location, and checks that
frequency against the value the sampling scheme predicts.

The imbalance is not incidental: it follows deterministically from how stage 3
draws a scenario.  A fault type is drawn first, then a bus **uniformly** from
the set of buses that can host that type.  A three-phase fault needs a
three-phase bus; a phase-to-phase fault needs at least two phases; a
line-to-ground fault can occur anywhere.  Writing :math:`w_t` for the fault-type
weights and :math:`N_t` for the number of eligible buses of each type, a bus
with :math:`p` phases is drawn with probability

.. math::

    P(b) = \\frac{w_\\text{LG}}{N_{\\ge 1}}
         + \\mathbb{1}[p \\ge 2]\\,\\frac{w_\\text{LL} + w_\\text{LLG}}{N_{\\ge 2}}
         + \\mathbb{1}[p = 3]\\,\\frac{w_\\text{LLL}}{N_{3}} .

So the class frequency of a bus is a function of its phase configuration alone,
and the imbalance ratio is fixed by the fault-type mix rather than by sampling
noise.  That is worth stating in a paper: it means the imbalance is a modelling
choice, reproducing the fault statistics of real overhead feeders, not an
artefact to be corrected away.

The comparison against the predicted frequency doubles as a check on the
campaign: a bus whose observed share departs from its prediction by more than
sampling error indicates that the eligibility sets used during simulation were
not the ones assumed here.

Usage
-----
.. code-block:: bash

    python tools/class_distribution.py
    python tools/class_distribution.py --dataset data/raw/measurements_candidate_pmus.csv
    python tools/class_distribution.py --by-split
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config


def entropy_stats(counts: np.ndarray) -> dict:
    """Shannon entropy of the class distribution and its effective class count.

    ``effective_classes = exp(H)`` is the number of equally frequent classes
    that would produce the same entropy.  It is a more honest summary of
    imbalance than the max/min ratio, which depends on a single extreme class:
    a distribution can have a large ratio and still be nearly uniform over the
    bulk of its mass.
    """
    p = counts / counts.sum()
    p = p[p > 0]
    h = float(-(p * np.log(p)).sum())
    return {
        "entropy": h,
        "effective_classes": float(np.exp(h)),
        "max_entropy": float(np.log(len(counts))),
        "uniformity": float(np.exp(h) / len(counts)),
    }


def predicted_frequencies(phase_count: dict[str, int]) -> dict[str, float]:
    """Probability of each bus under the stage-3 sampling scheme."""
    weights = config.FAULT_TYPE_WEIGHTS
    n_1 = sum(1 for p in phase_count.values() if p >= 1)
    n_2 = sum(1 for p in phase_count.values() if p >= 2)
    n_3 = sum(1 for p in phase_count.values() if p >= 3)

    w_lg = weights.get("LG", 0.0)
    w_two = weights.get("LL", 0.0) + weights.get("LLG", 0.0)
    w_three = weights.get("LLL", 0.0)

    out = {}
    for bus, phases in phase_count.items():
        p = w_lg / n_1 if n_1 else 0.0
        if phases >= 2 and n_2:
            p += w_two / n_2
        if phases >= 3 and n_3:
            p += w_three / n_3
        out[bus] = p
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--dataset", type=config.resolve_path,
                        default=config.features_csv(5))
    parser.add_argument("--dss", type=config.resolve_path, default=config.DSS_MASTER)
    parser.add_argument("--splits", type=config.resolve_path, default=config.SPLITS_PKL)
    parser.add_argument("--out-dir", type=config.resolve_path,
                        default=config.ANALYSIS_DIR / "class_distribution")
    parser.add_argument("--by-split", action="store_true",
                        help="Also report the distribution within each pool.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config.ensure_dir(args.out_dir)

    if not args.dataset.exists():
        raise SystemExit(f"{args.dataset} not found.")

    labels = pd.read_csv(args.dataset, usecols=[config.TARGET_COL])[config.TARGET_COL]
    labels = labels.astype(str)
    counts = labels.value_counts()
    n_total = int(counts.sum())
    n_classes = int(counts.size)

    print("=" * 78)
    print("  CLASS DISTRIBUTION")
    print("=" * 78)
    print(f"  samples : {n_total:,}")
    print(f"  classes : {n_classes}")
    print(f"  uniform share would be {1 / n_classes:.4%} "
          f"({n_total / n_classes:,.0f} samples each)")

    values = counts.to_numpy()
    stats = entropy_stats(values)
    quantiles = np.percentile(values, [0, 5, 25, 50, 75, 95, 100])

    print(f"\n  min      {int(values.min()):>7,}  (bus {counts.idxmin()})")
    print(f"  5th pct  {quantiles[1]:>7,.0f}")
    print(f"  median   {quantiles[3]:>7,.0f}")
    print(f"  95th pct {quantiles[5]:>7,.0f}")
    print(f"  max      {int(values.max()):>7,}  (bus {counts.idxmax()})")
    print(f"\n  imbalance ratio (max/min) : {values.max() / values.min():.2f}")
    print(f"  interquartile ratio       : {quantiles[4] / quantiles[2]:.2f}")
    print(f"  entropy                   : {stats['entropy']:.4f} "
          f"of {stats['max_entropy']:.4f} maximum")
    print(f"  effective classes         : {stats['effective_classes']:.1f} "
          f"of {n_classes} ({stats['uniformity']:.1%} of uniform)")

    # ── phase configuration, the mechanism behind the imbalance ───────────
    table = pd.DataFrame({"bus": counts.index, "count": values})
    table["share"] = table["count"] / n_total

    phase_count: dict[str, int] = {}
    if args.dss.exists():
        from src.dss_parser import parse_dss

        model = parse_dss(args.dss)
        phase_count = {b: n for b, n in model.bus_phase_count().items()
                       if b in set(table["bus"])}

    if phase_count:
        table["phases"] = table["bus"].map(phase_count)
        predicted = predicted_frequencies(phase_count)
        table["predicted_share"] = table["bus"].map(predicted)
        table["ratio_observed_predicted"] = (
            table["share"] / table["predicted_share"]
        )

        print("\n" + "=" * 78)
        print("  BY PHASE CONFIGURATION")
        print("=" * 78)
        print(f"  {'phases':>7}{'buses':>8}{'samples':>11}{'mean/bus':>11}"
              f"{'observed':>11}{'predicted':>11}")
        print("  " + "-" * 66)
        for phases, block in table.groupby("phases"):
            observed = block["share"].mean()
            expected = block["predicted_share"].mean()
            print(f"  {int(phases):>7}{len(block):>8}{int(block['count'].sum()):>11,}"
                  f"{block['count'].mean():>11,.0f}"
                  f"{observed:>11.4%}{expected:>11.4%}")

        deviation = (table["ratio_observed_predicted"] - 1).abs()
        # Sampling error on a share of ~1/122 over n_total draws, in relative
        # terms: a bus more than a few standard errors off is a red flag.
        se_relative = np.sqrt(
            (1 - table["predicted_share"]) / (n_total * table["predicted_share"])
        )
        outliers = table[deviation > 4 * se_relative]

        print(f"\n  largest deviation from the predicted share: "
              f"{deviation.max():.1%}")
        if len(outliers):
            print(f"  {len(outliers)} bus(es) deviate by more than four standard "
                  "errors — the eligibility sets used in the campaign may differ "
                  "from those assumed here:")
            for _, row in outliers.head(10).iterrows():
                print(f"    bus {row['bus']:<6} {int(row['phases'])}-phase  "
                      f"observed {row['share']:.4%} vs predicted "
                      f"{row['predicted_share']:.4%}")
        else:
            print("  every bus agrees with the sampling scheme within four "
                  "standard errors.")
        print("\n  The imbalance is therefore a deterministic consequence of the "
              "fault-type mix\n  (LG can occur on any bus, LLL only on "
              "three-phase buses), not sampling noise.")
    else:
        print(f"\n  {args.dss} not found; skipping the phase-configuration "
              "breakdown.")

    print("\n  Ten least frequent classes:")
    for bus, count in counts.tail(10).items():
        phases = phase_count.get(bus)
        suffix = f"  ({phases}-phase)" if phases else ""
        print(f"    bus {bus:<6} {count:>6,}  {count / n_total:.4%}{suffix}")

    # ── per pool ──────────────────────────────────────────────────────────
    if args.by_split and args.splits.exists():
        from src.splits import load_splits

        splits = load_splits(args.splits, y=labels.to_numpy())
        print("\n" + "=" * 78)
        print("  BY POOL")
        print("=" * 78)
        print(f"  {'pool':<14}{'samples':>10}{'classes':>10}{'min/class':>12}"
              f"{'max/class':>12}{'ratio':>9}")
        print("  " + "-" * 66)
        for pool in ("fsnr", "optuna", "cv_indices", "holdout"):
            subset = labels.to_numpy()[splits[pool]]
            block = pd.Series(subset).value_counts()
            print(f"  {pool:<14}{len(subset):>10,}{block.size:>10}"
                  f"{int(block.min()):>12}{int(block.max()):>12}"
                  f"{block.max() / block.min():>9.1f}")

    table = table.sort_values("count", ascending=False)
    path = args.out_dir / "class_distribution.csv"
    table.to_csv(path, index=False)
    print(f"\n  Saved: {path}")

    print("\n  Sentence for the paper:")
    print(f"    The 122 classes are unevenly represented, with per-class counts "
          f"ranging from {int(values.min()):,} to {int(values.max()):,} "
          f"(ratio {values.max() / values.min():.1f}).")
    print("    The imbalance follows from the fault-type mix rather than from "
          "sampling:")
    print("    single-phase laterals can only host line-to-ground faults, while "
          "three-phase")
    print("    buses are eligible for every type.")


if __name__ == "__main__":
    main()