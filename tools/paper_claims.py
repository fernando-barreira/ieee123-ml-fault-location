"""Regenerate the numbers quoted in the paper's prose.

Statistics that appear as sentences rather than as table entries are the ones
most likely to go stale: nothing recomputes them, and nothing fails when they
drift. This stage derives them from the stored artefacts so the text can be
checked against the current run.

**Concentration of the impedance baseline.** The claim is that the classical
method exercises only a fraction of the 122 candidate buses and piles most of
its predictions onto a handful. That is the quantitative form of the paper's
central argument: with distributed generation injecting fault current from
inside the feeder, the apparent impedance no longer maps monotonically onto
distance, so the criterion collapses onto whichever buses happen to minimise
the residual regardless of the actual fault. Reported here as the number of
distinct predicted buses and the share taken by the most-predicted one or two.

**Per-bus recall for the best model.** The claim is that most buses are located
reliably while a small tail fails. Recall is Top-1 conditioned on the true
class, pooled across folds so every test sample counts exactly once. The tail
is where the physics shows: buses joined to another by a closed switch cannot
be separated by any measurement-based method, and buses on laterals far from
every PMU are weakly observed.

Both are read from the saved predictions; no model is reloaded, and nothing
downstream depends on this tool's output — the per-bus figure computes the same
statistic from the same shared function, so the figure and the prose agree by
construction rather than by one reading the other's file.

Usage
-----
.. code-block:: bash

    python tools/paper_claims.py
    python tools/paper_claims.py --model "Deep Residual MLP"
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config
from src.analysis import (
    Report,
    iter_folds,
    load_preprocessing,
    per_bus_recall,
    switch_partner,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--cv-dir", type=config.resolve_path, default=config.CV_DIR)
    parser.add_argument("--baseline-dir", type=config.resolve_path,
                        default=config.BASELINE_DIR)
    parser.add_argument("--out-dir", type=config.resolve_path,
                        default=config.ANALYSIS_DIR / "paper_claims")
    parser.add_argument("--model", type=str, default="Deep Residual MLP",
                        help="Model whose per-bus recall is reported.")
    parser.add_argument("--baseline", type=str, default="Baseline Impedance")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config.ensure_dir(args.out_dir)
    report = Report()

    pre = load_preprocessing(args.cv_dir)
    classes = [str(c) for c in pre["classes_buses"]]
    n_classes = len(classes)
    n_folds = pre["n_folds"]

    report("=" * 78)
    report("  NUMBERS QUOTED IN THE PAPER'S PROSE")
    report("=" * 78)

    baseline_preds: list[np.ndarray] = []
    model_preds: list[np.ndarray] = []
    model_truth: list[np.ndarray] = []

    for fold, results in iter_folds(args.cv_dir, n_folds, args.baseline_dir):
        if args.baseline in results:
            baseline_preds.append(np.asarray(results[args.baseline]["preds"]))
        if args.model in results:
            model_preds.append(np.asarray(results[args.model]["preds"]))
            model_truth.append(np.asarray(results[args.model]["y_true"]))

    # ── [1] concentration of the impedance baseline ───────────────────────
    report("\n" + "=" * 78)
    report(f"  [1] PREDICTION CONCENTRATION — {args.baseline}")
    report("=" * 78)

    if not baseline_preds:
        report(f"  '{args.baseline}' not found. Run stage 10 first.")
    else:
        pooled = np.concatenate(baseline_preds)
        counts = Counter(pooled.tolist())
        total = len(pooled)
        ordered = counts.most_common()

        distinct = len(counts)
        share_top1 = ordered[0][1] / total
        share_top2 = sum(c for _, c in ordered[:2]) / total
        share_top5 = sum(c for _, c in ordered[:5]) / total

        report(f"  predictions made          : {total:,}")
        report(f"  distinct buses predicted  : {distinct} of {n_classes} "
               f"({distinct / n_classes:.1%})")
        report(f"  share on the top bus      : {share_top1:.1%} "
               f"(bus {classes[ordered[0][0]]})")
        report(f"  share on the top two      : {share_top2:.1%}")
        report(f"  share on the top five     : {share_top5:.1%}")
        report("\n  ten most-predicted buses:")
        for index, count in ordered[:10]:
            report(f"    bus {classes[index]:<6} {count:>8,}  ({count / total:>6.1%})")

        report(
            f"\n  Sentence: the impedance baseline predicts only {distinct} of "
            f"the {n_classes} candidate buses, placing {share_top1:.0%} of its "
            f"predictions on a single bus and {share_top2:.0%} on two."
        )
        pd.DataFrame(
            [{"bus": classes[i], "n_predictions": c, "share": c / total}
             for i, c in ordered]
        ).to_csv(args.out_dir / "impedance_concentration.csv", index=False)

    # ── [2] per-bus recall of the best model ──────────────────────────────
    report("\n" + "=" * 78)
    report(f"  [2] PER-BUS RECALL — {args.model}")
    report("=" * 78)

    if not model_preds:
        report(f"  '{args.model}' not found. Run stage 9 first.")
        report.save(args.out_dir / "paper_claims.txt")
        return

    preds = np.concatenate(model_preds)
    truth = np.concatenate(model_truth)
    overall = float((preds == truth).mean())

    table = per_bus_recall(args.cv_dir, args.model, classes, n_folds)
    recall = table["recall"].to_numpy()
    support = table["support"].to_numpy()
    evaluated = support > 0
    threshold = config.PER_BUS_HEALTHY_THRESHOLD
    n_healthy = int((recall[evaluated] >= threshold).sum())
    n_poor = int((recall[evaluated] < 0.70).sum())
    n_evaluated = int(evaluated.sum())

    report(f"  pooled samples        : {len(truth):,}")
    report(f"  overall Top-1         : {overall:.4f}")
    report(f"  buses with samples    : {n_evaluated} of {n_classes}")
    report(f"  recall >= {threshold:.0%}         : {n_healthy} "
           f"({n_healthy / n_evaluated:.1%})")
    report(f"  recall < 70%          : {n_poor} ({n_poor / n_evaluated:.1%})")
    report(f"  median recall         : {np.nanmedian(recall[evaluated]):.4f}")

    report("\n  fifteen worst buses:")
    report(f"    {'bus':<8}{'recall':>9}{'support':>10}  note")
    order = np.argsort(np.where(evaluated, recall, np.inf))
    for index in order[:15]:
        if not evaluated[index]:
            continue
        bus = classes[index]
        partner = switch_partner(bus)
        note = f"switch-adjacent to bus {partner}" if partner else ""
        report(f"    {bus:<8}{recall[index]:>9.3f}{support[index]:>10}  {note}")

    in_pairs = [b for pair in config.SWITCH_PAIRS for b in pair]
    pair_indices = [classes.index(b) for b in in_pairs if b in classes]
    if pair_indices:
        pair_recall = float(np.nanmean(recall[pair_indices]))
        others = [i for i in range(n_classes) if evaluated[i] and i not in pair_indices]
        report(f"\n  mean recall on switch-adjacent buses : {pair_recall:.3f} "
               f"({len(pair_indices)} buses)")
        report(f"  mean recall elsewhere                : "
               f"{float(np.nanmean(recall[others])):.3f} ({len(others)} buses)")
        report("  These pairs are separated by a closed switch of a few "
               "milliohms; no measurement-based method can distinguish them.")

    report(
        f"\n  Sentence: {n_healthy / n_evaluated:.0%} of the buses are located "
        f"correctly in at least {threshold:.0%} of cases, while a tail of "
        f"{n_poor} buses falls below 70%."
    )

    table.to_csv(args.out_dir / "per_bus_recall.csv", index=False)
    report.save(args.out_dir / "paper_claims.txt")


if __name__ == "__main__":
    main()
