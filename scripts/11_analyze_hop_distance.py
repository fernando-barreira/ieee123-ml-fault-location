"""Stage 11 — hop-distance error, the topological view of the residual errors.

Top-1 accuracy treats every mistake as equally bad.  For a fault locator it is
not: predicting a bus one span away sends the crew to the right stretch of
feeder, while predicting a bus on another lateral sends them somewhere else
entirely.  This stage measures that distinction.

For each prediction, the error is the shortest-path hop count between the
predicted and the true bus on the feeder graph,

.. math:: \\delta(\\hat b, b^\\star) = d_G(\\hat b, b^\\star)

and the reported quantity is its empirical CDF,
:math:`F_\\delta(h) = \\Pr[\\delta \\le h]`.  By construction
:math:`F_\\delta(0)` is Top-1 accuracy, so the curve extends the headline
metric rather than replacing it — and :math:`F_\\delta(1)` is the operationally
interesting number, "right bus or a neighbour".

The graph is unweighted on purpose.  A hop is a dispatch decision, not a
distance: the crew drives to a bus and inspects the spans around it, and that
cost is roughly constant per hop regardless of conductor length.  The weighted
alternative would answer a different question.

Predictions are read from the stored fold results; no model is reloaded and no
inference is repeated, so the numbers here are exactly the ones behind the
accuracy tables.

Usage
-----
.. code-block:: bash

    python scripts/11_analyze_hop_distance.py
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config
from src.analysis import (
    Report,
    iter_folds,
    load_preprocessing,
    ordered_models,
)


def cdf(deltas: np.ndarray, levels) -> dict[int, float]:
    """``{h: Pr[delta <= h]}``."""
    return {h: float((deltas <= h).mean()) for h in levels}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--cv-dir", type=config.resolve_path, default=config.CV_DIR)
    parser.add_argument("--baseline-dir", type=config.resolve_path,
                        default=config.BASELINE_DIR)
    parser.add_argument("--out-dir", type=config.resolve_path,
                        default=config.ANALYSIS_DIR / "hop_distance")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config.ensure_dir(args.out_dir)
    report = Report()

    pre = load_preprocessing(args.cv_dir)
    hops = pre["hop_matrix"]
    classes = pre["classes_buses"]
    n_folds = pre["n_folds"]

    report("=" * 78)
    report("  HOP-DISTANCE ERROR")
    report("=" * 78)
    report(f"  {len(classes)} candidate buses | hop matrix {hops.shape}")
    off_diagonal = hops[~np.eye(len(classes), dtype=bool)]
    report(f"  mean distance between distinct buses: {off_diagonal.mean():.2f} hops")
    report(f"  feeder diameter: {int(hops.max())} hops")

    # The hop matrix comes from stage 9, which built it from the single
    # canonical topology graph and raised on any bus it could not reach. No
    # unreachable-pair sentinel is needed or accepted here: a sentinel would
    # inflate the mean error silently, which is precisely the failure this
    # metric is meant to expose.
    if (hops < 0).any():
        raise SystemExit(
            "Hop matrix contains negative entries — it was not produced by the "
            "current stage 9. Re-run it."
        )

    deltas: dict[str, list[np.ndarray]] = {}
    for fold, results in iter_folds(args.cv_dir, n_folds, args.baseline_dir):
        for name, entry in results.items():
            preds = np.asarray(entry["preds"])
            truth = np.asarray(entry["y_true"])
            deltas.setdefault(name, []).append(hops[preds, truth])

    if not deltas:
        raise SystemExit("No fold results found. Run stage 9 (and 10) first.")

    models = ordered_models(deltas)
    report(f"\n  Models: {models}")

    summary: dict[str, dict] = {}
    for name in models:
        per_fold = deltas[name]
        means = [float(d.mean()) for d in per_fold]
        curves = [cdf(d, config.HOP_CDF_LEVELS) for d in per_fold]
        summary[name] = {
            "mean_delta": (float(np.mean(means)), float(np.std(means)), means),
            "cdf_mean": {h: float(np.mean([c[h] for c in curves]))
                         for h in config.HOP_CDF_LEVELS},
            "cdf_std": {h: float(np.std([c[h] for c in curves]))
                        for h in config.HOP_CDF_LEVELS},
            "n_folds": len(per_fold),
        }

    report("\n" + "=" * 102)
    report(f"  mean +/- std over {n_folds} folds")
    report("=" * 102)
    header = f"  {'Model':<22}{'mean hops':>12}"
    for h in config.HOP_CDF_LEVELS:
        header += f"{'F(d<=' + str(h) + ')':>16}"
    report(header)
    report("  " + "-" * 100)
    for name in models:
        entry = summary[name]
        mean, std, _ = entry["mean_delta"]
        row = f"  {name:<22}{mean:>7.2f}+/-{std:<4.2f}"
        for h in config.HOP_CDF_LEVELS:
            cdf_m = entry['cdf_mean'][h]
            cdf_s = entry['cdf_std'][h]
            row += f"{cdf_m:>8.3f}+/-{cdf_s:<5.3f}"
        report(row)
    report("=" * 102)

    best = models[0]
    exact = summary[best]["cdf_mean"][0]
    adjacent = summary[best]["cdf_mean"][1]
    report(
        f"\n  {best}: {exact:.1%} exact, {adjacent:.1%} within one hop. "
        f"Of the {1 - exact:.1%} of errors, "
        f"{(adjacent - exact) / max(1 - exact, 1e-9):.0%} land on a bus "
        "adjacent to the true one."
    )

    with open(args.out_dir / "hop_per_fold.pkl", "wb") as fh:
        pickle.dump(deltas, fh, protocol=pickle.HIGHEST_PROTOCOL)
    with open(args.out_dir / "summary.pkl", "wb") as fh:
        pickle.dump(summary, fh, protocol=pickle.HIGHEST_PROTOCOL)
    report.save(args.out_dir / "hop_distance.txt")


if __name__ == "__main__":
    main()
