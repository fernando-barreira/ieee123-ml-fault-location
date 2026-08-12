"""Stage 13 — bootstrap confidence intervals on the pooled predictions.

What the interval means
-----------------------
It quantifies **scenario sampling uncertainty with the trained models held
fixed**: how much the reported Top-1 would move if the 75 000 test scenarios
were redrawn from the same simulation distribution.  It does not capture
variability from retraining, from a different placement, or from a different
feeder — those are larger and are not estimated here.

Why pooled rather than over folds
---------------------------------
Bootstrapping the five per-fold means resamples five numbers.  Only
:math:`\\binom{2\\cdot5-1}{5} = 126` distinct resamples exist, so the resulting
percentile interval is discretised into a handful of achievable values and its
endpoints are essentially artefacts of that granularity.  Pooling the
per-sample correctness indicators across folds gives 75 000 exchangeable
Bernoulli outcomes, which is what the percentile bootstrap assumes.

The folds are disjoint test sets, so pooling double-counts nothing; each sample
is predicted exactly once, by the model that did not see it.

Why the binomial shortcut is exact
----------------------------------
For a binary vector, the mean of a size-:math:`N` resample drawn with
replacement is distributed **exactly** as :math:`\\mathrm{Binomial}(N,
\\hat p)/N`, because each draw is an independent Bernoulli(:math:`\\hat p`).
Sampling from that binomial is therefore mathematically identical to the
classical percentile bootstrap, without materialising a
:math:`10\\,000 \\times 75\\,000` index array.

On significance testing
-----------------------
No pairwise test is reported.  With five paired folds the smallest attainable
two-sided Wilcoxon signed-rank p-value is :math:`2/2^5 = 0.0625`, so rejection
at :math:`\\alpha=0.05` is impossible whatever the effect size.  Non-overlapping
intervals are reported as an indication of separation under this protocol, not
as a formal test.

Usage
-----
.. code-block:: bash

    python scripts/13_bootstrap_ci.py
    python scripts/13_bootstrap_ci.py --n-boot 20000
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config
from src.analysis import Report, iter_folds, load_preprocessing, ordered_models


def bootstrap_interval(
    correct: np.ndarray, n_boot: int, ci: tuple[float, float], rng
) -> tuple[float, float, float, int]:
    """Percentile bootstrap of the mean of a binary vector.

    Returns ``(point_estimate, lower, upper, n_samples)``.
    """
    n = len(correct)
    point = float(correct.mean())
    draws = rng.binomial(n, point, size=n_boot) / n
    low, high = np.percentile(draws, ci)
    return point, float(low), float(high), n


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--cv-dir", type=config.resolve_path, default=config.CV_DIR)
    parser.add_argument("--baseline-dir", type=config.resolve_path,
                        default=config.BASELINE_DIR)
    parser.add_argument("--out-dir", type=config.resolve_path,
                        default=config.ANALYSIS_DIR / "bootstrap")
    parser.add_argument("--n-boot", type=int, default=config.N_BOOTSTRAP)
    parser.add_argument("--seed", type=int, default=config.SEED)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config.ensure_dir(args.out_dir)
    rng = np.random.default_rng(args.seed)
    report = Report()

    pre = load_preprocessing(args.cv_dir)
    n_folds = pre["n_folds"]

    report("=" * 78)
    low_pct, high_pct = config.BOOTSTRAP_CI
    report(f"  POOLED BOOTSTRAP — Top-1, {args.n_boot} resamples, "
           f"{low_pct}-{high_pct}% percentile interval")
    report("=" * 78)

    correct: dict[str, list[np.ndarray]] = {}
    for fold, results in iter_folds(args.cv_dir, n_folds, args.baseline_dir):
        for name, entry in results.items():
            hit = (np.asarray(entry["preds"]) == np.asarray(entry["y_true"]))
            correct.setdefault(name, []).append(hit.astype(np.int8))

    if not correct:
        raise SystemExit("No fold results found. Run stage 9 (and 10) first.")

    models = ordered_models(correct)
    report(f"\n  {'Model':<24}{'N':>8}{'Top-1':>9}{'95% interval':>22}"
           f"{'fold std':>11}")
    report("  " + "-" * 74)

    summary = {}
    for name in models:
        per_fold = correct[name]
        pooled = np.concatenate(per_fold)
        point, low, high, n = bootstrap_interval(
            pooled, args.n_boot, config.BOOTSTRAP_CI, rng
        )
        fold_std = float(np.std([float(h.mean()) for h in per_fold]))
        summary[name] = {
            "top1": point, "ci_low": low, "ci_high": high,
            "n_samples": n, "n_folds": len(per_fold),
            "fold_top1": [float(h.mean()) for h in per_fold],
            "fold_std": fold_std,
        }
        report(f"  {name:<24}{n:>8}{point:>9.4f}"
               f"   [{low:.4f}, {high:.4f}]{fold_std:>11.4f}")

    report("\n  LaTeX:")
    for name in models:
        entry = summary[name]
        report(f"    {name:<24} -> ${entry['top1']:.3f}$ "
               f"$[{entry['ci_low']:.3f},\\,{entry['ci_high']:.3f}]$")

    # Separation is reported descriptively; see the module docstring for why no
    # significance test accompanies it.
    report("\n  Interval overlap (descriptive, not a significance test):")
    for i, a in enumerate(models):
        for b in models[i + 1:]:
            overlap = not (summary[a]["ci_high"] < summary[b]["ci_low"]
                           or summary[b]["ci_high"] < summary[a]["ci_low"])
            if not overlap:
                report(f"    {a} vs {b}: intervals do not overlap "
                       f"(delta = {summary[a]['top1'] - summary[b]['top1']:+.4f})")

    with open(args.out_dir / "bootstrap.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    report.save(args.out_dir / "bootstrap.txt")


if __name__ == "__main__":
    main()
