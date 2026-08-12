"""The sensor-budget table (Table I of the paper).

Reports the proxy score of three sensor configurations, which together justify
five PMUs as the operating point:

===============================  =========================================
``Substation only``              A single PMU at the feeder head.  This is
                                 the deployment a utility already has, so it
                                 is the honest zero-cost reference — and it
                                 is deliberately *not* the first bus of the
                                 forward path, which is chosen for
                                 information rather than for practicality.
``FSNR set``                     The refined placement at the operating
                                 point.
``FSNR set + next candidates``   The forward path extended to its full
                                 length, showing what more sensors buy.
===============================  =========================================

All three scores come from the evaluation cache written by stage 6, so this
script re-runs nothing and cannot disagree with the search that produced the
placement.  A configuration missing from the cache is reported as such rather
than silently recomputed under different conditions.

The comparison is at *proxy* level by design.  Retraining the full model stack
at each budget would cost days and would answer a question the paper does not
ask: the placement search is what selects the budget, so the budget must be
justified with the search's own criterion.

This reads only the stage-6 evaluation cache, so it can run as soon as the
placement search finishes — it does not belong anywhere in the stage 9-17
sequence and is kept out of it.

Usage
-----
.. code-block:: bash

    python tools/sensor_budget_table.py
    python tools/sensor_budget_table.py --budgets 1,5,9
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config
from src.analysis import Report
from src.placement.fsnr import subset_key


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--fsnr-dir", type=config.resolve_path, default=config.FSNR_DIR)
    parser.add_argument("--out-dir", type=config.resolve_path,
                        default=config.ANALYSIS_DIR / "sensor_budget")
    parser.add_argument("--budgets", type=str, default="1,5,9")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config.ensure_dir(args.out_dir)
    budgets = sorted({int(b) for b in args.budgets.split(",") if b.strip()})
    report = Report()

    cache_path = args.fsnr_dir / "fsnr_final.pkl"
    if not cache_path.exists():
        raise SystemExit(
            f"{cache_path} not found. Run scripts/06_run_fsnr.py first."
        )
    with open(cache_path, "rb") as fh:
        stored = pickle.load(fh)

    cache = stored.get("cache", {})
    forward_path = [str(b) for b in stored.get("forward_path", [])]
    refined = [str(b) for b in stored.get("refined_placement",
                                          stored.get("selected_pmus", []))]
    refine_k = stored.get("refine_k", len(refined))

    report("=" * 78)
    report("  SENSOR-BUDGET COMPARISON (proxy score on the FSNR pool)")
    report("=" * 78)
    report(f"  forward path      : {forward_path}")
    report(f"  refined placement : {sorted(refined, key=int)} (K={refine_k})")
    report(f"  cached evaluations: {len(cache)}")

    rows = []
    for k in budgets:
        if k == 1:
            buses = [config.SUBSTATION_BUS]
            label = f"Substation only (bus {config.SUBSTATION_BUS})"
        elif k == refine_k and refined:
            buses = list(refined)
            label = f"FSNR set {{{', '.join(sorted(buses, key=int))}}}"
        else:
            if k > len(forward_path):
                report(f"\n  K={k}: the forward path has only "
                       f"{len(forward_path)} entries, skipped")
                continue
            buses = forward_path[:k]
            extra = k - refine_k
            label = (f"FSNR set + {extra} next-best candidates"
                     if extra > 0 else f"FSNR forward prefix (K={k})")

        entry = cache.get(subset_key(buses))
        if entry is None:
            report(
                f"\n  K={k} {sorted(buses, key=int)}: not in the FSNR cache. "
                "Re-run stage 6 (the cache is preserved between runs) so this "
                "configuration is scored under exactly the same conditions as "
                "the others."
            )
            continue

        rows.append({
            "configuration": label,
            "n_pmus": k,
            "buses": " ".join(sorted(buses, key=int)),
            "geometric_top1": entry["geom"],
            "proxy_mlp_top1": entry["mlp"],
            "proxy_rf_top1": entry["rf"],
            "n_features": entry["n_features"],
        })

    if not rows:
        raise SystemExit("No configuration could be scored from the cache.")

    report("\n" + "=" * 78)
    report(f"  {'Configuration':<40}{'#PMU':>6}{'Geometric Top-1':>18}")
    report("  " + "-" * 74)
    for row in rows:
        report(f"  {row['configuration']:<40}{row['n_pmus']:>6}"
               f"{row['geometric_top1']:>18.3f}")
    report("=" * 78)

    report(f"\n  {'Configuration':<40}{'proxy MLP':>12}{'proxy RF':>11}{'#feat':>8}")
    for row in rows:
        report(f"  {row['configuration']:<40}{row['proxy_mlp_top1']:>12.4f}"
               f"{row['proxy_rf_top1']:>11.4f}{row['n_features']:>8}")

    # Marginal value: the sentence the table exists to support.
    by_budget = {r["n_pmus"]: r["geometric_top1"] for r in rows}
    if 1 in by_budget and refine_k in by_budget:
        low_gain = by_budget[refine_k] / by_budget[1]
        report(f"\n  One to {refine_k} PMUs improves the proxy score by "
               f"{low_gain:.2f}x.")
        if max(by_budget) > refine_k:
            top = max(by_budget)
            high_gain = by_budget[top] / by_budget[refine_k]
            share = by_budget[refine_k] / by_budget[top]
            report(f"  {refine_k} to {top} PMUs improves it by only "
                   f"{high_gain:.2f}x.")
            report(f"  The {refine_k}-PMU configuration captures {share:.0%} of "
                   f"the score attainable with {top}, which is what justifies it "
                   "as the operating point.")

    report("\n  LaTeX rows for Table I:")
    for row in rows:
        report(f"    {row['configuration']} & {row['n_pmus']} & "
               f"{row['geometric_top1']:.3f} \\\\")

    frame = pd.DataFrame(rows)
    frame.to_csv(args.out_dir / "sensor_budget.csv", index=False)
    with open(args.out_dir / "sensor_budget.json", "w") as fh:
        json.dump({"rows": rows, "forward_path": forward_path,
                   "refined_placement": refined, "refine_k": refine_k},
                  fh, indent=2)
    report.save(args.out_dir / "sensor_budget.txt")


if __name__ == "__main__":
    main()
