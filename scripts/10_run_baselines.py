"""Stage 10 — classical baselines on the same test folds.

Evaluates two learning-free reference methods — impedance matching and a purely
topological rule — on **exactly** the fold test sets used in stage 9, so the
comparison in the paper is like for like.  See :mod:`src.baselines` for what
each method computes and why both are needed.
The reactance-only variant of the impedance baseline discussed in the paper
is also computed here, by using the flag.

This stage must run after stage 9, from which it reads the label encoder and
the fold indices.

A note on the index space
-------------------------
``fold_indices.pkl`` stores positions **into the full dataset**.  The folds as
stored in ``splits.pkl`` are local to the cross-validation pool, and an earlier
version of this stage used those local positions to index the full-length
arrays.  That silently evaluated the baselines on the wrong rows — rows drawn
from the head of the dataset rather than from the fold's test set, overlapping
the pools reserved for placement and hyper-parameter search — while producing
plausible-looking numbers.  Stage 9 now writes global positions and this stage
asserts the range, so the two cannot drift apart again.

Usage
-----
.. code-block:: bash

    python scripts/10_run_baselines.py
    python scripts/10_run_baselines.py --absolute      # unnormalised score
    python scripts/10_run_baselines.py --calibration rank
    python scripts/10_run_baselines.py --with-reactance      # reactance-only variant of the impedance criterion
"""

from __future__ import annotations

import argparse
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config
from src.baselines import (
    build_impedance_graph,
    compile_circuit,
    compute_paths,
    extract_observations,
    infer_pmus_from_columns,
    score_impedance,
    score_reactance,
    score_topological,
    scores_to_probs,
)
from src.training import metrics_from_probs, require

BASELINE_NAMES = ("Baseline Impedance", "Baseline Reactance",
                  "Baseline Topological")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--dss", type=config.resolve_path, default=config.DSS_MASTER)
    parser.add_argument("--dataset", type=config.resolve_path,
                        default=config.features_csv(5))
    parser.add_argument("--cv-dir", type=config.resolve_path, default=config.CV_DIR)
    parser.add_argument("--out-dir", type=config.resolve_path, default=config.BASELINE_DIR)
    parser.add_argument(
        "--absolute", action="store_true",
        help="Use the absolute impedance residual instead of the relative one.",
    )
    parser.add_argument(
        "--calibration", choices=("softmax", "rank"),
        default=config.BASELINE_CALIBRATION,
    )
    parser.add_argument(
        "--with-reactance", action="store_true",
        help="Also evaluate the reactance-only variant of the impedance "
             "baseline. Off by default: it scores the same as the modulus "
             "criterion, so it belongs in a sentence rather than in the "
             "results table. Its value is diagnostic — see the module "
             "docstring.",
    )
    parser.add_argument(
        "--dominant-phase", action="store_true",
        help="Score using only the phase with the largest superimposed current.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config.ensure_dir(args.out_dir)
    require("py_dss_interface")
    normalize = not args.absolute

    print("=" * 72)
    print("  Classical baselines: impedance matching and topology")
    print("=" * 72)
    print(f"  impedance score : {'relative' if normalize else 'absolute'}")
    print(f"  calibration     : {args.calibration}")

    # ── stage 9 artefacts ────────────────────────────────────────────────
    preprocessing_path = args.cv_dir / "preprocessing_global.pkl"
    folds_path = args.cv_dir / "fold_indices.pkl"
    if not folds_path.exists():
        raise SystemExit(
            f"{folds_path} not found. Run scripts/09_cross_validate.py first."
        )
    with open(preprocessing_path, "rb") as fh:
        preprocessing = pickle.load(fh)
    with open(folds_path, "rb") as fh:
        fold_indices = pickle.load(fh)

    encoder = preprocessing["target_encoder"]
    n_classes = preprocessing["n_classes"]
    print(f"\n  {n_classes} classes | {len(fold_indices)} folds")

    if preprocessing.get("fold_index_space") != "global dataset positions":
        raise SystemExit(
            "preprocessing_global.pkl does not declare global fold indices. "
            "It was produced by an older stage 9 whose fold indices were local "
            "to the cross-validation pool; using them here would evaluate the "
            "baselines on the wrong samples. Re-run stage 9."
        )

    # ── dataset ──────────────────────────────────────────────────────────
    print(f"\n  reading {args.dataset}")
    df = pd.read_csv(args.dataset)
    print(f"    shape: {df.shape}")

    max_index = max(int(idx["test"].max()) for idx in fold_indices.values())
    if max_index >= len(df):
        raise SystemExit(
            f"Fold indices reach {max_index} but the dataset has {len(df)} rows."
        )

    pmus = infer_pmus_from_columns(df)
    if not pmus:
        raise SystemExit(
            "No Z_est_<bus>_f1 columns found; cannot infer the PMU placement."
        )
    print(f"    PMUs read from the feature columns: {pmus}")

    y = encoder.transform(df[config.TARGET_COL])

    # ── electrical graph and path impedances ─────────────────────────────
    import py_dss_interface

    print(f"\n  building the impedance graph from {args.dss}")
    dss = compile_circuit(py_dss_interface.DSS(), args.dss)
    graph = build_impedance_graph(dss, config.BASELINE_ZERO_IMPEDANCE_FLOOR)

    # Fictitious buses stay in the graph as transit nodes but must never be
    # candidate answers; the label space is asserted against that here.
    candidate_buses = [str(c).lower() for c in encoder.classes_]
    fictitious = sorted(set(candidate_buses) & {b.lower() for b in config.FICTITIOUS_BUSES})
    if fictitious:
        raise SystemExit(
            f"Fictitious buses present in the label space: {fictitious}. "
            "They are regulator or switch terminals, not fault locations."
        )

    z_path, hops = compute_paths(graph, [p.lower() for p in pmus], candidate_buses)
    finite = np.abs(z_path)
    print(f"    |Z_path| range: [{finite.min():.4f}, {finite.max():.4f}] ohm")
    print(f"    hop range: [{hops.min()}, {hops.max()}]")
    with open(args.out_dir / "z_paths.pkl", "wb") as fh:
        pickle.dump(
            {"z_path": z_path, "hops": hops, "pmus": pmus,
             "candidate_buses": candidate_buses},
            fh, protocol=pickle.HIGHEST_PROTOCOL,
        )

    # ── observations ─────────────────────────────────────────────────────
    z_app, weights = extract_observations(
        df, pmus, config.PHASES,
        z_clip=config.BASELINE_Z_APP_CLIP,
        dominant_phase_only=args.dominant_phase,
    )
    print(f"    Z_app {z_app.shape}, mean |Z| = {np.abs(z_app).mean():.3f} ohm")

    # ── per fold ─────────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("  Evaluating on the stage-9 test folds")
    print("=" * 72)

    collected: dict[str, list[dict]] = {name: [] for name in BASELINE_NAMES}
    latencies: dict[str, list[float]] = {name: [] for name in BASELINE_NAMES}

    for fold in sorted(fold_indices):
        test_idx = fold_indices[fold]["test"]
        z_test, w_test, y_test = z_app[test_idx], weights[test_idx], y[test_idx]
        results: dict[str, dict] = {}

        t0 = time.perf_counter()
        scores = score_impedance(z_test, w_test, z_path, normalize_by_path=normalize)
        probs, temperature = scores_to_probs(
            scores, args.calibration, config.BASELINE_SOFTMAX_TEMP
        )
        latency_imp = (time.perf_counter() - t0) / len(test_idx) * 1000.0
        metrics_imp, preds = metrics_from_probs(probs, y_test)
        results["Baseline Impedance"] = dict(
            probs=probs, preds=preds, y_true=y_test, metrics=metrics_imp,
            history=None, importance=None, training_time=0.0,
            inference_time_ms=latency_imp, n_params=0,
            config={"normalize_by_path": normalize,
                    "calibration": args.calibration, "temperature": temperature},
        )

        if args.with_reactance:
            t0 = time.perf_counter()
            scores_x = score_reactance(z_test, w_test, z_path,
                                       normalize_by_path=normalize)
            probs_x, temperature_x = scores_to_probs(
                scores_x, args.calibration, config.BASELINE_SOFTMAX_TEMP
            )
            latency_x = (time.perf_counter() - t0) / len(test_idx) * 1000.0
            metrics_x, preds_x = metrics_from_probs(probs_x, y_test)
            results["Baseline Reactance"] = dict(
                probs=probs_x, preds=preds_x, y_true=y_test, metrics=metrics_x,
                history=None, importance=None, training_time=0.0,
                inference_time_ms=latency_x, n_params=0,
                config={"criterion": "reactance only",
                        "normalize_by_path": normalize,
                        "calibration": args.calibration,
                        "temperature": temperature_x},
            )
            collected["Baseline Reactance"].append(metrics_x)
            latencies["Baseline Reactance"].append(latency_x)

        t0 = time.perf_counter()
        scores_topo = score_topological(w_test, hops)
        probs_topo, temperature_topo = scores_to_probs(
            scores_topo, args.calibration, config.BASELINE_SOFTMAX_TEMP
        )
        latency_topo = (time.perf_counter() - t0) / len(test_idx) * 1000.0
        metrics_topo, preds_topo = metrics_from_probs(probs_topo, y_test)
        results["Baseline Topological"] = dict(
            probs=probs_topo, preds=preds_topo, y_true=y_test, metrics=metrics_topo,
            history=None, importance=None, training_time=0.0,
            inference_time_ms=latency_topo, n_params=0,
            config={"calibration": args.calibration, "temperature": temperature_topo},
        )

        with open(args.out_dir / f"fold_{fold}_results.pkl", "wb") as fh:
            pickle.dump(results, fh, protocol=pickle.HIGHEST_PROTOCOL)

        collected["Baseline Impedance"].append(metrics_imp)
        collected["Baseline Topological"].append(metrics_topo)
        latencies["Baseline Impedance"].append(latency_imp)
        latencies["Baseline Topological"].append(latency_topo)

        print(f"  fold {fold} ({len(test_idx):,} samples)")
        print(f"    impedance   T1={metrics_imp['top1']:.4f} "
              f"T3={metrics_imp['top3']:.4f} T5={metrics_imp['top5']:.4f}")
        if args.with_reactance:
            print(f"    reactance   T1={metrics_x['top1']:.4f} "
                  f"T3={metrics_x['top3']:.4f} T5={metrics_x['top5']:.4f}")
        print(f"    topological T1={metrics_topo['top1']:.4f} "
              f"T3={metrics_topo['top3']:.4f} T5={metrics_topo['top5']:.4f}")

    # ── summary ──────────────────────────────────────────────────────────
    summary: dict[str, dict] = {}
    rows = []
    for name in BASELINE_NAMES:
        if not collected[name]:
            continue
        entry: dict[str, tuple] = {}
        for metric in ("top1", "top3", "top5"):
            values = [m[metric] for m in collected[name]]
            entry[metric] = (float(np.mean(values)), float(np.std(values)), values)
        entry["inference_time_ms"] = (
            float(np.mean(latencies[name])), float(np.std(latencies[name])),
            latencies[name],
        )
        entry["training_time"] = (0.0, 0.0, [0.0] * len(collected[name]))
        summary[name] = entry
        rows.append({
            "model": name,
            "top1_mean": entry["top1"][0], "top1_std": entry["top1"][1],
            "top3_mean": entry["top3"][0], "top3_std": entry["top3"][1],
            "top5_mean": entry["top5"][0], "top5_std": entry["top5"][1],
            "inference_time_ms_mean": entry["inference_time_ms"][0],
        })

    with open(args.out_dir / "summary.pkl", "wb") as fh:
        pickle.dump(summary, fh, protocol=pickle.HIGHEST_PROTOCOL)
    pd.DataFrame(rows).to_csv(args.out_dir / "summary.csv", index=False)

    chance = 1.0 / n_classes
    print("\n" + "=" * 72)
    print(f"  SUMMARY over {len(collected[BASELINE_NAMES[0]])} folds "
          f"(random guessing = {chance:.2%})")
    print("=" * 72)
    for name in BASELINE_NAMES:
        if name not in summary:
            continue
        entry = summary[name]
        print(f"  {name}")
        for metric in ("top1", "top3", "top5"):
            mean, std, _ = entry[metric]
            print(f"    {metric.upper():<6}: {mean:.4f} +/- {std:.4f} "
                  f"({mean / chance:.1f}x chance)")
    if "Baseline Impedance" in summary and "Baseline Reactance" in summary:
        modulus = summary["Baseline Impedance"]["top1"][0]
        reactance = summary["Baseline Reactance"]["top1"][0]
        print(f"\n  Reactance vs modulus criterion: {reactance:.4f} vs "
              f"{modulus:.4f} ({reactance - modulus:+.4f}).")
        print("  The reactance criterion is immune to a purely resistive fault "
              "path, so whatever gap remains to the learned models is not "
              "attributable to unknown fault resistance.")

    print(f"\n  outputs in {args.out_dir}")


if __name__ == "__main__":
    main()