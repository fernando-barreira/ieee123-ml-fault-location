"""Stage 6 — PMU placement search (FSNR).

Runs forward selection followed by neighbourhood refinement over the candidate
pool, scoring each placement with the paired MLP / random-forest proxies on the
frozen ``fsnr`` split.  See :mod:`src.placement.fsnr` for the algorithm and
:mod:`src.placement.proxies` for why two proxies are used.

The two phases have different budgets, and the distinction matters.  Forward
selection is grown to ``--k`` (default 9) purely to obtain the marginal value
of each additional sensor: because it is nested, every prefix of the path is
the placement it would have chosen at that budget.  Refinement is **not**
nested — a swap accepted at one budget says nothing about another — so it runs
on the prefix of length ``--refine-at`` (default 5, the study's operating
point), and that refined set is the reported placement.

Refining at the larger budget and then taking a prefix, as an earlier version
did, can yield a placement neither phase actually selected: a swap landing on
an early position of the path silently rewrites every smaller prefix.

Outputs, in ``results/fsnr/``:

===========================  =================================================
``pmus_fsnr.json``           Final placement, forward path, full history.
                             This is the canonical placement file read by
                             stage 7.
``fsnr_final.pkl``           The above plus the complete evaluation cache.
``fsnr_log.csv``             One row per evaluation (audit trail).
``fsnr_progress.json``       Resume point; delete to start over.
``fsnr_log.txt``             Timestamped log of accepted moves.
===========================  =================================================

Expect a few hours on a single GPU. The run is resumable: interrupt it and
launch it again with the same arguments.

Usage
-----
.. code-block:: bash

    python scripts/06_run_fsnr.py
    python scripts/06_run_fsnr.py --k 9 --refine-at 5 --radius 3
    python scripts/06_run_fsnr.py --fresh    # ignore any cached progress
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config
from src.graph import load_graph
from src.placement.fsnr import (
    ProgressStore,
    features_for_subset,
    forward_selection,
    geometric_mean,
    map_columns_to_pmus,
    neighborhood_refinement,
    subset_key,
)
from src.placement.proxies import train_mlp_proxy, train_rf_proxy
from src.pmu_candidates import load_candidates
from src.scaling import fit_transform_groupwise
from src.splits import load_splits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--dataset", type=config.resolve_path, default=config.FEATURES_ALL_CSV)
    parser.add_argument("--splits", type=config.resolve_path, default=config.SPLITS_PKL)
    parser.add_argument("--candidates", type=config.resolve_path, default=config.CANDIDATES_PKL)
    parser.add_argument("--graph", type=config.resolve_path, default=config.GRAPH_PKL)
    parser.add_argument("--out-dir", type=config.resolve_path, default=config.FSNR_DIR)
    parser.add_argument(
        "--k", type=int, default=config.FSNR_K_TARGET,
        help="Length of the forward-selection path (the marginal-value curve).",
    )
    parser.add_argument(
        "--refine-at", type=int, default=config.FSNR_REFINE_K,
        help="Budget at which neighbourhood refinement runs; its output is the "
             "reported placement. Defaults to the study's operating point.",
    )
    parser.add_argument("--radius", type=int, default=config.FSNR_NEIGHBOR_RADIUS)
    parser.add_argument(
        "--max-rounds", type=int, default=config.FSNR_MAX_REFINE_ROUNDS
    )
    parser.add_argument("--seed", type=int, default=config.SEED)
    parser.add_argument(
        "--fresh", action="store_true", help="Ignore any cached progress."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    # Per-evaluation seeding only yields reproducible scores if the kernels
    # themselves are deterministic. cuDNN autotuning picks different algorithms
    # depending on transient GPU state, which would reintroduce run-to-run
    # variation into the very comparison the seeding is meant to make clean.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config.ensure_dir(args.out_dir)

    print(f"Device  : {device}")
    print(f"Dataset : {args.dataset}")
    print(f"Splits  : {args.splits}")
    print(f"Output  : {args.out_dir}")

    # ── data ──────────────────────────────────────────────────────────────
    df = pd.read_csv(args.dataset)
    print(f"\nDataset shape: {df.shape}")

    encoder = LabelEncoder().fit(df[config.TARGET_COL])
    y_all = encoder.transform(df[config.TARGET_COL])
    n_classes = len(encoder.classes_)
    print(f"Classes: {n_classes}")

    splits = load_splits(args.splits, y=df[config.TARGET_COL].to_numpy())
    idx_train, idx_val = splits["fsnr_tr"], splits["fsnr_va"]
    print(f"fsnr_tr: {len(idx_train):,} samples | fsnr_va: {len(idx_val):,} samples")

    features_df = df.drop(columns=[config.TARGET_COL])
    column_to_pmu = map_columns_to_pmus(
        list(features_df.columns),
        config.TARGET_COL,
        config.SUBSET_DEPENDENT_FEATURES,
    )
    detected = sorted(
        {p for p in column_to_pmu.values() if p is not None},
        key=lambda b: int(b) if b.isdigit() else 10**9,
    )
    print(f"PMUs present in the dataset: {len(detected)}")

    drop_columns = (
        set(config.SUBSET_DEPENDENT_FEATURES)
        if config.FSNR_DROP_SUBSET_DEPENDENT
        else set()
    )
    if drop_columns:
        present = sorted(drop_columns & set(features_df.columns))
        print(
            f"Excluded during the search (computed over the full candidate "
            f"pool, so they would leak): {present}"
        )

    train_df = features_df.iloc[idx_train].reset_index(drop=True)
    val_df = features_df.iloc[idx_val].reset_index(drop=True)
    y_train, y_val = y_all[idx_train], y_all[idx_val]

    # ── candidates and topology ───────────────────────────────────────────
    candidates = [str(c) for c in load_candidates(args.candidates)["candidates"]]
    usable = [c for c in candidates if c in detected]
    if len(usable) < len(candidates):
        print(
            f"Candidates without features in the dataset (skipped): "
            f"{sorted(set(candidates) - set(usable))}"
        )
    candidates = usable
    print(f"Usable candidates: {len(candidates)}")

    graph = load_graph(args.graph)
    print(f"Graph: {graph.number_of_nodes()} buses, {graph.number_of_edges()} edges")

    # ── scoring function ──────────────────────────────────────────────────
    mlp_cfg = dict(config.FSNR_MLP)
    mlp_cfg.setdefault("focal_gamma", config.FOCAL_GAMMA)
    mlp_cfg.setdefault("label_smoothing", config.LABEL_SMOOTHING)

    def evaluate(pmus: list[str]) -> dict:
        """Score one placement: geometric mean of the two proxy accuracies."""
        columns = features_for_subset(column_to_pmu, pmus, drop_columns)
        if not columns:
            return {"geom": 0.0, "mlp": 0.0, "rf": 0.0, "n_features": 0}

        x_train, x_val, noise_columns = fit_transform_groupwise(
            train_df[columns], val_df[columns]
        )
        score_mlp = train_mlp_proxy(
            x_train, y_train, x_val, y_val,
            n_classes=n_classes,
            noise_columns=noise_columns,
            cfg=mlp_cfg,
            device=device,
            seed=args.seed,
        )
        score_rf = train_rf_proxy(
            x_train, y_train, x_val, y_val, cfg=config.FSNR_RF, seed=args.seed
        )
        return {
            "geom": geometric_mean(score_mlp, score_rf),
            "mlp": score_mlp,
            "rf": score_rf,
            "n_features": int(x_train.shape[1]),
        }

    # ── run ───────────────────────────────────────────────────────────────
    store = ProgressStore(args.out_dir)
    cache: dict[str, dict] = {}
    if not args.fresh:
        previous = store.load()
        cache = previous.get("cache", {})
        if cache:
            print(f"\nResuming with {len(cache)} cached evaluation(s).")

    t0 = time.time()
    selected, forward_history = forward_selection(
        candidates, args.k, evaluate, cache, store
    )
    t_forward = time.time() - t0
    forward_score = forward_history[-1]["score_geom"] if forward_history else 0.0
    print(
        f"\n[forward done] {selected}  geom={forward_score:.4f}  "
        f"({t_forward / 60:.1f} min)"
    )

    # ── refinement, at the operating point ────────────────────────────────
    # Refinement runs on the length-FSNR_REFINE_K prefix, not on the full
    # forward path. Forward selection is nested, so its prefixes are valid
    # placements at every budget; refinement is not, so a swap accepted at
    # K=9 says nothing about the best placement at K=5 — and if it lands on an
    # early position of the path it silently changes every smaller prefix too.
    refine_k = min(args.refine_at, len(selected))
    prefix = selected[:refine_k]
    prefix_result = cache.get(subset_key(prefix))
    if prefix_result is None:
        prefix_result = evaluate(prefix)
        cache[subset_key(prefix)] = prefix_result
    prefix_score = prefix_result["geom"]

    print(f"\n[operating point] refining the K={refine_k} prefix {prefix} "
          f"(geom={prefix_score:.4f})")

    t0 = time.time()
    final, final_score, refinement_history = neighborhood_refinement(
        list(prefix), graph, candidates, evaluate, cache, store,
        current_score=prefix_score,
        radius=args.radius,
        max_rounds=args.max_rounds,
    )
    t_refine = time.time() - t0

    final_result = cache[subset_key(final)]
    print(
        f"\n[refinement done] {final}  geom={final_result['geom']:.4f}  "
        f"MLP={final_result['mlp']:.4f}  RF={final_result['rf']:.4f}  "
        f"({t_refine / 60:.1f} min)"
    )
    swapped = [b for b in prefix if b not in final]
    if swapped:
        print(f"  refinement replaced {swapped} "
              f"(gain {final_result['geom'] - prefix_score:+.4f})")

    # ── benchmark against the heuristic placement ─────────────────────────
    baseline = sorted(config.HEURISTIC_PMU_SET, key=lambda b: int(b))
    if all(p in candidates for p in baseline):
        key = subset_key(baseline)
        if key not in cache:
            print(f"\nScoring the heuristic placement {baseline} for reference...")
            cache[key] = evaluate(baseline)
        reference = cache[key]
        print(f"\n  heuristic     K={len(baseline)} {baseline}: "
              f"geom={reference['geom']:.4f}")
        print(f"  FSNR forward  K={refine_k} "
              f"{sorted(prefix, key=lambda b: int(b))}: geom={prefix_score:.4f}")
        print(f"  FSNR refined  K={len(final)} "
              f"{sorted(final, key=lambda b: int(b))}: "
              f"geom={final_result['geom']:.4f}")
        if len(baseline) == len(final):
            print(f"  improvement over the heuristic placement: "
                  f"{final_result['geom'] - reference['geom']:+.4f} "
                  f"({final_result['geom'] / reference['geom'] - 1:+.1%})")

    # ── persist ───────────────────────────────────────────────────────────
    result = {
        # The reported placement: the refined set at the operating point.
        "selected_pmus": final,
        "selected_sorted": sorted(final, key=lambda b: int(b) if b.isdigit() else 10**9),
        "refine_k": refine_k,
        "refined_placement": sorted(final, key=lambda b: int(b) if b.isdigit() else 10**9),
        "forward_prefix_at_refine_k": list(prefix),
        "refinement_changed_prefix": bool(swapped),
        # Nested; use prefixes for the marginal-value curve at other budgets.
        "forward_path": [h["added"] for h in forward_history],
        "forward_history": forward_history,
        "refinement_history": refinement_history,
        "final_scores": final_result,
        "candidates": candidates,
        "n_evaluations": len(cache),
        "elapsed_minutes": {
            "forward": round(t_forward / 60, 2),
            "refinement": round(t_refine / 60, 2),
            "total": round((t_forward + t_refine) / 60, 2),
        },
        "config": {
            "k_target": args.k,
            "refine_at": refine_k,
            "neighbour_radius": args.radius,
            "max_refine_rounds": args.max_rounds,
            "seed": args.seed,
            "drop_subset_dependent": config.FSNR_DROP_SUBSET_DEPENDENT,
            "score": "geometric mean of proxy MLP and proxy RF top-1 accuracy",
            "mlp": mlp_cfg,
            "rf": config.FSNR_RF,
        },
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    with open(args.out_dir / "pmus_fsnr.json", "w") as fh:
        json.dump(result, fh, indent=2, default=str)
    with open(args.out_dir / "fsnr_final.pkl", "wb") as fh:
        pickle.dump({**result, "cache": cache}, fh, protocol=pickle.HIGHEST_PROTOCOL)
    log_csv = store.flush_csv()

    print("\n" + "=" * 70)
    print("FSNR complete.")
    print(f"  placement (K={refine_k}, refined) : {result['selected_sorted']}")
    print(f"  forward path (K={args.k})         : {result['forward_path']}")
    if swapped:
        print(f"  note: refinement replaced {swapped}; the reported placement "
              f"is NOT the forward prefix.")
    print(
        f"  proxy scores : MLP={final_result['mlp']:.4f}  "
        f"RF={final_result['rf']:.4f}  geom={final_result['geom']:.4f}"
    )
    print(f"  evaluations  : {len(cache)} in "
          f"{(t_forward + t_refine) / 60:.1f} min")
    print(f"  outputs      : {args.out_dir}")
    print(f"                 {log_csv.name}")
    print("=" * 70)


if __name__ == "__main__":
    main()