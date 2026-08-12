"""Stage 16 — the one-shot holdout check.

The 5 000-scenario holdout pool has been untouched since stage 5.  It is
evaluated **once**, here, and its only purpose is to answer one question: did
the repeated use of the cross-validation pool during model development inflate
the reported accuracy?

The discipline this stage depends on
------------------------------------
Nothing here may feed back into the pipeline.  A discrepancy against the
cross-validation estimate is a *finding to report*, not a trigger to re-tune,
re-select or retrain.  The moment the holdout influences a decision it stops
being a holdout, and every number computed from it afterward is worth nothing.
This script only reads: the frozen splits, the saved models and scalers, and
the dataset.

What is compared
----------------
The comparison must be like for like, so the cross-validation reference is
recomputed here from the same artefacts rather than quoted from stage 9: each
fold's model evaluated on its own test fold, then averaged across folds. Two
holdout numbers are produced against it:

*per-fold* — each fold's model predicts the holdout using its own scalers, and
the five results are averaged. This is the apples-to-apples comparison, since
both sides are "one fold's model on data it did not see".

*ensemble* — the five models' probabilities are averaged into a single
prediction per sample. This is how the system would actually be deployed and is
normally at least as good as any single fold, so it is reported separately
rather than compared against the cross-validation figure.

Usage
-----
.. code-block:: bash

    python scripts/16_holdout_check.py
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import warnings

# RandomForest.predict_proba emits this once per parallel batch. In a previous
# run it buried 86 lines of real output under 6 500 lines of noise, which is a
# correctness problem in its own right: output nobody can read is output nobody
# checks.
warnings.filterwarnings("ignore", category=UserWarning,
                        message=".*sklearn.utils.parallel.delayed.*")
warnings.filterwarnings("ignore", category=FutureWarning)

import config
from src.analysis import (
    Report,
    build_feature_matrix,
    load_fold_indices,
    load_fold_model,
    load_preprocessing,
    load_scalers,
    mean_std,
    model_input,
    ordered_models,
    predict_probs,
)
from src.splits import load_splits
from src.training import metrics_from_probs

#: A gap larger than this between the cross-validation and holdout estimates is
#: flagged. It is a reporting threshold, not a pass/fail criterion.
DIVERGENCE_THRESHOLD = 0.02


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--dataset", type=config.resolve_path,
                        default=config.features_csv(5))
    parser.add_argument("--splits", type=config.resolve_path, default=config.SPLITS_PKL)
    parser.add_argument("--cv-dir", type=config.resolve_path, default=config.CV_DIR)
    parser.add_argument("--out-dir", type=config.resolve_path,
                        default=config.ANALYSIS_DIR / "holdout")
    parser.add_argument("--models", type=str, default=",".join(
        ["Deep Residual MLP", "MLP Baseline", "TabNet",
         "Random Forest", "LightGBM"]))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config.ensure_dir(args.out_dir)
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    report = Report()

    report("=" * 78)
    report("  HOLDOUT CHECK — evaluated once, read-only")
    report("=" * 78)

    pre = load_preprocessing(args.cv_dir)
    fold_indices = load_fold_indices(args.cv_dir)

    df = pd.read_csv(args.dataset)
    raw_all = build_feature_matrix(df, pre)
    targets = df[config.TARGET_COL].to_numpy()
    y_all = pre["target_encoder"].transform(targets)

    splits = load_splits(args.splits, y=targets)
    holdout_idx = splits["holdout"]
    raw_holdout, y_holdout = raw_all[holdout_idx], y_all[holdout_idx]

    # The holdout must not overlap anything the models were exposed to. This is
    # guaranteed by stage 5, and asserted here because the entire value of this
    # check rests on it.
    for fold, entry in fold_indices.items():
        for part in ("train", "val", "test"):
            overlap = np.intersect1d(holdout_idx, entry[part])
            if overlap.size:
                raise SystemExit(
                    f"Holdout overlaps fold {fold} '{part}' on {overlap.size} "
                    "sample(s). The check would be meaningless."
                )
    report(f"  holdout: {len(holdout_idx):,} scenarios, disjoint from every fold")
    report(f"  device: {device}")

    cross_validation: dict[str, list[dict]] = {m: [] for m in models}
    per_fold_holdout: dict[str, list[dict]] = {m: [] for m in models}
    probability_sum: dict[str, np.ndarray] = {}
    n_models_seen: dict[str, int] = {}

    for fold in range(pre["n_folds"]):
        if not (Path(args.cv_dir) / f"fold_{fold}" / "scalers.pkl").exists():
            report(f"\n  fold {fold}: not trained, skipped")
            continue
        scalers = load_scalers(args.cv_dir, fold)
        test_idx = fold_indices[fold]["test"]
        report(f"\n  -- fold {fold} --")

        for name in models:
            try:
                kind, model = load_fold_model(name, args.cv_dir, fold, pre, device)
            except (FileNotFoundError, KeyError) as exc:
                report(f"    {name}: not available ({exc}), skipped")
                continue

            probs = predict_probs(
                kind, model,
                model_input(kind, name, raw_all[test_idx], scalers, pre),
                device,
            )
            cv_metrics, _ = metrics_from_probs(probs, y_all[test_idx])
            cross_validation[name].append(cv_metrics)

            probs = predict_probs(
                kind, model,
                model_input(kind, name, raw_holdout, scalers, pre),
                device,
            )
            holdout_metrics, _ = metrics_from_probs(probs, y_holdout)
            per_fold_holdout[name].append(holdout_metrics)

            probability_sum[name] = probability_sum.get(name, 0.0) + probs
            n_models_seen[name] = n_models_seen.get(name, 0) + 1

            report(f"    {name:<22} own fold Top-1={cv_metrics['top1']:.4f} | "
                   f"holdout Top-1={holdout_metrics['top1']:.4f}")
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

    available = [m for m in models if per_fold_holdout[m]]
    if not available:
        raise SystemExit("No trained models found. Run stage 9 first.")

    ensemble: dict[str, dict] = {}
    for name in available:
        probs = probability_sum[name] / n_models_seen[name]
        metrics, preds = metrics_from_probs(probs, y_holdout)
        ensemble[name] = {"metrics": metrics, "probs": probs, "preds": preds}

    report("\n" + "=" * 78)
    report("  CROSS-VALIDATION vs HOLDOUT")
    report("=" * 78)
    report(f"  {'Model':<22}{'CV Top-1':>18}{'Holdout (per fold)':>22}"
           f"{'delta':>9}{'Ensemble':>11}")
    report("  " + "-" * 74)

    summary: dict[str, dict] = {}
    for name in ordered_models(available):
        cv_mean, cv_std = mean_std([m["top1"] for m in cross_validation[name]])
        ho_mean, ho_std = mean_std([m["top1"] for m in per_fold_holdout[name]])
        delta = ho_mean - cv_mean
        flag = "  <-- check" if abs(delta) > DIVERGENCE_THRESHOLD else ""
        report(f"  {name:<22}{cv_mean:>10.4f}+/-{cv_std:<6.4f}"
               f"{ho_mean:>13.4f}+/-{ho_std:<6.4f}"
               f"{delta:>+9.4f}{ensemble[name]['metrics']['top1']:>11.4f}{flag}")
        summary[name] = {
            "cv": {"top1_mean": cv_mean, "top1_std": cv_std,
                   "per_fold": cross_validation[name]},
            "holdout_per_fold": {"top1_mean": ho_mean, "top1_std": ho_std,
                                 "per_fold": per_fold_holdout[name]},
            "holdout_ensemble": ensemble[name]["metrics"],
            "delta_top1": delta,
        }

    report("\n  Top-3 / Top-5 on the holdout (ensemble)")
    for name in ordered_models(available):
        metrics = ensemble[name]["metrics"]
        report(f"    {name:<22} T1={metrics['top1']:.4f}  "
               f"T3={metrics['top3']:.4f}  T5={metrics['top5']:.4f}")

    deltas = [summary[m]["delta_top1"] for m in available]
    report("\n" + "=" * 78)
    signs = {np.sign(d) for d in deltas if d != 0}
    if all(abs(d) <= DIVERGENCE_THRESHOLD for d in deltas):
        report(f"  Every model agrees within {DIVERGENCE_THRESHOLD:.0%}. The "
               "cross-validation estimates are not inflated by repeated use of "
               "that pool.")
    else:
        report(f"  At least one model differs by more than "
               f"{DIVERGENCE_THRESHOLD:.0%}; report it as a finding.")
    if len(signs) > 1:
        report("  The differences are mixed in sign, which indicates sampling "
               "noise rather than systematic optimism.")
    elif signs:
        direction = "below" if deltas[0] < 0 else "above"
        report(f"  All differences point the same way (holdout {direction} CV), "
               "which is worth examining rather than dismissing.")
    report("=" * 78)

    with open(args.out_dir / "holdout_results.pkl", "wb") as fh:
        pickle.dump(summary, fh, protocol=pickle.HIGHEST_PROTOCOL)
    pd.DataFrame([
        {"model": m,
         "cv_top1": summary[m]["cv"]["top1_mean"],
         "holdout_per_fold_top1": summary[m]["holdout_per_fold"]["top1_mean"],
         "holdout_ensemble_top1": summary[m]["holdout_ensemble"]["top1"],
         "delta": summary[m]["delta_top1"]}
        for m in ordered_models(available)
    ]).to_csv(args.out_dir / "holdout_summary.csv", index=False)
    report.save(args.out_dir / "holdout_summary.txt")


if __name__ == "__main__":
    main()
