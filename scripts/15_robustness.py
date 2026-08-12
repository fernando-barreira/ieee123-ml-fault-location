"""Stage 15 — operational robustness: re-testing without retraining.

Evaluates the already-trained models under three deployment conditions they
were not trained for.  Nothing is refitted; only the test inputs change, so any
degradation is attributable to the perturbation alone.

**Losing a PMU.**  Telemetry drops.  The lost sensor's columns are replaced by
the training-set mean — the imputation a deployed system would use, and one
that keeps the input inside the distribution the model was fitted on. Crucially,
the three aggregate features are then **re-derived from the surviving PMUs**:
after an outage, "which sensor saw the largest change" and "mean superimposed
current" are computed over the sensors that remain. Leaving them reporting the
dead PMU would describe an outage no real system experiences. The per-PMU drop
in accuracy is a direct measure of how critical each sensor is.

**Measurement error.**  A complex multiplicative perturbation is applied to the
twelve phasors a PMU actually measures, and **every derived feature is then
recomputed from the perturbed phasors** with the same code that built the
training data.  The perturbation magnitude is the *total vector error* of
IEC/IEEE 60255-118-1, so the sweep is expressed in the units by which PMU accuracy
is specified rather than in an abstract noise level.

Recomputing matters, and it is not a refinement: perturbing the derived columns
directly would describe a measurement no instrument can produce, and would
understate the error where it matters most.  The superimposed quantities are
differences of large numbers — |V| is around 2400 V while |dV| is around 100 V —
so a given relative error on the measured voltage appears roughly |V|/|dV|
times larger on the superimposed voltage.  Under the direct-perturbation model
that amplification is simply absent.

Three correlation structures are swept, at matched total variance so the
comparison isolates structure rather than magnitude:

* *independent* — one draw per phasor; errors average out across channels.
* *correlated within a PMU* — half the variance shared across one instrument's
  channels, modelling a reference or timing drift.
* *fixed bias per PMU* — one offset per instrument, constant across all
  samples: a calibration error, which does not average out over repeated
  events and is therefore the hardest case.

**Isolated instrument chains.**  The same error applied to the voltage
transformers alone, then to the current transformers alone.  Apparent power is
not perturbed independently because it has no independent error source — it is
computed from both chains.

Each model is additionally evaluated at exactly the noise level it was trained
with (its Optuna-selected augmentation), which fixes the in-distribution point
on the curve and makes the slope immediately above it readable.

Usage
-----
.. code-block:: bash

    python scripts/15_robustness.py
    python scripts/15_robustness.py --models "Deep Residual MLP,Random Forest"
    python scripts/15_robustness.py --missing-mode zero
"""

from __future__ import annotations

import os
import warnings

# Must be set BEFORE scikit-learn or joblib is imported, and must live in the
# environment rather than only in this process's filter list: RandomForest's
# predict_proba fans out to joblib workers, and a filter registered at runtime
# does not reach them. loky propagates the parent environment to its children,
# so PYTHONWARNINGS is what actually silences them.
#
# The notice being suppressed is scikit-learn's `sklearn.utils.parallel.delayed`
# advisory. It is harmless, but it fires once per parallel batch: in an earlier
# run it buried 86 lines of results under 6 500 lines of noise, and output
# nobody can read is output nobody checks.
os.environ.setdefault("PYTHONWARNINGS", "ignore::UserWarning")
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import argparse
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config
from src.analysis import (
    DISPLAY_TO_KEY,
    Report,
    apply_tve,
    build_feature_matrix,
    global_feature_index,
    load_fold_indices,
    load_fold_model,
    load_preprocessing,
    load_scalers,
    mean_std,
    model_input,
    pmu_column_indices,
    predict_probs,
    raw_measurement_frame,
    rebuild_features,
    recompute_aggregates,
)
from src.baselines import infer_pmus_from_columns


# ═══════════════════════════════════════════════════════════════════════════
#  PERTURBATION
# ═══════════════════════════════════════════════════════════════════════════
def perturbed_matrix(
    base_frame, pmus, sigma, seed, mode, quantities, pre,
) -> np.ndarray:
    """Apply a TVE perturbation to the phasors and re-derive every feature.

    The random seed is passed rather than a generator so that a given
    ``(fold, mode, sigma, run)`` produces the *same* perturbation for every
    model.  That makes the comparison paired: differences between models are
    differences in their response, not in the noise they happened to receive.
    """
    if sigma <= 0:
        return rebuild_features(base_frame, pmus, pre)

    frame = apply_tve(
        base_frame, pmus, config.PHASES, sigma,
        np.random.default_rng(seed), mode=mode, quantities=quantities,
    )
    return rebuild_features(frame, pmus, pre)


# ═══════════════════════════════════════════════════════════════════════════
#  EVALUATION
# ═══════════════════════════════════════════════════════════════════════════
def evaluate(probs, y, hops=None) -> dict[str, float]:
    preds = probs.argmax(axis=1)
    top3 = np.argpartition(-probs, kth=2, axis=1)[:, :3]
    out = {
        "top1": float((preds == y).mean()),
        "top3": float(np.any(top3 == y[:, None], axis=1).mean()),
        "mean_confidence": float(probs.max(axis=1).mean()),
    }
    if hops is not None:
        out["mean_hop_error"] = float(hops[preds, y].mean())
    return out


def run_once(kind, model, name, raw, y, scalers, pre, device, hops):
    probs = predict_probs(
        kind, model, model_input(kind, name, raw, scalers, pre), device
    )
    return evaluate(probs, y, hops)


def aggregate_runs(runs: list[dict]) -> dict[str, tuple[float, float]]:
    return {
        key: (float(np.mean([r[key] for r in runs])),
              float(np.std([r[key] for r in runs])))
        for key in runs[0]
    }


def scenario_missing_pmu(kind, model, name, raw_test, y_test, pmus,
                         columns_per_pmu, index, train_mean, scalers, pre,
                         device, hops, mode):
    """Ablate each PMU in turn, recomputing the aggregates from the survivors."""
    out = {"_baseline": run_once(kind, model, name, raw_test, y_test,
                                 scalers, pre, device, hops)}
    for position, pmu in enumerate(pmus):
        perturbed = raw_test.copy()
        columns = columns_per_pmu.get(pmu, [])
        if columns:
            if mode == "zero":
                perturbed[:, columns] = 0.0
            else:
                perturbed[:, columns] = train_mean[columns]
        recompute_aggregates(perturbed, position, index)
        out[pmu] = run_once(kind, model, name, perturbed, y_test,
                            scalers, pre, device, hops)
    return out


def scenario_noise(kind, model, name, base_frame, pmus, y_test, scalers, pre,
                   device, hops, sigmas, mode, n_runs, seed_base):
    """Sweep the TVE levels for one correlation structure."""
    out = {}
    for sigma in sigmas:
        runs = []
        for run in range(n_runs if sigma > 0 else 1):
            raw = perturbed_matrix(
                base_frame, pmus, sigma,
                seed_base + int(sigma * 1e6) * 100 + run,
                mode, config.MEASURED_QUANTITIES, pre,
            )
            runs.append(run_once(kind, model, name, raw, y_test,
                                 scalers, pre, device, hops))
        out[sigma] = aggregate_runs(runs)
    return out


def scenario_isolated_chains(kind, model, name, base_frame, pmus, y_test,
                             scalers, pre, device, hops, sigma, n_runs,
                             seed_base):
    """The same error applied to one instrument chain at a time."""
    out = {}
    for quantity in config.MEASURED_QUANTITIES:
        runs = []
        for run in range(n_runs):
            raw = perturbed_matrix(
                base_frame, pmus, sigma,
                seed_base + hash(quantity) % 1000 * 100 + run,
                "independent", (quantity,), pre,
            )
            runs.append(run_once(kind, model, name, raw, y_test,
                                 scalers, pre, device, hops))
        out[quantity] = aggregate_runs(runs)
    return out


# ═══════════════════════════════════════════════════════════════════════════
#  DERIVED QUANTITIES
# ═══════════════════════════════════════════════════════════════════════════
def training_sigma(name: str, best_hp: dict) -> tuple[float, str]:
    """Noise level a model was trained with, read from its hyper-parameters.

    Tree ensembles are trained without augmentation, so their in-distribution
    point coincides with sigma = 0.
    """
    key = DISPLAY_TO_KEY.get(name, name)
    hp = best_hp.get(key)
    if isinstance(hp, dict) and hp.get("noise_std") is not None:
        return float(hp["noise_std"]), f"best_hp[{key}][noise_std]"
    return 0.0, "no augmentation"


def criticality(missing: dict[str, dict[int, dict]], pmus) -> dict[str, dict]:
    """Accuracy lost per PMU outage, averaged over folds."""
    out: dict[str, dict] = {}
    for name, per_fold in missing.items():
        if not per_fold:
            continue
        out[name] = {}
        for pmu in pmus:
            drops = [d["_baseline"]["top1"] - d[pmu]["top1"]
                     for d in per_fold.values() if pmu in d]
            if drops:
                out[name][pmu] = (float(np.mean(drops)), float(np.std(drops)), drops)
    return out


def format_tve(sigma: float) -> str:
    """TVE as a percentage, with enough precision to stay unambiguous.

    A fixed one-decimal format collapses 0.05 % and 0.1 % into the same column
    header on the refined grid, which is exactly the range that matters.
    """
    return f"{sigma * 100:g}%"


def half_accuracy_tve(sigmas, accuracy) -> float:
    """TVE at which Top-1 falls to half its error-free value.

    Reported alongside the linear slope because the degradation is not linear:
    propagated through the superimposed quantities, measurement error produces
    a threshold rather than a ramp, and a straight-line fit through a cliff has
    a slope that depends mostly on where the grid points happen to fall. This
    threshold is the scale-free summary — "how good must the instrument be for
    this model to keep half its accuracy" — and it is directly comparable
    against the 1 % compliance limit of IEEE Std C37.118.

    Returned by linear interpolation between bracketing points; ``nan`` if the
    sweep never reaches half the initial accuracy.
    """
    accuracy = np.asarray(accuracy, dtype=float)
    target = accuracy[0] / 2.0
    for i in range(len(accuracy) - 1):
        if accuracy[i] >= target >= accuracy[i + 1]:
            span = accuracy[i] - accuracy[i + 1]
            if span <= 0:
                return float(sigmas[i])
            fraction = (accuracy[i] - target) / span
            return float(sigmas[i] + fraction * (sigmas[i + 1] - sigmas[i]))
    return float("nan")


def degradation_slope(noise: dict[str, dict[int, dict]], sigmas) -> dict[str, dict]:
    """Linear slope of Top-1 against TVE, the AUC, and the half-accuracy TVE.

    The slope and the R-squared are reported together on purpose: a low
    R-squared is the signal that the degradation is a threshold rather than a
    ramp, and that the slope should not be quoted on its own.
    """
    from scipy import stats

    out: dict[str, dict] = {}
    for name, per_fold in noise.items():
        if not per_fold:
            continue
        slopes, r_squared, areas, halves = [], [], [], []
        for record in per_fold.values():
            accuracy = np.array([record[s]["top1"][0] for s in sigmas])
            fit = stats.linregress(np.array(sigmas), accuracy)
            slopes.append(fit.slope)
            r_squared.append(fit.rvalue ** 2)
            areas.append(float(np.trapezoid(accuracy, sigmas)))
            half = half_accuracy_tve(sigmas, accuracy)
            if np.isfinite(half):
                halves.append(half)
        out[name] = {
            "slope": (float(np.mean(slopes)), float(np.std(slopes))),
            "r2": (float(np.mean(r_squared)), float(np.std(r_squared))),
            "auc": (float(np.mean(areas)), float(np.std(areas))),
            # Empty when no fold ever loses half its accuracy over the sweep —
            # the good case, and one that must not raise.
            "tve_half": ((float(np.mean(halves)), float(np.std(halves)))
                         if halves else (float("nan"), float("nan"))),
        }
    return out


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--dataset", type=config.resolve_path,
                        default=config.features_csv(5))
    parser.add_argument("--cv-dir", type=config.resolve_path, default=config.CV_DIR)
    parser.add_argument("--out-dir", type=config.resolve_path,
                        default=config.ANALYSIS_DIR / "robustness")
    parser.add_argument(
        "--models", type=str,
        default="Deep Residual MLP,MLP Baseline,Random Forest,LightGBM",
    )
    parser.add_argument("--missing-mode", choices=("mean", "zero"),
                        default=config.MISSING_PMU_MODE)
    parser.add_argument("--n-runs", type=int, default=config.N_NOISE_RUNS)
    parser.add_argument("--seed", type=int, default=config.SEED)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config.ensure_dir(args.out_dir)
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    report = Report()

    report("=" * 78)
    report("  OPERATIONAL ROBUSTNESS (no retraining)")
    report("=" * 78)
    report(f"  device: {device} | missing-PMU mode: '{args.missing_mode}' "
           f"| noise runs: {args.n_runs}")

    pre = load_preprocessing(args.cv_dir)
    fold_indices = load_fold_indices(args.cv_dir)
    groups = pre["groups"]
    feature_names = pre["feature_names"]
    hops = pre["hop_matrix"]

    df = pd.read_csv(args.dataset)
    pmus = infer_pmus_from_columns(df)
    if len(pmus) != pre["n_pmus"]:
        raise SystemExit(
            f"Dataset has {len(pmus)} PMUs but stage 9 recorded "
            f"{pre['n_pmus']}. The results and the dataset disagree."
        )
    report(f"  PMUs: {pmus}")

    raw_all = build_feature_matrix(df, pre)
    y_all = pre["target_encoder"].transform(df[config.TARGET_COL].to_numpy())

    columns_per_pmu = pmu_column_indices(feature_names, pmus)
    aggregate_index = global_feature_index(feature_names, pmus, config.PHASES)

    # Errors are applied to the measured phasors and every feature is then
    # re-derived, so the perturbation works on this reduced frame rather than
    # on the assembled matrix.
    base_frame_all = raw_measurement_frame(df, config.TARGET_COL)
    n_measured = len([c for c in base_frame_all.columns if c != config.TARGET_COL])
    report(f"    measured phasor columns: {n_measured} "
           f"({n_measured // max(len(pmus), 1)} per PMU)")
    for pmu in pmus:
        report(f"    PMU {pmu:<5} {len(columns_per_pmu[pmu]):>4} derived features")
    report(f"    aggregates: severity={aggregate_index['severity'] is not None}, "
           f"one-hot dI={sorted(aggregate_index['onehot_dI'])}, "
           f"dV={sorted(aggregate_index['onehot_dV'])}")

    # The rebuilt features must reproduce the stored matrix exactly when no
    # error is applied; otherwise every degradation figure would be measured
    # against the wrong reference.
    check = rebuild_features(base_frame_all.iloc[:256], pmus, pre)
    if not np.allclose(check, raw_all[:256], rtol=1e-4, atol=1e-4):
        worst = int(np.abs(check - raw_all[:256]).argmax())
        raise SystemExit(
            "Re-deriving the features from the stored phasors does not "
            "reproduce the dataset; the noise reference would be wrong. "
            f"Largest discrepancy at column "
            f"{feature_names[worst % len(feature_names)]!r}."
        )
    report("    sanity: features re-derived at sigma=0 reproduce the dataset")

    sigmas = list(config.NOISE_SIGMAS)
    sigma_train = {m: training_sigma(m, pre["best_hp"]) for m in models}
    report("    training noise: " + ", ".join(
        f"{m}={s * 100:.3f}% [{src}]" for m, (s, src) in sigma_train.items()))

    missing: dict[str, dict] = {m: {} for m in models}
    noise_ind: dict[str, dict] = {m: {} for m in models}
    noise_corr: dict[str, dict] = {m: {} for m in models}
    noise_bias: dict[str, dict] = {m: {} for m in models}
    isolated: dict[str, dict] = {m: {} for m in models}
    at_train_sigma: dict[str, dict] = {m: {} for m in models}

    start = time.time()
    for fold in range(pre["n_folds"]):
        if not (Path(args.cv_dir) / f"fold_{fold}" / "scalers.pkl").exists():
            report(f"\n  fold {fold}: not trained, skipped")
            continue
        scalers = load_scalers(args.cv_dir, fold)
        train_idx = fold_indices[fold]["train"]
        test_idx = fold_indices[fold]["test"]
        raw_test, y_test = raw_all[test_idx], y_all[test_idx]
        base_frame = base_frame_all.iloc[test_idx].reset_index(drop=True)
        # Imputation uses the mean of the data the model was actually fitted
        # on, which excludes the internal validation slice.
        train_mean = raw_all[train_idx].mean(axis=0).astype(np.float32)
        report(f"\n  -- fold {fold} | n_test={len(test_idx)} --")

        for position, name in enumerate(models):
            try:
                kind, model = load_fold_model(name, args.cv_dir, fold, pre, device)
            except (FileNotFoundError, KeyError) as exc:
                report(f"    {name}: not available ({exc}), skipped")
                continue
            seed_base = args.seed + fold * 100_000 + position * 10_000
            t0 = time.time()

            missing[name][fold] = scenario_missing_pmu(
                kind, model, name, raw_test, y_test, pmus, columns_per_pmu,
                aggregate_index, train_mean, scalers, pre, device, hops,
                args.missing_mode,
            )
            for label, store, mode in (
                ("independent", noise_ind, "independent"),
                ("correlated", noise_corr, "correlated"),
                ("bias", noise_bias, "bias"),
            ):
                store[name][fold] = scenario_noise(
                    kind, model, name, base_frame, pmus, y_test, scalers, pre,
                    device, hops, sigmas, mode, args.n_runs,
                    seed_base + hash(label) % 977,
                )
            isolated[name][fold] = scenario_isolated_chains(
                kind, model, name, base_frame, pmus, y_test, scalers, pre,
                device, hops, config.ISOLATED_GROUP_SIGMA, args.n_runs,
                seed_base + 7,
            )
            sigma = sigma_train[name][0]
            at_train_sigma[name][fold] = aggregate_runs([
                run_once(kind, model, name,
                         perturbed_matrix(base_frame, pmus, sigma,
                                          seed_base + 13 + run, "independent",
                                          config.MEASURED_QUANTITIES, pre),
                         y_test, scalers, pre, device, hops)
                for run in range(args.n_runs if sigma > 0 else 1)
            ])

            report(f"    {name}: done ({time.time() - t0:.0f}s)")
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

    # ── report ────────────────────────────────────────────────────────────
    report("\n" + "=" * 78)
    report(f"  [1] PMU CRITICALITY — Top-1 lost when a sensor is unavailable "
           f"(mode '{args.missing_mode}')")
    report("=" * 78)
    critical = criticality(missing, pmus)
    for name, entries in critical.items():
        baseline = mean_std([d["_baseline"]["top1"] for d in missing[name].values()])
        report(f"\n  {name}  (intact Top-1 = {baseline[0]:.4f})")
        for pmu, (mean, std, _) in sorted(entries.items(), key=lambda kv: -kv[1][0]):
            report(f"    PMU {pmu:<6} -{mean:.4f} +/- {std:.4f}")

    report("\n" + "=" * 78)
    report("  [2] MEASUREMENT ERROR — Top-1 e Top-3 vs total vector error (IEEE C37.118)")
    report("=" * 78)
    for label, results in (("independent", noise_ind),
                           ("correlated within PMU", noise_corr),
                           ("fixed bias per PMU", noise_bias)):
        for metric_name, metric_key in (("Top-1", "top1"), ("Top-3", "top3")):
            report(f"\n  {label} ({metric_name})")
            report(f"    {'Model':<22}" + "".join(
                f"{format_tve(s):>12}" for s in sigmas))
            report("    " + "-" * 68)
            for name in models:
                if not results[name]:
                    continue
                row = f"    {name:<22}"
                for sigma in sigmas:
                    mean, _ = mean_std([d[sigma][metric_key][0] for d in results[name].values()])
                    row += f"{mean:>12.4f}"
                report(row)

    report("\n  Degradation slope (Top-1 per unit TVE; less negative = more robust)")
    for label, results in (("independent", noise_ind),
                           ("correlated", noise_corr),
                           ("bias", noise_bias)):
        slopes = degradation_slope(results, sigmas)
        report(f"\n    {label}")
        for name, entry in slopes.items():
            half = entry["tve_half"][0]
            half_text = f"{half * 100:.3f}%" if np.isfinite(half) else "  > max"
            report(f"      {name:<22} slope={entry['slope'][0]:>8.3f} "
                   f"+/-{entry['slope'][1]:<6.3f}  R2={entry['r2'][0]:.3f}  "
                   f"AUC={entry['auc'][0]:.4f}  TVE(50%)={half_text}")
        if slopes and min(e["r2"][0] for e in slopes.values()) < 0.7:
            report("      (low R2: the degradation is a threshold, not a ramp; "
                   "quote TVE(50%) rather than the slope)")

    report("\n  Top-1 at each model's own training noise level")
    for name in models:
        if not at_train_sigma[name]:
            continue
        sigma, source = sigma_train[name]
        mean, std = mean_std([d["top1"][0] for d in at_train_sigma[name].values()])
        report(f"    {name:<22} sigma={sigma * 100:.3f}%  "
               f"Top-1={mean:.4f} +/- {std:.4f}   [{source}]")

    report("\n" + "=" * 78)
    report(f"  [3] ISOLATED INSTRUMENT CHAIN (TVE = "
           f"{config.ISOLATED_GROUP_SIGMA:.1%} on one chain only)")
    report("=" * 78)
    report(f"  {'Model':<22}" + "".join(
        f"{q + ' transformers':>22}" for q in config.MEASURED_QUANTITIES))
    report("  " + "-" * 66)
    for name in models:
        if not isolated[name]:
            continue
        row = f"  {name:<22}"
        for quantity in config.MEASURED_QUANTITIES:
            values = [d[quantity]["top1"][0] for d in isolated[name].values()
                      if quantity in d]
            mean, _ = mean_std(values)
            row += f"{mean:>22.4f}"
        report(row)

    for filename, payload in (
        ("missing_pmu.pkl", missing),
        ("noise_independent.pkl", noise_ind),
        ("noise_correlated.pkl", noise_corr),
        ("noise_bias.pkl", noise_bias),
        ("noise_isolated_chains.pkl", isolated),
        ("noise_at_training_sigma.pkl",
         {"results": at_train_sigma,
          "sigma_train": {m: s for m, (s, _) in sigma_train.items()}}),
        ("criticality.pkl", critical),
    ):
        with open(args.out_dir / filename, "wb") as fh:
            pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)

    report(f"\n  elapsed: {time.time() - start:.0f}s")
    report.save(args.out_dir / "robustness.txt")


if __name__ == "__main__":
    main()