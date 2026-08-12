"""Stage 9 — 5-fold cross-validated training of every model.

Produces the headline numbers of the paper.  Each of the five folds trains all
five models with the hyper-parameters fixed by stage 8, and evaluates them on
that fold's held-out test set.

Protocol
--------
* **Frozen folds.**  The cross-validation pool (~75 000 samples) and its five
  stratified folds come from ``splits.pkl`` and are disjoint from the pools
  used for the placement search and the hyper-parameter search.  Fold indices
  are *local* to that pool.
* **Hyper-parameters are fixed**, never re-tuned here.  The 15 % internal
  validation split inside each fold decides only *when to stop*; it selects
  nothing.
* **Scalers are refitted per fold** on that fold's training portion.  Fitting
  once on the whole pool would leak the test distribution into the transform.
* **Trees receive the unscaled matrix**, matching stage 8.
* **Hop distance is recorded per prediction**, using the topology graph from
  stage 1.  For a fault locator this is the operationally meaningful error
  measure: predicting an adjacent bus sends the crew to the right span, while
  predicting a bus on another lateral sends them to the wrong place entirely.

Outputs, under ``results/cv/``:

===============================  =============================================
``preprocessing_global.pkl``     Encoder, feature names, groups, hop matrix.
``fold_indices.pkl``             Train/validation/test indices per fold.
``fold_<k>/``                    Fitted models, scalers, ``fold_results.pkl``.
``summary.csv`` / ``summary.pkl``  Mean and standard deviation per model.
===============================  =============================================

``fold_indices.pkl`` records indices **into the full dataset**, not into the
cross-validation pool, so any later stage can index the dataset directly.

Results are keyed by the models' display names (``"Deep Residual MLP"``,
``"Random Forest"``, ...), matching the classical baselines of stage 10 and the
paper's tables, so every downstream analysis sees one naming convention.

``fold_results.pkl`` is rewritten after **every** model rather than once at the
end of the fold, and re-running a fold skips models already recorded in it.
A single model here can take an hour; a failure in a later one must not discard
the earlier ones.  Delete the file to force a fold to be recomputed from
scratch.

Usage
-----
.. code-block:: bash

    python scripts/09_cross_validate.py
    python scripts/09_cross_validate.py --resume-from 2
    python scripts/09_cross_validate.py --skip TabNet
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
import warnings
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config
from src.graph import load_graph
from src.models import build_model, class_weights
from src.scaling import prepare_matrix, scale_splits
from src.splits import load_splits
from src.training import (
    make_loader,
    metrics_from_probs,
    nn_latency,
    predict_probs,
    require,
    sklearn_latency,
    train_neural,
)

warnings.filterwarnings("ignore")


# ═══════════════════════════════════════════════════════════════════════════
#  HOP DISTANCE
# ═══════════════════════════════════════════════════════════════════════════
def hop_matrix(graph, classes: list[str]) -> np.ndarray:
    """Pairwise hop distance between every pair of classes.

    Indexed by encoder position, so ``M[pred, true]`` is the hop distance of a
    prediction.  Distances come from breadth-first search over the graph, not
    from bus numbering: bus labels in this model are not ordered along the
    feeder, so ``|97 - 197|`` says nothing about how far apart they are.

    Raises
    ------
    ValueError
        If any class is missing from the graph or unreachable.  A sentinel
        value was previously used instead, which quietly poisoned the mean hop
        distance reported in the paper.
    """
    missing = [c for c in classes if c not in graph]
    if missing:
        raise ValueError(
            f"{len(missing)} class(es) absent from the topology graph, first "
            f"few: {missing[:5]}. The label space and the graph must agree."
        )

    n = len(classes)
    matrix = np.zeros((n, n), dtype=np.int32)
    position = {bus: i for i, bus in enumerate(classes)}
    for bus, i in position.items():
        import networkx as nx

        lengths = nx.single_source_shortest_path_length(graph, bus)
        for target, j in position.items():
            if target not in lengths:
                raise ValueError(f"No path between {bus!r} and {target!r}.")
            matrix[i, j] = lengths[target]
    return matrix


# ═══════════════════════════════════════════════════════════════════════════
#  TABNET
# ═══════════════════════════════════════════════════════════════════════════
def tabnet_history(model) -> list[float] | None:
    """Validation-accuracy curve from a fitted ``TabNetClassifier``.

    ``model.history`` is pytorch-tabnet's ``History`` **callback object**, not a
    dictionary: it implements ``__getitem__`` but has no ``.get``, so the
    dictionary idiom raises ``AttributeError`` — and it does so only after the
    model has finished training, which is the worst possible moment.

    The metric key also depends on how ``eval_name`` was passed: ``val_accuracy``
    with an explicit name, ``val_0_accuracy`` without one.  Both are tried, and
    the underlying ``.history`` dict is used when the object exposes it.

    Returns ``None`` rather than raising if no curve is available: the curve is
    diagnostic output for the training-dynamics figure, and losing it must never
    cost a fold that has already been trained.
    """
    history = getattr(model, "history", None)
    if history is None:
        return None

    store = getattr(history, "history", history)
    for key in ("val_accuracy", "val_0_accuracy", "valid_accuracy"):
        try:
            curve = store[key]
        except (KeyError, TypeError, AttributeError):
            continue
        if curve is not None and len(curve):
            return [float(v) for v in curve]

    print("     note: no TabNet validation curve recorded (metric key not found)")
    return None


# ═══════════════════════════════════════════════════════════════════════════
#  ONE FOLD
# ═══════════════════════════════════════════════════════════════════════════
def run_fold(
    fold: int,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    matrix: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    best_hp: dict,
    n_classes: int,
    hops: np.ndarray,
    out_dir: Path,
    skip: set[str],
    device: torch.device,
) -> dict:
    """Train and evaluate every model on one fold."""
    print(f"\n{'=' * 70}\n  FOLD {fold}\n{'=' * 70}")
    print(f"  train {len(train_idx):,} | val {len(val_idx):,} | test {len(test_idx):,}")

    fold_dir = config.ensure_dir(out_dir / f"fold_{fold}")

    x_train_raw = matrix[train_idx]
    x_val_raw = matrix[val_idx]
    x_test_raw = matrix[test_idx]
    y_train, y_val, y_test = y[train_idx], y[val_idx], y[test_idx]

    x_train, (x_val, x_test), scalers, noise_columns = scale_splits(
        x_train_raw, [x_val_raw, x_test_raw], feature_names
    )
    # The scalers are stored per fold because they are fitted per fold; the
    # downstream robustness and importance analyses re-normalise perturbed data
    # and must use the very transform the model was trained under.
    with open(fold_dir / "scalers.pkl", "wb") as fh:
        pickle.dump(scalers, fh, protocol=pickle.HIGHEST_PROTOCOL)

    # Re-seeded per fold so each fold is independently reproducible; without
    # this, resuming from fold 2 would give different results than a full run.
    torch.manual_seed(config.SEED + fold)
    np.random.seed(config.SEED + fold)

    noise_tensor = torch.as_tensor(noise_columns, dtype=torch.long)
    results_path = fold_dir / "fold_results.pkl"

    # Resume support: a previous attempt may have completed some models before
    # failing. Anything already recorded is kept, so a crash in the last model
    # does not cost the hours spent on the earlier ones.
    results: dict[str, dict] = {}
    if results_path.exists():
        try:
            with open(results_path, "rb") as fh:
                results = pickle.load(fh)
            if results:
                print(f"  resuming: {sorted(results)} already recorded")
        except Exception as exc:  # noqa: BLE001
            print(f"  could not read {results_path.name} ({exc}); starting over")
            results = {}

    def record(name: str, probs: np.ndarray, truth: np.ndarray, **extra) -> None:
        # Keyed by DISPLAY name, not by the internal key. The classical
        # baselines of stage 10 and every analysis downstream identify models
        # this way, and so do the paper's tables; two naming conventions for the
        # same object is how an analysis silently finds nothing.
        display = config.MODEL_DISPLAY_NAMES.get(name, name)
        metrics, preds = metrics_from_probs(probs, truth)
        hop_dist = hops[preds, truth]
        results[display] = dict(
            probs=probs, preds=preds, y_true=truth, metrics=metrics,
            hop_dist=hop_dist, **extra
        )
        print(f"     T1={metrics['top1']:.4f} T3={metrics['top3']:.4f} "
              f"T5={metrics['top5']:.4f} | hop mean={hop_dist.mean():.2f} "
              f"| exact+adjacent={100 * (hop_dist <= 1).mean():.1f}%")
        # Written immediately: training one model here costs up to an hour, and
        # a failure in a later model must not discard it.
        with open(results_path, "wb") as fh:
            pickle.dump(results, fh, protocol=pickle.HIGHEST_PROTOCOL)

    def already_done(name: str) -> bool:
        display = config.MODEL_DISPLAY_NAMES.get(name, name)
        if display in results:
            print(f"  -- {display}: already recorded, skipping")
            return True
        return False

    # ── MLP-family models ────────────────────────────────────────────────
    for name in ("MLP", "DeepResidualMLP"):
        if name in skip or already_done(name):
            continue
        torch.manual_seed(config.SEED + fold)
        hp = best_hp[name]

        weights = torch.as_tensor(
            class_weights(y_train, n_classes, hp["class_weight_mode"]),
            dtype=torch.float32, device=device,
        )
        batch = hp["batch_size"]
        train_loader = make_loader(
            x_train, y_train, batch, device, noise=hp["noise_std"],
            noise_columns=noise_tensor, train=True, shuffle=True,
            num_workers=config.DATALOADER_WORKERS,
        )
        val_loader = make_loader(x_val, y_val, batch, device,
                                 num_workers=config.DATALOADER_WORKERS)
        test_loader = make_loader(x_test, y_test, batch, device,
                                  num_workers=config.DATALOADER_WORKERS)

        model = build_model(name, hp, x_train.shape[1], n_classes)
        outcome = train_neural(
            model, train_loader, val_loader,
            n_classes=n_classes, hp=hp, device=device,
            epochs=config.CV_EPOCHS, patience=config.CV_PATIENCE,
            warmup_epochs=config.CV_WARMUP_EPOCHS, weights=weights,
            log_every=20, label=config.MODEL_DISPLAY_NAMES[name],
        )
        torch.save(model.state_dict(), fold_dir / f"{name}.pt")

        probs, truth = predict_probs(model, test_loader, device)
        record(
            name, probs, truth,
            history=outcome["history"], importance=None,
            training_time=outcome["training_time"],
            inference_time_ms=nn_latency(model, x_test, device, config.N_INFERENCE_RUNS),
            n_params=outcome["n_params"],
        )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # ── Random forest ────────────────────────────────────────────────────
    if "RF" not in skip and not already_done("RF"):
        from sklearn.ensemble import RandomForestClassifier

        hp = best_hp["RF"]
        print(f"  -- Random Forest | trees={hp['n_estimators']} "
              f"leaf={hp['min_samples_leaf']}")
        t0 = time.time()
        forest = RandomForestClassifier(
            n_estimators=hp["n_estimators"], max_features=hp["max_features"],
            min_samples_leaf=hp["min_samples_leaf"], max_depth=hp["max_depth"],
            max_samples=hp["max_samples"], class_weight=hp["class_weight"],
            bootstrap=True, n_jobs=-1, random_state=config.SEED,
        )
        forest.fit(x_train_raw, y_train)
        elapsed = time.time() - t0
        with open(fold_dir / "RF.pkl", "wb") as fh:
            pickle.dump(forest, fh, protocol=pickle.HIGHEST_PROTOCOL)
        record(
            "RF", forest.predict_proba(x_test_raw), y_test,
            history=None, importance=forest.feature_importances_,
            training_time=elapsed,
            inference_time_ms=sklearn_latency(forest, x_test_raw, config.N_INFERENCE_RUNS),
            n_params=int(sum(t.tree_.node_count for t in forest.estimators_)),
        )
        del forest

    # ── LightGBM ─────────────────────────────────────────────────────────
    if "LightGBM" not in skip and not already_done("LightGBM"):
        import lightgbm as lgb

        hp = best_hp["LightGBM"]
        print(f"  -- LightGBM | leaves={hp['num_leaves']} lr={hp['learning_rate']:.4f}")
        t0 = time.time()
        booster = lgb.LGBMClassifier(
            objective="multiclassova", num_class=n_classes,
            n_estimators=config.LGB_N_ESTIMATORS,
            num_leaves=hp["num_leaves"], learning_rate=hp["learning_rate"],
            min_data_in_leaf=hp["min_data_in_leaf"],
            feature_fraction=hp["feature_fraction"],
            bagging_fraction=hp["bagging_fraction"],
            bagging_freq=config.LGB_BAGGING_FREQ,
            lambda_l1=hp["lambda_l1"], lambda_l2=hp["lambda_l2"],
            n_jobs=-1, random_state=config.SEED, verbosity=-1,
        )
        booster.fit(
            x_train_raw, y_train, eval_set=[(x_val_raw, y_val)],
            eval_metric="multi_error",
            callbacks=[lgb.early_stopping(config.LGB_EARLY_STOPPING_ROUNDS,
                                          verbose=False)],
        )
        elapsed = time.time() - t0
        print(f"     best iteration: {booster.best_iteration_}")
        with open(fold_dir / "LightGBM.pkl", "wb") as fh:
            pickle.dump(booster, fh, protocol=pickle.HIGHEST_PROTOCOL)
        record(
            "LightGBM", booster.predict_proba(x_test_raw), y_test,
            history=None, importance=booster.feature_importances_,
            training_time=elapsed,
            inference_time_ms=sklearn_latency(booster, x_test_raw, config.N_INFERENCE_RUNS),
            n_params=int(booster.booster_.num_trees()),
        )
        del booster

    # ── TabNet ───────────────────────────────────────────────────────────
    if "TabNet" not in skip and not already_done("TabNet"):
        from pytorch_tabnet.tab_model import TabNetClassifier

        torch.manual_seed(config.SEED + fold)
        hp = best_hp["TabNet"]
        print(f"  -- TabNet | n_d=n_a={hp['n_d']} steps={hp['n_steps']}")
        weights = class_weights(
            y_train, n_classes, hp.get("class_weight_mode", "sqrt")
        )
        t0 = time.time()
        tabnet = TabNetClassifier(
            n_d=hp["n_d"], n_a=hp["n_d"], n_steps=hp["n_steps"],
            gamma=hp["gamma"], lambda_sparse=hp["lambda_sparse"],
            optimizer_fn=torch.optim.AdamW,
            optimizer_params={"lr": hp["lr"], "weight_decay": 1e-5},
            scheduler_fn=torch.optim.lr_scheduler.StepLR,
            scheduler_params={"step_size": 20, "gamma": 0.7},
            mask_type=hp["mask_type"], seed=config.SEED,
            device_name=device.type, verbose=20,
        )
        tabnet.fit(
            X_train=x_train, y_train=y_train,
            eval_set=[(x_val, y_val)], eval_name=["val"], eval_metric=["accuracy"],
            max_epochs=config.CV_EPOCHS, patience=config.CV_PATIENCE,
            batch_size=hp["batch_size"],
            virtual_batch_size=hp["virtual_batch_size"],
            weights={i: float(w) for i, w in enumerate(weights)},
            drop_last=False,
        )
        elapsed = time.time() - t0
        tabnet.save_model(str(fold_dir / "TabNet"))

        history = tabnet_history(tabnet)
        record(
            "TabNet", tabnet.predict_proba(x_test), y_test,
            history={"val_acc": list(history)} if history else None,
            importance=tabnet.feature_importances_,
            training_time=elapsed,
            inference_time_ms=sklearn_latency(tabnet, x_test, config.N_INFERENCE_RUNS),
            n_params=sum(
                p.numel() for p in tabnet.network.parameters() if p.requires_grad
            ),
        )
        del tabnet
        if device.type == "cuda":
            torch.cuda.empty_cache()

    print(f"\n  fold_{fold}/fold_results.pkl holds {len(results)} model(s): "
          f"{sorted(results)}")
    return results


# ═══════════════════════════════════════════════════════════════════════════
#  AGGREGATION
# ═══════════════════════════════════════════════════════════════════════════
def aggregate(out_dir: Path, n_folds: int) -> pd.DataFrame:
    """Collect per-fold metrics into mean and standard deviation per model.

    The standard deviation across five folds is **descriptive**, not
    inferential.  Five values are far too few to support a significance test,
    and the folds are not independent — they share most of their training data.
    It indicates the spread of the estimate, nothing more.
    """
    collected: dict[str, dict[str, list[float]]] = {}

    for fold in range(n_folds):
        path = out_dir / f"fold_{fold}" / "fold_results.pkl"
        if not path.exists():
            print(f"  fold {fold} missing; excluded from the summary")
            continue
        with open(path, "rb") as fh:
            fold_results = pickle.load(fh)
        for name, entry in fold_results.items():
            bucket = collected.setdefault(
                name,
                {"top1": [], "top3": [], "top5": [], "hop_mean": [],
                 "hop_within_1": [], "training_time": [], "inference_time_ms": [],
                 "n_params": []},
            )
            bucket["top1"].append(entry["metrics"]["top1"])
            bucket["top3"].append(entry["metrics"]["top3"])
            bucket["top5"].append(entry["metrics"]["top5"])
            bucket["hop_mean"].append(float(entry["hop_dist"].mean()))
            bucket["hop_within_1"].append(float((entry["hop_dist"] <= 1).mean()))
            bucket["training_time"].append(entry["training_time"])
            bucket["inference_time_ms"].append(entry["inference_time_ms"])
            bucket["n_params"].append(float(entry.get("n_params", 0)))

    rows = []
    summary: dict[str, dict] = {}
    for name, bucket in collected.items():
        summary[name] = {k: (float(np.mean(v)), float(np.std(v)), v)
                         for k, v in bucket.items() if v}
        row = {"model": name, "n_folds": len(bucket["top1"])}
        for metric, values in bucket.items():
            row[f"{metric}_mean"] = float(np.mean(values))
            row[f"{metric}_std"] = float(np.std(values))
        rows.append(row)

    table = pd.DataFrame(rows).sort_values("top1_mean", ascending=False)
    with open(out_dir / "summary.pkl", "wb") as fh:
        pickle.dump(summary, fh, protocol=pickle.HIGHEST_PROTOCOL)
    table.to_csv(out_dir / "summary.csv", index=False)

    print(f"\n{'=' * 78}")
    print(f"  {'Model':<20}{'Top-1':>16}{'Top-3':>16}{'Top-5':>16}")
    print(f"  {'-' * 68}")
    for _, r in table.iterrows():
        print(f"  {r['model']:<20}"
              f"{r['top1_mean']:>9.2%} ±{r['top1_std']:>5.2%}"
              f"{r['top3_mean']:>9.2%} ±{r['top3_std']:>5.2%}"
              f"{r['top5_mean']:>9.2%} ±{r['top5_std']:>5.2%}")
    print(f"{'=' * 78}")
    print(f"  {'Model':<20}{'Hop mean':>14}{'<=1 hop':>14}{'Infer (ms)':>16}")
    print(f"  {'-' * 68}")
    for _, r in table.iterrows():
        print(f"  {r['model']:<20}"
              f"{r['hop_mean_mean']:>9.2f} ±{r['hop_mean_std']:>4.2f}"
              f"{r['hop_within_1_mean']:>9.2%}     "
              f"{r['inference_time_ms_mean']:>9.3f} ±{r['inference_time_ms_std']:>5.3f}")
    print(f"{'=' * 78}")
    # Model size and training cost: quoted in the discussion, and the pairing
    # of accuracy with parameter count is what makes the comparison fair.
    print(f"  {'Model':<20}{'parameters':>16}{'train (s)':>14}")
    print(f"  {'-' * 68}")
    for _, r in table.iterrows():
        print(f"  {r['model']:<20}{r['n_params_mean']:>16,.0f}"
              f"{r['training_time_mean']:>14.0f}")
    print(f"{'=' * 78}")
    return table


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--dataset", type=config.resolve_path,
                        default=config.features_csv(5))
    parser.add_argument("--splits", type=config.resolve_path, default=config.SPLITS_PKL)
    parser.add_argument("--best-hp", type=config.resolve_path, default=config.BEST_HP_JSON)
    parser.add_argument("--graph", type=config.resolve_path, default=config.GRAPH_PKL)
    parser.add_argument("--out-dir", type=config.resolve_path, default=config.CV_DIR)
    parser.add_argument("--resume-from", type=int, default=0,
                        help="Skip folds below this index.")
    parser.add_argument("--skip", type=str, default="",
                        help="Comma-separated models to skip, e.g. TabNet.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config.ensure_dir(args.out_dir)

    skip = {m.strip() for m in args.skip.split(",") if m.strip()}
    needed = []
    if "TabNet" not in skip:
        needed.append("pytorch_tabnet")
    if "LightGBM" not in skip:
        needed.append("lightgbm")
    require(*needed)

    torch.manual_seed(config.SEED)
    np.random.seed(config.SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if not args.best_hp.exists():
        raise SystemExit(
            f"{args.best_hp} not found. Run scripts/08_optuna_search.py first."
        )
    with open(args.best_hp) as fh:
        best_hp = json.load(fh)
    # Tolerate both the flat mapping and the per-model result objects.
    best_hp = {
        k: (v["best_params"] if isinstance(v, dict) and "best_params" in v else v)
        for k, v in best_hp.items()
    }
    missing = [m for m in config.MODELS if m not in skip and m not in best_hp]
    if missing:
        raise SystemExit(
            f"best_hp.json has no entry for: {missing}. Run stage 8 for those "
            "models, or add them to --skip."
        )
    print(f"Hyper-parameters loaded for: {sorted(best_hp)}")

    print(f"\nReading {args.dataset}")
    df = pd.read_csv(args.dataset)
    features, targets, n_pmus = prepare_matrix(df, config.TARGET_COL)
    encoder = LabelEncoder().fit(targets)
    y_all = encoder.transform(targets)
    n_classes = len(encoder.classes_)
    feature_names = list(features.columns)
    matrix = features.to_numpy(dtype=np.float32)
    print(f"  {df.shape[0]:,} samples | {len(feature_names)} features "
          f"| {n_classes} classes | {n_pmus} PMUs")

    splits = load_splits(args.splits, y=targets)
    cv_pool = splits["cv_indices"]
    cv_folds = splits["cv_folds"]
    print(f"  CV pool: {len(cv_pool):,} samples in {len(cv_folds)} folds")

    # Group membership and the noise-safe column set are properties of the
    # feature schema, not of a fold, so they are computed once and stored
    # globally. The later analyses need them to perturb the right columns and
    # to re-apply the per-fold scalers; recomputing them from a different
    # config would silently desynchronise those analyses from the training.
    from src.scaling import group_indices

    exclude_prefixes = tuple(f"{c}_" for c in config.CATEGORICAL_FEATURES)
    feature_groups, noise_safe_columns = group_indices(
        feature_names, config.FEATURE_GROUPS, exclude_prefixes
    )
    unassigned = feature_groups.get("rest", [])
    if unassigned:
        print(f"  WARNING: {len(unassigned)} feature(s) match no group and were "
              f"scaled as 'rest': {[feature_names[i] for i in unassigned[:5]]}")

    graph = load_graph(args.graph)
    classes = [str(c) for c in encoder.classes_]
    hops = hop_matrix(graph, classes)
    print(f"  hop matrix {n_classes}x{n_classes}, "
          f"mean off-diagonal {hops[~np.eye(n_classes, dtype=bool)].mean():.2f}")

    # Fold indices are converted to GLOBAL dataset positions here, once. The
    # folds stored in splits.pkl are local to cv_indices, and a downstream
    # stage that indexed the full dataset with local positions would silently
    # evaluate on the wrong samples.
    fold_indices: dict[int, dict[str, np.ndarray]] = {}
    for k, (train_pool_local, test_local) in enumerate(cv_folds):
        train_local, val_local = train_test_split(
            train_pool_local,
            test_size=config.INTERNAL_VAL_FRACTION,
            stratify=y_all[cv_pool[train_pool_local]],
            random_state=config.SEED + k,
        )
        fold_indices[k] = {
            "train": cv_pool[train_local],
            "val": cv_pool[val_local],
            "test": cv_pool[test_local],
        }
    with open(args.out_dir / "fold_indices.pkl", "wb") as fh:
        pickle.dump(fold_indices, fh, protocol=pickle.HIGHEST_PROTOCOL)

    with open(args.out_dir / "preprocessing_global.pkl", "wb") as fh:
        pickle.dump(
            {
                "target_encoder": encoder,
                "classes_buses": classes,
                "feature_names": feature_names,
                "n_features": len(feature_names),
                "n_classes": n_classes,
                "n_pmus": n_pmus,
                "groups": feature_groups,
                "noise_safe_columns": noise_safe_columns,
                "clip_range": config.CLIP_RANGE,
                "hop_matrix": hops,
                "best_hp": best_hp,
                "seed": config.SEED,
                "n_folds": len(cv_folds),
                "dataset": str(args.dataset),
                "fold_index_space": "global dataset positions",
            },
            fh,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    print("  saved preprocessing_global.pkl and fold_indices.pkl")

    t_start = time.time()
    for fold in range(len(cv_folds)):
        if fold < args.resume_from:
            print(f"\n  fold {fold} skipped (--resume-from {args.resume_from})")
            continue
        run_fold(
            fold=fold,
            train_idx=fold_indices[fold]["train"],
            val_idx=fold_indices[fold]["val"],
            test_idx=fold_indices[fold]["test"],
            matrix=matrix, y=y_all, feature_names=feature_names,
            best_hp=best_hp, n_classes=n_classes, hops=hops,
            out_dir=args.out_dir, skip=skip, device=device,
        )
        done = fold - args.resume_from + 1
        remaining = len(cv_folds) - args.resume_from - done
        if remaining > 0:
            elapsed = time.time() - t_start
            print(f"\n  elapsed {timedelta(seconds=int(elapsed))} | "
                  f"ETA {timedelta(seconds=int(elapsed / done * remaining))}")

    aggregate(args.out_dir, len(cv_folds))
    print(f"\n  total: {timedelta(seconds=int(time.time() - t_start))}")
    print(f"  results in {args.out_dir}")


if __name__ == "__main__":
    main()
