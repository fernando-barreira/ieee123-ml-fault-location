"""Stage 8 — per-model hyper-parameter search with Optuna.

Searches each of the five models independently on the frozen ``optuna`` pool,
which is disjoint from the cross-validation pool that produces the reported
results and from the untouched holdout.

Protocol
--------
* **Frozen split.**  ``optuna_tr`` (10 500) trains, ``optuna_va`` (4 500)
  scores.  The split never changes between trials, so differences between
  trials reflect hyper-parameters rather than resampling noise.
* **Identical training budget for every neural model.**  ``SEARCH_EPOCHS``,
  ``SEARCH_PATIENCE`` and ``SEARCH_MIN_EPOCHS`` come from ``config`` and are
  not searched.  The paper compares architectures; if one were allowed a longer
  schedule than another, the comparison would partly measure the schedule.
* **Preprocessing done once.**  Scalers are fitted on ``optuna_tr`` and reused
  across trials — they do not depend on the hyper-parameters, and refitting per
  trial would only add noise and time.
* **Trees see the unscaled matrix**, matching stage 9 exactly.  Splitting
  thresholds are invariant to the affine part of the scaling, but the
  post-scaling clip is not, so a tree tuned on clipped inputs and then trained
  on raw ones would not be the model that was selected.
* **Pruning** with ``MedianPruner`` for the neural models, whose per-epoch
  validation accuracy is a usable progress signal.  Tree models produce a
  single score at the end, so pruning does not apply.
* **SQLite storage**, so an interrupted search resumes where it stopped.

Outputs, under ``results/optuna/``:

=============================  ===============================================
``<model>/study.db``           Optuna storage; delete to restart that model.
``<model>/best_params.json``   Best trial and its parameters.
``<model>/trials_history.csv`` Every trial, for the appendix.
``best_hp.json``               Consolidated input to stage 9.
``optuna_summary.csv``         One row per model.
=============================  ===============================================

Usage
-----
.. code-block:: bash

    python scripts/08_optuna_search.py
    python scripts/08_optuna_search.py --models MLP,RF
    python scripts/08_optuna_search.py --models DeepResidualMLP --n-trials 100
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config
from src.models import build_model, class_weights
from src.scaling import prepare_matrix, scale_splits
from src.splits import load_splits
from src.training import make_loader, require, train_neural

warnings.filterwarnings("ignore", category=UserWarning)
# LightGBM >= 4.6 renamed eval_set to eval_X/eval_y but still honours the old
# name; keep the old spelling for compatibility with earlier versions and
# silence the notice, which would otherwise print once per trial.
warnings.filterwarnings("ignore", message=".*eval_set.*deprecated.*")


# ═══════════════════════════════════════════════════════════════════════════
#  SEARCH SPACES
# ═══════════════════════════════════════════════════════════════════════════
def space_mlp(trial) -> dict:
    """Depth, width, regularisation and the loss shape."""
    return {
        "n_layers": trial.suggest_int("n_layers", 2, 4),
        "hidden_0": trial.suggest_categorical("hidden_0", [512, 1024, 2048]),
        "dropout": trial.suggest_float("dropout", 0.2, 0.5),
        "activation": trial.suggest_categorical("activation", ["gelu", "relu", "silu"]),
        "lr": trial.suggest_float("lr", 1e-4, 5e-3, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [256, 512, 1024]),
        "focal_gamma": trial.suggest_float("focal_gamma", 1.0, 2.5),
        "label_smoothing": trial.suggest_float("label_smoothing", 0.0, 0.1),
        "class_weight_mode": trial.suggest_categorical(
            "class_weight_mode", ["sqrt", "uniform", "linear"]
        ),
        "noise_std": trial.suggest_float("noise_std", 0.0, 0.01),
        "scheduler_patience": trial.suggest_categorical("scheduler_patience", [10, 20]),
    }


def space_deep_residual(trial) -> dict:
    """As above, plus depth of the residual trunk and the dropout ramp.

    ``dropout_end`` is drawn from ``[dropout_start, 0.6]`` so the ramp can only
    increase with depth.  Allowing it to decrease would let dropout be
    strongest on the layers closest to the raw phasors, where information is
    densest and dropping it costs the most signal.
    """
    dropout_start = trial.suggest_float("dropout_start", 0.1, 0.4)
    return {
        "lr": trial.suggest_float("lr", 3e-4, 3e-3, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-4, 1e-2, log=True),
        "dropout_start": dropout_start,
        "dropout_end": trial.suggest_float("dropout_end", dropout_start, 0.6),
        "hidden": trial.suggest_categorical("hidden", [256, 384, 512, 640]),
        "n_blocks": trial.suggest_int("n_blocks", 3, 7),
        "batch_size": trial.suggest_categorical("batch_size", [512, 1024, 2048]),
        "focal_gamma": trial.suggest_float("focal_gamma", 1.0, 2.5),
        "label_smoothing": trial.suggest_float("label_smoothing", 0.0, 0.1),
        "class_weight_mode": trial.suggest_categorical(
            "class_weight_mode", ["sqrt", "uniform", "linear"]
        ),
        "noise_std": trial.suggest_float("noise_std", 0.0, 0.01),
        "activation": trial.suggest_categorical("activation", ["gelu", "relu", "silu"]),
        "residual_scale": trial.suggest_float("residual_scale", 0.7, 1.0),
        "scheduler_patience": trial.suggest_categorical(
            "scheduler_patience", [15, 25, 40]
        ),
    }


def space_tabnet(trial) -> dict:
    """TabNet's attention capacity, sparsity and step count.

    ``virtual_batch_size`` controls ghost batch normalisation and is kept well
    below ``batch_size``, as the method requires.
    """
    return {
        "n_d": trial.suggest_categorical("n_d", [16, 32, 64]),
        "n_steps": trial.suggest_int("n_steps", 3, 7),
        "gamma": trial.suggest_float("gamma", 1.0, 2.0),
        "lambda_sparse": trial.suggest_float("lambda_sparse", 1e-5, 1e-3, log=True),
        "lr": trial.suggest_float("lr", 5e-3, 5e-2, log=True),
        "mask_type": trial.suggest_categorical("mask_type", ["sparsemax", "entmax"]),
        "virtual_batch_size": trial.suggest_categorical(
            "virtual_batch_size", [64, 128, 256]
        ),
        "batch_size": trial.suggest_categorical("batch_size", [512, 1024]),
        "class_weight_mode": trial.suggest_categorical(
            "class_weight_mode", ["sqrt", "uniform", "linear"]
        ),
    }


def space_rf(trial) -> dict:
    """Random forest.

    ``max_features`` is searched over a deliberately low range: the phasor
    features are strongly correlated, so considering all of them at every split
    would produce near-identical trees and defeat the ensemble.
    """
    return {
        "n_estimators": trial.suggest_categorical("n_estimators", [300, 500, 800]),
        "max_features": trial.suggest_float("max_features", 0.1, 0.5),
        "min_samples_leaf": trial.suggest_categorical("min_samples_leaf", [1, 3, 5, 10]),
        "max_depth": trial.suggest_categorical("max_depth", [None, 20, 40]),
        "max_samples": trial.suggest_float("max_samples", 0.6, 1.0),
        "class_weight": trial.suggest_categorical(
            "class_weight", [None, "balanced", "balanced_subsample"]
        ),
    }


def space_lightgbm(trial) -> dict:
    """LightGBM.  The tree count is fixed high and settled by early stopping."""
    return {
        "num_leaves": trial.suggest_categorical("num_leaves", [31, 63, 127, 255]),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        "min_data_in_leaf": trial.suggest_categorical(
            "min_data_in_leaf", [20, 50, 100]
        ),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
        "lambda_l1": trial.suggest_float("lambda_l1", 1e-4, 1.0, log=True),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-4, 1.0, log=True),
    }


SEARCH_SPACES = {
    "MLP": space_mlp,
    "DeepResidualMLP": space_deep_residual,
    "TabNet": space_tabnet,
    "RF": space_rf,
    "LightGBM": space_lightgbm,
}


# ═══════════════════════════════════════════════════════════════════════════
#  OBJECTIVES
# ═══════════════════════════════════════════════════════════════════════════
def evaluate_torch(name: str, data: dict, hp: dict, trial, device) -> float:
    """Train an MLP-family model and return its best validation top-1."""
    import optuna

    weights = torch.as_tensor(
        class_weights(data["y_train"], data["n_classes"], hp["class_weight_mode"]),
        dtype=torch.float32,
        device=device,
    )
    noise_columns = torch.as_tensor(data["noise_columns"], dtype=torch.long)

    train_loader = make_loader(
        data["x_train"], data["y_train"], hp["batch_size"], device,
        noise=hp["noise_std"], noise_columns=noise_columns,
        train=True, shuffle=True, num_workers=config.DATALOADER_WORKERS,
    )
    val_loader = make_loader(
        data["x_val"], data["y_val"], hp["batch_size"], device,
        num_workers=config.DATALOADER_WORKERS,
    )

    def report(epoch: int, val_top1: float) -> None:
        trial.report(val_top1, step=epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()

    result = train_neural(
        build_model(name, hp, data["n_features"], data["n_classes"]),
        train_loader, val_loader,
        n_classes=data["n_classes"],
        hp=hp,
        device=device,
        epochs=config.SEARCH_EPOCHS,
        patience=config.SEARCH_PATIENCE,
        warmup_epochs=config.SEARCH_WARMUP_EPOCHS,
        weights=weights,
        min_epochs=config.SEARCH_MIN_EPOCHS,
        on_epoch=report,
        keep_best_weights=False,
        track_history=False,
    )
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result["best_top1"]


def evaluate_tabnet(data: dict, hp: dict, trial, device) -> float:
    """Train TabNet and return its best validation accuracy."""
    import optuna
    from pytorch_tabnet.callbacks import Callback
    from pytorch_tabnet.tab_model import TabNetClassifier

    weights = class_weights(
        data["y_train"], data["n_classes"], hp["class_weight_mode"]
    )

    class PruningCallback(Callback):
        """Report validation accuracy to Optuna after each epoch.

        Must subclass pytorch-tabnet's ``Callback`` and override
        ``on_epoch_end``; a plain callable is never invoked by the fit loop.
        With ``eval_name=["val"]`` the metric key is ``val_accuracy``.
        """

        def __init__(self, trial, min_epochs: int) -> None:
            super().__init__()
            self.trial = trial
            self.min_epochs = min_epochs

        def on_epoch_end(self, epoch, logs=None):
            if logs is None or self.trial is None or epoch < self.min_epochs:
                return
            accuracy = logs.get("val_accuracy", logs.get("val_0_accuracy", 0.0))
            self.trial.report(accuracy, step=epoch)
            if self.trial.should_prune():
                raise optuna.TrialPruned()

    model = TabNetClassifier(
        n_d=hp["n_d"], n_a=hp["n_d"],
        n_steps=hp["n_steps"],
        gamma=hp["gamma"],
        lambda_sparse=hp["lambda_sparse"],
        optimizer_fn=torch.optim.AdamW,
        optimizer_params={"lr": hp["lr"], "weight_decay": 1e-5},
        scheduler_fn=torch.optim.lr_scheduler.StepLR,
        scheduler_params={"step_size": 20, "gamma": 0.7},
        mask_type=hp["mask_type"],
        seed=config.SEED,
        device_name=device.type,
        verbose=0,
    )
    model.fit(
        X_train=data["x_train"], y_train=data["y_train"],
        eval_set=[(data["x_val"], data["y_val"])],
        eval_name=["val"], eval_metric=["accuracy"],
        max_epochs=config.SEARCH_EPOCHS,
        patience=config.SEARCH_PATIENCE,
        batch_size=hp["batch_size"],
        virtual_batch_size=hp["virtual_batch_size"],
        drop_last=False,
        weights={i: float(w) for i, w in enumerate(weights)},
        callbacks=[PruningCallback(trial, config.SEARCH_MIN_EPOCHS)] if trial else [],
    )
    score = float(max(model.history["val_accuracy"]))
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return score


def evaluate_rf(data: dict, hp: dict) -> float:
    """Train a random forest on the unscaled matrix and score it."""
    from sklearn.ensemble import RandomForestClassifier

    forest = RandomForestClassifier(
        n_estimators=hp["n_estimators"],
        max_features=hp["max_features"],
        min_samples_leaf=hp["min_samples_leaf"],
        max_depth=hp["max_depth"],
        max_samples=hp["max_samples"],
        class_weight=hp["class_weight"],
        bootstrap=True,  # required for max_samples to take effect
        n_jobs=-1,
        random_state=config.SEED,
    )
    forest.fit(data["x_train_raw"], data["y_train"])
    return float(forest.score(data["x_val_raw"], data["y_val"]))


def evaluate_lightgbm(data: dict, hp: dict) -> float:
    """Train LightGBM on the unscaled matrix and score it.

    ``multiclassova`` (one-vs-all) rather than the softmax objective: with 122
    classes the softmax formulation is slower and, here, less stable.
    """
    import lightgbm as lgb

    model = lgb.LGBMClassifier(
        objective="multiclassova",
        num_class=data["n_classes"],
        n_estimators=config.LGB_N_ESTIMATORS,
        num_leaves=hp["num_leaves"],
        learning_rate=hp["learning_rate"],
        min_data_in_leaf=hp["min_data_in_leaf"],
        feature_fraction=hp["feature_fraction"],
        bagging_fraction=hp["bagging_fraction"],
        bagging_freq=config.LGB_BAGGING_FREQ,
        lambda_l1=hp["lambda_l1"],
        lambda_l2=hp["lambda_l2"],
        random_state=config.SEED,
        n_jobs=-1,
        verbosity=-1,
    )
    model.fit(
        data["x_train_raw"], data["y_train"],
        eval_set=[(data["x_val_raw"], data["y_val"])],
        eval_metric="multi_error",
        callbacks=[lgb.early_stopping(config.LGB_EARLY_STOPPING_ROUNDS, verbose=False)],
    )
    return float(model.score(data["x_val_raw"], data["y_val"]))


def make_objective(name: str, data: dict, device):
    """Build the Optuna objective for one model."""
    import optuna

    space = SEARCH_SPACES[name]

    def objective(trial) -> float:
        hp = space(trial)
        t0 = time.time()
        try:
            if name in ("MLP", "DeepResidualMLP"):
                score = evaluate_torch(name, data, hp, trial, device)
            elif name == "TabNet":
                score = evaluate_tabnet(data, hp, trial, device)
            elif name == "RF":
                score = evaluate_rf(data, hp)
            else:
                score = evaluate_lightgbm(data, hp)
        except optuna.TrialPruned:
            raise
        except Exception as exc:  # noqa: BLE001
            # A failed configuration is information, not a crash: record it as
            # the worst possible score so the sampler avoids that region.
            print(f"    trial {trial.number} failed: {type(exc).__name__}: {exc}")
            return 0.0
        print(f"    trial {trial.number}: {score:.4f} ({time.time() - t0:.0f}s)")
        return score

    return objective


# ═══════════════════════════════════════════════════════════════════════════
#  DATA
# ═══════════════════════════════════════════════════════════════════════════
def load_data(dataset: Path, splits_path: Path) -> dict:
    """Load the dataset, apply the frozen split and fit the scalers once."""
    print(f"Reading {dataset}")
    df = pd.read_csv(dataset)
    print(f"  shape: {df.shape}")

    features, targets, n_pmus = prepare_matrix(df, config.TARGET_COL)
    encoder = LabelEncoder().fit(targets)
    y = encoder.transform(targets)
    print(f"  classes: {len(encoder.classes_)} | PMUs: {n_pmus} | "
          f"features after one-hot: {features.shape[1]}")

    splits = load_splits(splits_path, y=targets)
    idx_train, idx_val = splits["optuna_tr"], splits["optuna_va"]
    print(f"  optuna_tr: {len(idx_train):,} | optuna_va: {len(idx_val):,}")

    matrix = features.to_numpy(dtype=np.float32)
    x_train_raw, x_val_raw = matrix[idx_train], matrix[idx_val]
    x_train, (x_val,), _, noise_columns = scale_splits(
        x_train_raw, [x_val_raw], features.columns
    )
    print(f"  noise-safe columns: {len(noise_columns)}")

    return {
        "x_train": x_train, "x_val": x_val,
        "x_train_raw": x_train_raw, "x_val_raw": x_val_raw,
        "y_train": y[idx_train], "y_val": y[idx_val],
        "n_classes": len(encoder.classes_),
        "n_features": x_train.shape[1],
        "noise_columns": noise_columns,
        "feature_names": list(features.columns),
    }


# ═══════════════════════════════════════════════════════════════════════════
#  DRIVER
# ═══════════════════════════════════════════════════════════════════════════
def run_search(
    name: str,
    data: dict,
    n_trials: int,
    out_dir: Path,
    device,
    pruner_warmup: int = config.PRUNER_WARMUP_STEPS,
    prune: bool = True,
) -> dict:
    """Create or resume a study for one model and optimise it."""
    import optuna
    from optuna.pruners import MedianPruner, NopPruner
    from optuna.samplers import TPESampler
    from optuna.trial import TrialState

    model_dir = config.ensure_dir(out_dir / name)
    storage = f"sqlite:///{(model_dir / 'study.db').as_posix()}"

    print(f"\n{'=' * 70}\n{name}  ({n_trials} trials)\n{'=' * 70}")

    study = optuna.create_study(
        study_name=f"optuna_{name}",
        direction="maximize",
        sampler=TPESampler(seed=config.SEED, n_startup_trials=config.TPE_STARTUP_TRIALS),
        # Pruning applies only to the neural models, whose per-epoch validation
        # accuracy is a usable progress signal; the tree models produce a single
        # score at the end. The warmup must be long enough that a deep network
        # has passed its slow initial phase, otherwise the pruner systematically
        # favours architectures that converge fast rather than far.
        pruner=(
            MedianPruner(
                n_startup_trials=config.PRUNER_STARTUP_TRIALS,
                n_warmup_steps=pruner_warmup,
            )
            if (prune and name in config.NEURAL_MODELS)
            else NopPruner()
        ),
        storage=storage,
        load_if_exists=True,
    )

    done = sum(
        t.state in (TrialState.COMPLETE, TrialState.PRUNED, TrialState.FAIL)
        for t in study.trials
    )
    remaining = max(0, n_trials - done)
    print(f"  already finished: {done} | remaining: {remaining}")

    if remaining:
        t0 = time.time()
        try:
            study.optimize(
                make_objective(name, data, device),
                n_trials=remaining,
                show_progress_bar=False,
            )
        except KeyboardInterrupt:
            print("  interrupted; progress is stored in study.db")
        print(f"  elapsed: {(time.time() - t0) / 60:.1f} min")

    print(f"\n  best top-1 (validation): {study.best_value:.4f}")
    for key, value in study.best_params.items():
        formatted = f"{value:.5g}" if isinstance(value, float) else value
        print(f"    {key:<22s}: {formatted}")

    result = {
        "model": name,
        "best_value": float(study.best_value),
        "best_params": study.best_params,
        "n_trials_completed": sum(
            t.state == TrialState.COMPLETE for t in study.trials
        ),
        "n_trials_pruned": sum(t.state == TrialState.PRUNED for t in study.trials),
    }
    config.write_json_atomic(model_dir / "best_params.json", result)
    study.trials_dataframe().to_csv(model_dir / "trials_history.csv", index=False)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--dataset", type=config.resolve_path,
                        default=config.features_csv(5))
    parser.add_argument("--splits", type=config.resolve_path, default=config.SPLITS_PKL)
    parser.add_argument("--out-dir", type=config.resolve_path, default=config.OPTUNA_DIR)
    parser.add_argument(
        "--models", type=str, default=",".join(config.MODELS),
        help=f"Comma-separated subset of {list(config.MODELS)}.",
    )
    parser.add_argument("--n-trials", type=int, default=config.N_TRIALS)
    parser.add_argument(
        "--pruner-warmup", type=int, default=config.PRUNER_WARMUP_STEPS,
        help="Epochs before the median pruner may terminate a trial. Raise it "
             "for deep architectures, which converge more slowly early on and "
             "can otherwise be pruned before they overtake shallower ones.",
    )
    parser.add_argument(
        "--no-prune", action="store_true",
        help="Disable pruning entirely. Slower, but removes any bias against "
             "configurations that start slowly.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config.ensure_dir(args.out_dir)

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    unknown = [m for m in models if m not in config.MODELS]
    if unknown:
        raise SystemExit(f"Unknown model(s): {unknown}. Choose from {list(config.MODELS)}.")

    # Checked before the sweep starts. The per-model try/except below is there
    # to survive a diverging configuration, not a missing package; without this
    # an absent import is reported once per model and looks like five failures.
    needed = ["optuna"]
    if "TabNet" in models:
        needed.append("pytorch_tabnet")
    if "LightGBM" in models:
        needed.append("lightgbm")
    require(*needed)

    # Checked now, not after the sweep: these are written last, so a locked or
    # read-only file would otherwise surface only after hours of searching.
    config.ensure_writable_file(args.out_dir / "best_hp.json")
    config.ensure_writable_file(args.out_dir / "optuna_summary.csv")

    torch.manual_seed(config.SEED)
    np.random.seed(config.SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Device : {device}")
    print(f"Models : {models}")
    print(f"Trials : {args.n_trials} each")
    print(f"Pruning: "
          + ("disabled" if args.no_prune
             else f"median, warmup {args.pruner_warmup} epochs") + "\n")

    data = load_data(args.dataset, args.splits)

    results: dict[str, dict] = {}
    for name in models:
        try:
            results[name] = run_search(
                name, data, args.n_trials, args.out_dir, device,
                pruner_warmup=args.pruner_warmup, prune=not args.no_prune,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"\n  {name} failed: {type(exc).__name__}: {exc}")

    if not results:
        raise SystemExit("No model completed; nothing to consolidate.")

    # Merge with any previously searched models so that running one model at a
    # time still produces a complete best_hp.json for stage 9.
    merged: dict[str, dict] = {}
    if config.BEST_HP_JSON.exists():
        with open(config.BEST_HP_JSON) as fh:
            merged = json.load(fh)
    merged.update({name: r["best_params"] for name, r in results.items()})
    written = config.write_json_atomic(args.out_dir / "best_hp.json", merged)

    summary = pd.DataFrame(
        [
            {
                "model": name,
                "best_top1_val": r["best_value"],
                "n_completed": r["n_trials_completed"],
                "n_pruned": r["n_trials_pruned"],
            }
            for name, r in results.items()
        ]
    ).sort_values("best_top1_val", ascending=False)
    try:
        summary.to_csv(args.out_dir / "optuna_summary.csv", index=False)
    except OSError as exc:  # noqa: BLE001 - the per-model files already hold this
        print(f"WARNING: could not write optuna_summary.csv ({exc})")

    print(f"\n{'=' * 70}")
    print(summary.to_string(index=False))
    print(f"\n  best_hp.json : {written} ({len(merged)} model(s))")
    print("  Per-model results are also in <model>/best_params.json; "
          "tools/rebuild_best_hp.py reconstructs the merged file from them.")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()