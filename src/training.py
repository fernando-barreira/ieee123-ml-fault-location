"""Training loop, metrics and instrumentation shared by stages 8 and 9.

One loop trains every neural model.  That is a methodological requirement, not
a convenience: the paper compares architectures, so the optimiser, schedule,
stopping rule and epoch budget must be identical across them. If one model were
given a longer patience than another, the reported gap would partly measure the
budget rather than the architecture.

Schedule
--------
A short linear warm-up, then ``ReduceLROnPlateau`` on validation top-1.
Cyclic schedules were tried and rejected on this problem:
``CosineAnnealingWarmRestarts`` drops accuracy sharply at each restart and does
not always recover within the budget, and ``OneCycleLR`` stepped per batch
oscillates without settling. Plateau reduction is the schedule that behaves.

Metrics
-------
Top-1, Top-3 and Top-5 are all reported. For a fault locator the top-*k* sets
are operationally meaningful, not just a softer score: a crew dispatched with
three candidate buses on one lateral inspects them in a single trip, so Top-3
accuracy maps onto a real deployment mode.
"""

from __future__ import annotations

import time
from typing import Any, Callable

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from src.models import FaultDataset, FocalLoss, init_kaiming


# ═══════════════════════════════════════════════════════════════════════════
#  METRICS
# ═══════════════════════════════════════════════════════════════════════════
def topk_correct(logits: torch.Tensor, targets: torch.Tensor, k: int) -> int:
    """Number of samples whose true class is among the ``k`` highest logits."""
    _, topk = logits.topk(k, dim=1)
    return topk.eq(targets.unsqueeze(1).expand_as(topk)).any(dim=1).sum().item()


def topk_from_probs(probs: np.ndarray, y_true: np.ndarray, k: int) -> float:
    """Top-``k`` accuracy from a probability matrix.

    Uses ``argpartition`` rather than a full ``argsort``: only membership of
    the top-``k`` set matters, not the order within it, and the matrix is
    ``n_samples x 122``.
    """
    topk = np.argpartition(-probs, kth=k - 1, axis=1)[:, :k]
    return float(np.any(topk == y_true[:, None], axis=1).mean())


def metrics_from_probs(
    probs: np.ndarray, y_true: np.ndarray
) -> tuple[dict[str, float], np.ndarray]:
    """``({"top1", "top3", "top5"}, predictions)``."""
    preds = probs.argmax(axis=1)
    return (
        {
            "top1": float((preds == y_true).mean()),
            "top3": topk_from_probs(probs, y_true, 3),
            "top5": topk_from_probs(probs, y_true, 5),
        },
        preds,
    )


# ═══════════════════════════════════════════════════════════════════════════
#  LOADERS
# ═══════════════════════════════════════════════════════════════════════════
def make_loader(
    x: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    device: torch.device,
    noise: float = 0.0,
    noise_columns: torch.Tensor | None = None,
    train: bool = False,
    shuffle: bool = False,
    num_workers: int = 0,
) -> DataLoader:
    """Build a :class:`~torch.utils.data.DataLoader` over the feature matrix."""
    return DataLoader(
        FaultDataset(x, y, noise=noise, noise_columns=noise_columns, train=train),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(num_workers > 0),
    )


# ═══════════════════════════════════════════════════════════════════════════
#  TRAINING LOOP
# ═══════════════════════════════════════════════════════════════════════════
def train_neural(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    n_classes: int,
    hp: dict,
    device: torch.device,
    epochs: int,
    patience: int,
    warmup_epochs: int,
    weights: torch.Tensor | None = None,
    min_epochs: int = 0,
    on_epoch: Callable[[int, float], None] | None = None,
    keep_best_weights: bool = True,
    track_history: bool = True,
    log_every: int = 0,
    label: str = "",
) -> dict[str, Any]:
    """Train one neural model, returning the best validation state.

    Parameters
    ----------
    hp
        Must provide ``lr``, ``weight_decay``, ``focal_gamma``,
        ``label_smoothing`` and ``scheduler_patience``.
    on_epoch
        Called as ``on_epoch(epoch, val_top1)`` after each epoch, once
        ``epoch >= min_epochs``.  Stage 8 uses it to report intermediate values
        to Optuna's pruner; raising from inside it aborts the run.
    keep_best_weights
        Restore the parameters of the best epoch before returning.  Stage 9
        needs this because it saves and re-uses the model; stage 8 does not,
        and skipping it avoids copying the full state dict every improvement.

    Returns
    -------
    dict
        ``best_top1``, ``best_top3``, ``epochs_run``, ``training_time``,
        ``n_params`` and, if requested, ``history``.

    Notes
    -----
    Early stopping is on **top-1 accuracy**, not on validation loss.  Top-1 is
    the reported quantity, and with focal loss plus label smoothing the loss
    and the accuracy do not reach their optima at the same epoch.
    """
    model = model.to(device)
    model.apply(init_kaiming)

    criterion = FocalLoss(
        hp["focal_gamma"], hp["label_smoothing"], n_classes, weights
    )
    optimizer = optim.AdamW(
        model.parameters(), lr=hp["lr"], weight_decay=hp["weight_decay"]
    )
    warmup = optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda e: (e + 1) / max(warmup_epochs, 1)
    )
    plateau = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=hp["scheduler_patience"],
        min_lr=1e-5,
    )

    history: dict[str, list[float]] = {
        "train_loss": [], "val_loss": [],
        "train_acc": [], "val_acc": [], "val_top3": [], "val_top5": [],
    }
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if label:
        print(f"  -- {label} | params: {n_params:,} | lr: {hp['lr']:.2e} "
              f"| wd: {hp['weight_decay']:.2e}")

    best_top1 = best_top3 = 0.0
    best_state: dict | None = None
    stalled = 0
    epochs_run = 0
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        epochs_run = epoch

        model.train()
        train_loss = train_correct = train_total = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            if track_history:
                train_loss += loss.item() * yb.numel()
                train_correct += (logits.argmax(1) == yb).sum().item()
            train_total += yb.numel()

        model.eval()
        val_loss = 0.0
        c1 = c3 = c5 = 0
        val_total = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                logits = model(xb)
                if track_history:
                    val_loss += criterion(logits, yb).item() * yb.numel()
                c1 += (logits.argmax(1) == yb).sum().item()
                c3 += topk_correct(logits, yb, 3)
                c5 += topk_correct(logits, yb, 5)
                val_total += yb.numel()

        val_top1, val_top3, val_top5 = c1 / val_total, c3 / val_total, c5 / val_total

        if track_history:
            history["train_loss"].append(train_loss / train_total)
            history["val_loss"].append(val_loss / val_total)
            history["train_acc"].append(train_correct / train_total)
            history["val_acc"].append(val_top1)
            history["val_top3"].append(val_top3)
            history["val_top5"].append(val_top5)

        if epoch < warmup_epochs:
            warmup.step()
        else:
            plateau.step(val_top1)

        if log_every and (epoch % log_every == 0 or epoch == 1):
            print(f"     ep {epoch:4d} | T1 {val_top1:.4f} T3 {val_top3:.4f} "
                  f"| LR {optimizer.param_groups[0]['lr']:.2e}")

        if val_top1 > best_top1:
            best_top1, best_top3 = val_top1, val_top3
            stalled = 0
            if keep_best_weights:
                best_state = {
                    k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                }
        else:
            stalled += 1

        if on_epoch is not None and epoch >= min_epochs:
            on_epoch(epoch, val_top1)

        if epoch >= min_epochs and stalled >= patience:
            if label:
                print(f"     early stop at epoch {epoch} "
                      f"(T1={best_top1:.4f}, T3={best_top3:.4f})")
            break

    if keep_best_weights and best_state is not None:
        model.load_state_dict(best_state)

    result: dict[str, Any] = {
        "best_top1": float(best_top1),
        "best_top3": float(best_top3),
        "epochs_run": epochs_run,
        "training_time": time.time() - t0,
        "n_params": n_params,
    }
    if track_history:
        result["history"] = history
    return result


# ═══════════════════════════════════════════════════════════════════════════
#  PREDICTION AND INSTRUMENTATION
# ═══════════════════════════════════════════════════════════════════════════
def predict_probs(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> tuple[np.ndarray, np.ndarray]:
    """Class probabilities and labels over a loader, in loader order."""
    model.eval()
    probs, labels = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            probs.append(torch.softmax(model(xb), dim=1).cpu().numpy())
            labels.append(yb.numpy())
    return np.vstack(probs), np.concatenate(labels)


def inference_latency(
    predict_one: Callable[[int], Any], n_runs: int, synchronise: bool = False
) -> float:
    """Mean single-sample latency in milliseconds.

    Batch size 1 is the operationally meaningful setting: a fault locator runs
    once per event, so throughput on large batches says nothing about how long
    an operator waits.  The first ten calls are discarded as warm-up — they
    include lazy CUDA context creation and kernel autotuning.
    """
    for i in range(min(10, n_runs)):
        predict_one(i)
    if synchronise:
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    for i in range(n_runs):
        predict_one(i)
    if synchronise:
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n_runs * 1000.0


def nn_latency(
    model: nn.Module, x: np.ndarray, device: torch.device, n_runs: int
) -> float:
    """Single-sample latency of a PyTorch model."""
    model.eval()
    tensor = torch.as_tensor(x[:n_runs], dtype=torch.float32).to(device)

    def one(i: int):
        with torch.no_grad():
            return model(tensor[i:i + 1])

    return inference_latency(one, n_runs, synchronise=(device.type == "cuda"))


def sklearn_latency(model, x: np.ndarray, n_runs: int) -> float:
    """Single-sample latency of an estimator exposing ``predict_proba``."""
    return inference_latency(lambda i: model.predict_proba(x[i:i + 1]), n_runs)


# ═══════════════════════════════════════════════════════════════════════════
#  DEPENDENCY PREFLIGHT
# ═══════════════════════════════════════════════════════════════════════════
#: Import name -> (pip name, what it is needed for).
OPTIONAL_DEPENDENCIES = {
    "optuna": ("optuna", "the hyper-parameter search (stage 8)"),
    "lightgbm": ("lightgbm", "the LightGBM model"),
    "pytorch_tabnet": ("pytorch-tabnet", "the TabNet model"),
    "py_dss_interface": ("py_dss_interface", "the OpenDSS-dependent stages"),
}


def require(*modules: str) -> None:
    """Fail immediately, and once, if a required package is missing.

    Stages 8 and 9 run for hours and catch per-model exceptions so that one bad
    configuration does not abort the whole sweep.  That is right for a model
    that diverges, and wrong for a missing import: the same ``ModuleNotFoundError``
    is then reported once per model, which reads like five unrelated failures
    instead of one ``pip install``.  Checking up front separates the two.

    Raises
    ------
    SystemExit
        Naming every missing package and the single command that installs them.
    """
    missing = []
    for name in modules:
        try:
            __import__(name)
        except ImportError:
            missing.append(name)

    if not missing:
        return

    lines = ["Missing required package(s):"]
    for name in missing:
        pip_name, purpose = OPTIONAL_DEPENDENCIES.get(name, (name, "this stage"))
        lines.append(f"  - {pip_name:<18s} needed for {purpose}")
    pip_names = " ".join(
        OPTIONAL_DEPENDENCIES.get(n, (n, ""))[0] for n in missing
    )
    lines += [
        "",
        f"Install them with:",
        f"  pip install {pip_names}",
        "",
        "Then re-run scripts/00_check_setup.py to confirm the environment.",
    ]
    raise SystemExit("\n".join(lines))