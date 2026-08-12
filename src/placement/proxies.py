"""Proxy models used to score a candidate PMU placement.

The placement search evaluates hundreds of PMU subsets.  Training the final
models on each one is out of the question, so each subset is scored with two
deliberately small models and the scores are combined.

Why two models
--------------
The two proxies have opposite inductive biases:

* the **MLP** builds dense linear combinations of all inputs, which suits
  continuous, strongly correlated phasor features;
* the **random forest** makes axis-aligned splits on individual features.

Scoring with a single proxy would bias the placement towards whatever that
architecture happens to exploit.  Since the final study compares gradient-based
and tree-based models against each other, letting one of them choose the
sensors would build the conclusion into the experimental setup.  The aggregate
score is the geometric mean

.. math:: s = \\sqrt{s_\\text{MLP}\\, s_\\text{RF}}

which, unlike the arithmetic mean, penalises placements that suit one family
and not the other: a subset scoring 0.40/0.10 gets 0.20, while a balanced
0.25/0.25 gets 0.25.

Both proxies are trained on the frozen ``fsnr_tr`` split and scored on the
frozen ``fsnr_va`` split.  Re-splitting per candidate would make the scores
incomparable across candidates, since some of the variance between subsets
would be split noise rather than sensor information.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.ensemble import RandomForestClassifier
from torch.utils.data import DataLoader, Dataset


# ═══════════════════════════════════════════════════════════════════════════
#  DATA
# ═══════════════════════════════════════════════════════════════════════════
class PhasorDataset(Dataset):
    """Feature matrix with optional multiplicative measurement noise.

    Noise emulates instrument-transformer and PMU accuracy limits.  It is
    applied only to the columns listed in ``noise_columns`` — voltages,
    currents and powers, i.e. quantities a real instrument measures — and it is
    **multiplicative**, because instrument error is specified as a percentage
    of reading rather than as an absolute value.

    .. note::
       Here the noise is applied to already-scaled columns purely because the
       proxy models are throwaway rankers.  For the final models, noise must be
       injected in the raw domain *before* normalisation; injecting it after
       changes the effective signal-to-noise ratio column by column and costs
       several accuracy points.
    """

    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        noise: float = 0.0,
        noise_columns: torch.Tensor | None = None,
        train: bool = False,
    ) -> None:
        self.x = torch.as_tensor(x, dtype=torch.float32)
        self.y = torch.as_tensor(y, dtype=torch.long)
        self.noise = noise
        self.noise_columns = noise_columns
        self.train = train

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, i: int):
        x = self.x[i].clone()
        if self.train and self.noise > 0 and self.noise_columns is not None:
            std = torch.clamp(self.noise * x[self.noise_columns].abs(), min=1e-6)
            perturbation = torch.zeros_like(x)
            perturbation[self.noise_columns] = (
                torch.randn_like(x[self.noise_columns]) * std
            )
            x = x + perturbation
        return x, self.y[i]


# ═══════════════════════════════════════════════════════════════════════════
#  MLP PROXY
# ═══════════════════════════════════════════════════════════════════════════
class ResidualBlock(nn.Module):
    """Pre-activation residual block with a damped skip connection.

    The residual branch is scaled by 0.9 rather than 1.0.  With 122 classes and
    only 3 500 training samples the unscaled identity path lets early-layer
    noise reach the head untouched; damping it slightly improves the stability
    of the proxy without changing its ranking behaviour.
    """

    def __init__(self, dim: int, dropout: float) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
        )
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(x + 0.9 * self.body(x))


class ProxyMLP(nn.Module):
    """Reduced version of the study's Deep Residual MLP.

    Two residual blocks instead of four and a narrower hidden width.  Its
    absolute accuracy is well below the final model's; what matters is that it
    ranks PMU subsets in the same order.
    """

    def __init__(
        self,
        n_features: int,
        n_classes: int,
        hidden: int,
        n_blocks: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Linear(n_features, hidden), nn.BatchNorm1d(hidden), nn.GELU()
        )
        self.blocks = nn.Sequential(
            *[ResidualBlock(hidden, dropout) for _ in range(n_blocks)]
        )
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.blocks(self.stem(x)))


class FocalLoss(nn.Module):
    """Focal loss with label smoothing and optional per-class weights.

    The class distribution is strongly imbalanced by construction: 75 % of the
    faults are single-line-to-ground, so buses on single-phase laterals are
    sampled far more often than three-phase trunk buses.  Plain cross-entropy
    lets the frequent classes dominate the gradient.

    The focal term :math:`(1-p_t)^\\gamma` down-weights samples the model
    already classifies confidently, concentrating capacity on the hard ones.
    ``gamma = 1.5`` is milder than the 2.0 of the original object-detection
    formulation, which suits a 122-class problem where even correct
    predictions rarely reach high confidence.

    Label smoothing spreads a small amount of probability mass over the other
    classes.  Beyond regularisation, it is physically appropriate here: buses
    joined by a closed switch are nearly indistinguishable, so a confidently
    wrong prediction between them should not be punished as harshly as one that
    picks a different lateral.
    """

    def __init__(
        self,
        gamma: float = 1.5,
        label_smoothing: float = 0.02,
        n_classes: int = 122,
        class_weights: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.n_classes = n_classes
        self.class_weights = class_weights

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            smooth = torch.full_like(
                logits, self.label_smoothing / self.n_classes
            )
            smooth.scatter_(
                1,
                target.unsqueeze(1),
                1 - self.label_smoothing + self.label_smoothing / self.n_classes,
            )
        log_p = F.log_softmax(logits, dim=1)
        p = log_p.exp()
        loss = -(smooth * (1 - p) ** self.gamma * log_p).sum(dim=1)
        if self.class_weights is not None:
            loss = loss * self.class_weights[target]
        return loss.mean()


def sqrt_class_weights(y: np.ndarray, n_classes: int) -> np.ndarray:
    """Class weights proportional to :math:`1/\\sqrt{n_c}`, normalised to mean 1.

    Full inverse-frequency weighting (:math:`1/n_c`) over-corrects here: with a
    max/min class ratio in the hundreds it makes the loss on the rarest buses
    dominate and destabilises training.  The square root is the usual
    compromise — it removes most of the imbalance while keeping the gradient
    scale comparable across classes.
    """
    counts = np.bincount(y, minlength=n_classes)
    weights = 1.0 / np.sqrt(np.where(counts > 0, counts, 1))
    return weights / weights.sum() * n_classes


def init_kaiming(module: nn.Module) -> None:
    """He initialisation, appropriate for the GELU/ReLU-family activations."""
    if isinstance(module, nn.Linear):
        nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.BatchNorm1d):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)


def train_mlp_proxy(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    n_classes: int,
    noise_columns: list[int],
    cfg: dict,
    device: torch.device,
    seed: int,
) -> float:
    """Train the proxy MLP and return its best top-1 validation accuracy.

    Uses ``ReduceLROnPlateau`` after a short linear warm-up.  Cyclic schedules
    were tried and rejected: ``CosineAnnealingWarmRestarts`` collapses accuracy
    at every restart, and ``OneCycleLR`` stepped per batch oscillates without
    settling on this dataset size.

    The "rescue" mechanism handles a specific failure: with only 3 500 training
    samples spread over 122 classes, an unlucky initialisation occasionally
    leaves the network stuck near random guessing.  If accuracy is still below
    ``rescue_threshold`` after ``min_epochs``, the learning rate is cut and the
    budget extended once, rather than recording a near-zero score that would
    wrongly disqualify the PMU subset under test.
    """
    torch.manual_seed(seed)

    weights = torch.as_tensor(
        sqrt_class_weights(y_train, n_classes), dtype=torch.float32, device=device
    )
    noise_idx = torch.as_tensor(noise_columns, dtype=torch.long)
    pin = device.type == "cuda"

    train_loader = DataLoader(
        PhasorDataset(
            x_train, y_train, cfg["train_noise"], noise_idx, train=True
        ),
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=0,
        pin_memory=pin,
    )
    val_loader = DataLoader(
        PhasorDataset(x_val, y_val),
        batch_size=cfg["batch_size"],
        num_workers=0,
        pin_memory=pin,
    )

    model = ProxyMLP(
        x_train.shape[1],
        n_classes,
        hidden=cfg["hidden"],
        n_blocks=cfg["n_blocks"],
        dropout=cfg["dropout"],
    ).to(device)
    model.apply(init_kaiming)

    criterion = FocalLoss(
        gamma=cfg.get("focal_gamma", 1.5),
        label_smoothing=cfg.get("label_smoothing", 0.02),
        n_classes=n_classes,
        class_weights=weights,
    )
    optimizer = optim.AdamW(
        model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"]
    )
    warmup = optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda e: (e + 1) / cfg["warmup_epochs"]
    )
    plateau = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=10, min_lr=1e-5
    )

    best = 0.0
    stalled = 0
    rescued = False
    epoch = 0
    budget = cfg["epochs"]

    while epoch < budget:
        epoch += 1
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        model.eval()
        correct = total = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                correct += (model(xb).argmax(1) == yb).sum().item()
                total += yb.numel()
        accuracy = correct / total

        if epoch < cfg["warmup_epochs"]:
            warmup.step()
        else:
            plateau.step(accuracy)

        if accuracy > best:
            best, stalled = accuracy, 0
        else:
            stalled += 1

        if (
            not rescued
            and epoch == cfg["min_epochs"]
            and best < cfg["rescue_threshold"]
        ):
            rescued = True
            budget += cfg["rescue_epochs"]
            stalled = 0
            for group in optimizer.param_groups:
                group["lr"] *= 0.3

        if epoch >= cfg["min_epochs"] and stalled >= cfg["patience"]:
            break

    del model, optimizer, train_loader, val_loader
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return float(best)


# ═══════════════════════════════════════════════════════════════════════════
#  RANDOM FOREST PROXY
# ═══════════════════════════════════════════════════════════════════════════
def train_rf_proxy(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    cfg: dict,
    seed: int,
) -> float:
    """Train the proxy random forest and return its top-1 validation accuracy.

    ``max_features=0.3`` keeps individual trees decorrelated despite the strong
    correlation between phasor features: the three phases of one PMU, and the
    same quantity at nearby PMUs, carry largely redundant information, so
    considering all features at every split would produce nearly identical
    trees.  ``class_weight="balanced_subsample"`` rebalances within each
    bootstrap sample, which matters given the 122-class imbalance.
    """
    forest = RandomForestClassifier(
        n_estimators=cfg["n_estimators"],
        max_features=cfg["max_features"],
        min_samples_leaf=cfg["min_samples_leaf"],
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=seed,
    )
    forest.fit(x_train, y_train)
    return float(forest.score(x_val, y_val))
