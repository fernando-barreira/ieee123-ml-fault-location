"""Model architectures, loss and initialisation shared by every training stage.

Defining these once means the hyper-parameter search (stage 8) and the
cross-validated training (stage 9) instantiate literally the same objects.  An
earlier version of this pipeline duplicated the definitions across two scripts,
which is how an architecture can drift between the run that selected its
hyper-parameters and the run that reports its accuracy.

The five models are chosen to span different inductive biases rather than
variations of one idea:

============================  ==============================================
``MLP``                       Plain feed-forward stack, the reference point.
``DeepResidualMLP``           Residual blocks with a dropout ramp.
``TabNet``                    Sequential attention over features.
``RF`` / ``LightGBM``         Tree ensembles: axis-aligned splits.
============================  ==============================================

The distinction that matters physically is dense-linear versus axis-aligned.
The features here are phasor quantities that are strongly correlated with one
another — the three phases of one PMU, and the same quantity at neighbouring
PMUs, carry largely overlapping information, and the discriminative signal
often lies in a *combination* such as a sequence component or an impedance
ratio. A dense layer forms those combinations directly; a decision tree has to
approximate them with a staircase of single-feature thresholds. This is the
mechanism behind the gap the paper reports between the two families.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

#: Activations selectable by the hyper-parameter search.
ACTIVATIONS = {"gelu": nn.GELU, "relu": nn.ReLU, "silu": nn.SiLU}


# ═══════════════════════════════════════════════════════════════════════════
#  DATA
# ═══════════════════════════════════════════════════════════════════════════
class FaultDataset(Dataset):
    """Feature matrix with optional multiplicative measurement noise.

    Noise emulates instrument-transformer and PMU accuracy limits.  Two choices
    are deliberate:

    * It is **multiplicative** — instrument accuracy is specified as a
      percentage of reading, not as an absolute volt or ampere figure, so the
      perturbation must scale with the measured value.
    * It is applied only to ``noise_columns``: voltages, currents and apparent
      powers, the quantities an instrument actually measures.  Angles, ratios
      and impedances are *derived*; perturbing them independently would
      produce a sample whose phasors and whose sequence components disagree,
      which no real instrument can produce.
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
#  ARCHITECTURES
# ═══════════════════════════════════════════════════════════════════════════
class MLPBaseline(nn.Module):
    """Feed-forward stack of ``Linear -> BatchNorm -> activation -> Dropout``.

    Widths halve at each layer (``h₀, h₀/2, h₀/4, …``), the classic funnel: the
    first layer does the mixing across the ~450 correlated inputs and the later
    ones compress towards the 122 classes.
    """

    def __init__(
        self,
        n_features: int,
        n_classes: int,
        hidden_dims: list[int],
        dropout: float,
        activation: str,
    ) -> None:
        super().__init__()
        act = ACTIVATIONS[activation]
        layers: list[nn.Module] = []
        prev = n_features
        for width in hidden_dims:
            layers += [
                nn.Linear(prev, width),
                nn.BatchNorm1d(width),
                act(),
                nn.Dropout(dropout),
            ]
            prev = width
        layers.append(nn.Linear(prev, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ResidualBlock(nn.Module):
    """Pre-activation residual block with a scaled skip connection.

    ``residual_scale`` below 1.0 damps the identity path.  With 122 classes and
    a deep stack, an undamped identity lets noise from the early layers reach
    the classification head unattenuated; the scale is exposed to the search
    rather than fixed, because its best value depends on depth.
    """

    def __init__(
        self, dim: int, dropout: float, activation: type[nn.Module], scale: float
    ) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            activation(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
        )
        self.activation = activation()
        self.scale = scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(x + self.scale * self.body(x))


class DeepResidualMLP(nn.Module):
    """Residual trunk at constant width, with a dropout ramp across depth.

    Dropout increases linearly from ``dropout_start`` in the first block to
    ``dropout_end`` in the last.  Early blocks still work with representations
    close to the raw phasors, where information is dense and dropping it costs
    signal; later blocks work with increasingly class-specific features, where
    stronger regularisation pays off.
    """

    def __init__(
        self,
        n_features: int,
        n_classes: int,
        hidden: int,
        n_blocks: int,
        dropout_start: float,
        dropout_end: float,
        activation: str,
        residual_scale: float,
    ) -> None:
        super().__init__()
        act = ACTIVATIONS[activation]
        rates = (
            np.linspace(dropout_start, dropout_end, n_blocks)
            if n_blocks > 1
            else [dropout_start]
        )
        self.stem = nn.Sequential(
            nn.Linear(n_features, hidden), nn.BatchNorm1d(hidden), act()
        )
        self.blocks = nn.Sequential(
            *[
                ResidualBlock(hidden, float(rates[i]), act, residual_scale)
                for i in range(n_blocks)
            ]
        )
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            act(),
            nn.Dropout(dropout_end),
            nn.Linear(hidden // 2, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.blocks(self.stem(x)))


def build_model(name: str, hp: dict, n_features: int, n_classes: int) -> nn.Module:
    """Instantiate a neural model from a hyper-parameter dictionary.

    The single construction path used by both stage 8 and stage 9, so a model
    cannot be built one way during the search and another way for the reported
    result.
    """
    if name == "MLP":
        widths = [hp["hidden_0"] // (2**i) for i in range(hp["n_layers"])]
        return MLPBaseline(
            n_features, n_classes, widths, hp["dropout"], hp["activation"]
        )
    if name == "DeepResidualMLP":
        return DeepResidualMLP(
            n_features,
            n_classes,
            hidden=hp["hidden"],
            n_blocks=hp["n_blocks"],
            dropout_start=hp["dropout_start"],
            dropout_end=hp["dropout_end"],
            activation=hp["activation"],
            residual_scale=hp["residual_scale"],
        )
    raise ValueError(f"{name!r} is not a PyTorch model built here.")


# ═══════════════════════════════════════════════════════════════════════════
#  LOSS, WEIGHTS, INITIALISATION
# ═══════════════════════════════════════════════════════════════════════════
class FocalLoss(nn.Module):
    """Focal loss with label smoothing and optional per-class weights.

    Three mechanisms, each addressing a distinct property of this problem:

    *Focal term* :math:`(1-p_t)^\\gamma` — down-weights samples the model
    already classifies confidently.  With 122 classes and a heavy tail of easy
    near-PMU buses, plain cross-entropy spends most of its gradient confirming
    what it already knows.

    *Label smoothing* — spreads a little probability mass over the other
    classes.  Beyond regularisation this is physically apt here: buses joined
    by a closed switch are electrically almost identical, so a confident
    prediction on the wrong member of such a pair should not be punished as
    harshly as one that picks a different lateral.

    *Class weights* — the fault-type mix makes buses on single-phase laterals
    far more frequent than three-phase trunk buses, since only line-to-ground
    faults can occur there.
    """

    def __init__(
        self,
        gamma: float,
        label_smoothing: float,
        n_classes: int,
        weights: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.n_classes = n_classes
        self.weights = weights

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
        if self.weights is not None:
            loss = loss * self.weights[target]
        return loss.mean()


def class_weights(y: np.ndarray, n_classes: int, mode: str) -> np.ndarray:
    """Per-class loss weights, normalised to mean 1.

    ``"sqrt"``
        :math:`1/\\sqrt{n_c}`.  The usual compromise: it removes most of the
        imbalance while keeping the gradient scale comparable across classes.
    ``"linear"``
        :math:`1/n_c`, full inverse frequency.  With a max/min class ratio in
        the hundreds this can make the rarest buses dominate the loss and
        destabilise training, so it is offered to the search rather than
        assumed.
    ``"uniform"``
        No reweighting.

    Classes absent from ``y`` are given the weight of a single sample rather
    than dividing by zero; they contribute nothing to the loss anyway.
    """
    counts = np.bincount(y, minlength=n_classes)
    safe = np.where(counts > 0, counts, 1)
    if mode == "sqrt":
        w = 1.0 / np.sqrt(safe)
    elif mode == "linear":
        w = 1.0 / safe
    elif mode == "uniform":
        w = np.ones(n_classes, dtype=float)
    else:
        raise ValueError(f"Unknown class-weight mode {mode!r}.")
    return w / w.sum() * n_classes


def init_kaiming(module: nn.Module) -> None:
    """He initialisation, matched to the ReLU-family activations used here.

    Applied to every neural model.  With the default PyTorch initialisation,
    deep stacks on this input dimensionality occasionally start on a plateau
    near the uniform prior and never leave it within the epoch budget.
    """
    if isinstance(module, nn.Linear):
        nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.BatchNorm1d):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)
