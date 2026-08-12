"""Shared loaders and utilities for the post-training analyses (stages 11-18).

Every analysis in this group reads the artefacts written by stage 9 and, where
it needs to re-run inference, reconstructs each fold exactly as that fold was
trained.  Centralising that reconstruction is the point of this module: the
analyses answer different questions, but they must all agree on which samples
belong to which fold, which scaler was fitted on which data, and which file
holds which model.  When those were duplicated across scripts they drifted, and
a drift here is invisible — the numbers still come out, they are simply wrong.

Two conventions are enforced here rather than left to each caller:

**Fold indices are global.**  ``fold_indices.pkl`` stores positions into the
full dataset.  The folds inside ``splits.pkl`` are local to the
cross-validation pool, and indexing the full dataset with them silently
selects the wrong rows — rows that overlap the pools reserved for placement and
hyper-parameter search.  :func:`load_fold_indices` reads the global file and
refuses stale artefacts.

**Predictions are read, never recomputed, when they already exist.**  Analyses
that only need ``preds``/``probs`` take them from ``fold_results.pkl``.  Only
the analyses that must perturb the inputs — robustness, permutation importance
— reload the models, and they do so through :func:`load_fold_model`, which
knows the canonical filenames.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

#: Display name used in results and figures -> key used in ``config.MODELS``.
DISPLAY_TO_KEY = {
    "Deep Residual MLP": "DeepResidualMLP",
    "MLP Baseline": "MLP",
    "TabNet": "TabNet",
    "Random Forest": "RF",
    "LightGBM": "LightGBM",
}
KEY_TO_DISPLAY = {v: k for k, v in DISPLAY_TO_KEY.items()}

#: Order used in every table and figure: gradient-based models first, then
#: trees, then the learning-free baselines.
MODEL_ORDER = [
    "Deep Residual MLP", "MLP Baseline", "TabNet",
    "Random Forest", "LightGBM",
    "Baseline Impedance", "Baseline Topological",
]

BASELINE_MODELS = ("Baseline Impedance", "Baseline Topological")


# ═══════════════════════════════════════════════════════════════════════════
#  ARTEFACT LOADING
# ═══════════════════════════════════════════════════════════════════════════
def load_preprocessing(cv_dir: Path) -> dict:
    """Load ``preprocessing_global.pkl`` and check it is the current schema.

    Raises
    ------
    SystemExit
        If the file is missing, or was written by a stage 9 old enough that its
        fold indices were local to the cross-validation pool.
    """
    path = Path(cv_dir) / "preprocessing_global.pkl"
    if not path.exists():
        raise SystemExit(
            f"{path} not found. Run scripts/09_cross_validate.py first."
        )
    with open(path, "rb") as fh:
        pre = pickle.load(fh)

    if pre.get("fold_index_space") != "global dataset positions":
        raise SystemExit(
            f"{path} was written by an older stage 9 whose fold indices were "
            "local to the cross-validation pool. Every analysis built on them "
            "would read the wrong samples. Re-run stage 9."
        )
    for key in ("groups", "noise_safe_columns"):
        if key not in pre:
            raise SystemExit(
                f"{path} has no '{key}'. Re-run stage 9 — the robustness and "
                "importance analyses need the feature grouping that was "
                "actually used during training."
            )
    return pre


def load_fold_indices(cv_dir: Path) -> dict[int, dict[str, np.ndarray]]:
    """Load the per-fold train / validation / test positions.

    These are **global** dataset positions.  Note that ``train`` excludes the
    internal validation slice, matching exactly what each model was fitted on —
    which is what any statistic computed from the training distribution (a
    mean used for missing-sensor imputation, for instance) must be based on.
    """
    path = Path(cv_dir) / "fold_indices.pkl"
    if not path.exists():
        raise SystemExit(
            f"{path} not found. Run scripts/09_cross_validate.py first."
        )
    with open(path, "rb") as fh:
        return pickle.load(fh)


def load_fold_results(
    cv_dir: Path, fold: int, baseline_dir: Path | None = None
) -> dict[str, dict]:
    """Predictions of every model on one fold's test set.

    Merges the trained models from ``cv/fold_<k>/fold_results.pkl`` with the
    classical baselines from ``baselines/fold_<k>_results.pkl`` when present,
    so downstream code sees one dictionary keyed by display name.
    """
    results: dict[str, dict] = {}

    path = Path(cv_dir) / f"fold_{fold}" / "fold_results.pkl"
    if path.exists():
        with open(path, "rb") as fh:
            results.update(pickle.load(fh))

    if baseline_dir is not None:
        path = Path(baseline_dir) / f"fold_{fold}_results.pkl"
        if path.exists():
            with open(path, "rb") as fh:
                results.update(pickle.load(fh))

    return results


def iter_folds(
    cv_dir: Path, n_folds: int, baseline_dir: Path | None = None
) -> Iterable[tuple[int, dict[str, dict]]]:
    """Yield ``(fold, results)`` for every fold that produced results."""
    for fold in range(n_folds):
        results = load_fold_results(cv_dir, fold, baseline_dir)
        if results:
            yield fold, results
        else:
            print(f"  fold {fold}: no results found, skipped")


def ordered_models(available: Iterable[str]) -> list[str]:
    """Sort model names into the canonical reporting order."""
    available = list(available)
    known = [m for m in MODEL_ORDER if m in available]
    return known + sorted(set(available) - set(known))


# ═══════════════════════════════════════════════════════════════════════════
#  FOLD RECONSTRUCTION
# ═══════════════════════════════════════════════════════════════════════════
def build_feature_matrix(df: pd.DataFrame, pre: dict) -> np.ndarray:
    """Rebuild the raw (unscaled) feature matrix in the training column order.

    The categorical indicators are one-hot encoded against the PMU count
    recorded by stage 9, not against the values present in this dataframe.
    Inferring the count from the data would silently produce a different number
    of columns whenever some PMU never happens to be the most-affected one.

    Raises
    ------
    SystemExit
        If the resulting columns do not match ``feature_names`` exactly.
    """
    from config import CATEGORICAL_FEATURES, TARGET_COL
    from src.scaling import one_hot_categoricals

    features = df.drop(columns=[TARGET_COL])
    features = one_hot_categoricals(features, CATEGORICAL_FEATURES, pre["n_pmus"])

    expected = list(pre["feature_names"])
    missing = [c for c in expected if c not in features.columns]
    extra = [c for c in features.columns if c not in expected]
    if missing or extra:
        raise SystemExit(
            "Feature columns do not match the training schema.\n"
            f"  missing: {missing[:5]}\n  unexpected: {extra[:5]}\n"
            "The dataset is not the one stage 9 was run on."
        )
    return features[expected].to_numpy(dtype=np.float32)


def normalize(raw: np.ndarray, scalers: dict, pre: dict) -> np.ndarray:
    """Apply a fold's fitted scalers, then the same clip used in training.

    Group membership comes from ``pre["groups"]`` — the grouping stage 9
    actually used — so a later edit to ``config.FEATURE_GROUPS`` cannot
    retroactively change how a saved model's inputs are transformed.
    """
    out = np.asarray(raw, dtype=np.float32).copy()
    for name, idx in pre["groups"].items():
        if not idx or name not in scalers:
            continue
        out[:, idx] = scalers[name].transform(raw[:, idx]).astype(np.float32)
    lo, hi = pre.get("clip_range", (-20.0, 20.0))
    return np.clip(out, lo, hi, out=out)


def load_scalers(cv_dir: Path, fold: int) -> dict:
    """Load the scalers fitted on one fold's training portion."""
    path = Path(cv_dir) / f"fold_{fold}" / "scalers.pkl"
    if not path.exists():
        raise SystemExit(f"{path} not found. Re-run stage 9 for this fold.")
    with open(path, "rb") as fh:
        return pickle.load(fh)


def load_fold_model(
    display_name: str, cv_dir: Path, fold: int, pre: dict, device
) -> tuple[str, Any]:
    """Load one trained model, returning ``(kind, model)``.

    ``kind`` is ``"torch"`` for the MLP family, ``"tabnet"`` for TabNet and
    ``"sklearn"`` for the tree ensembles.  Callers need it because the three
    families expect different inputs: the trees take the **raw** matrix (see
    ``config.UNSCALED_MODELS``) while the others take the normalised one.
    """
    import torch

    from config import MODELS
    from src.models import build_model

    key = DISPLAY_TO_KEY.get(display_name, display_name)
    if key not in MODELS:
        raise ValueError(f"Unknown model {display_name!r}.")

    fold_dir = Path(cv_dir) / f"fold_{fold}"
    hp = pre["best_hp"][key]

    if key in ("MLP", "DeepResidualMLP"):
        path = fold_dir / f"{key}.pt"
        if not path.exists():
            raise FileNotFoundError(path)
        model = build_model(key, hp, pre["n_features"], pre["n_classes"])
        model.load_state_dict(torch.load(path, map_location=device))
        model.to(device).eval()
        return "torch", model

    if key == "TabNet":
        from pytorch_tabnet.tab_model import TabNetClassifier

        path = fold_dir / "TabNet.zip"
        if not path.exists():
            raise FileNotFoundError(path)
        model = TabNetClassifier()
        model.load_model(str(path))
        return "tabnet", model

    path = fold_dir / f"{key}.pkl"
    if not path.exists():
        raise FileNotFoundError(path)
    with open(path, "rb") as fh:
        return "sklearn", pickle.load(fh)


def model_input(kind: str, display_name: str, raw: np.ndarray,
                scalers: dict, pre: dict) -> np.ndarray:
    """Feature matrix in the representation a given model expects."""
    from config import UNSCALED_MODELS

    key = DISPLAY_TO_KEY.get(display_name, display_name)
    if key in UNSCALED_MODELS:
        return raw
    return normalize(raw, scalers, pre)


def predict_probs(kind: str, model: Any, x: np.ndarray, device,
                  batch_size: int = 4096) -> np.ndarray:
    """Class probabilities, batched for the torch models."""
    import torch

    if kind == "torch":
        out = []
        with torch.no_grad():
            for i in range(0, len(x), batch_size):
                batch = torch.as_tensor(x[i:i + batch_size], dtype=torch.float32)
                logits = model(batch.to(device))
                out.append(torch.softmax(logits, dim=1).cpu().numpy())
        return np.vstack(out)
    return model.predict_proba(x)


# ═══════════════════════════════════════════════════════════════════════════
#  FEATURE INDEXING
# ═══════════════════════════════════════════════════════════════════════════
def pmu_column_indices(
    feature_names: Sequence[str], pmus: Sequence[str]
) -> dict[str, list[int]]:
    """``{pmu: [column indices]}`` for the per-PMU features.

    Uses :func:`src.features.extract_pmu_id` rather than substring matching.
    A substring test cannot distinguish the PMU identifier from a digit that
    happens to appear elsewhere in the name, and the one-hot columns
    ``pmu_max_dI_<k>`` end in a digit that is a *position*, not a bus.
    Those are excluded here and handled separately, because ablating a PMU
    changes them by recomputation rather than by removal.
    """
    from src.features import extract_pmu_id

    out: dict[str, list[int]] = {p: [] for p in pmus}
    for i, name in enumerate(feature_names):
        if name.startswith(("pmu_max_dI", "pmu_max_dV")):
            continue
        pmu = extract_pmu_id(name)
        if pmu in out:
            out[pmu].append(i)
    return out


def magnitude_indices(
    feature_names: Sequence[str], groups: dict[str, list[int]],
    pmus: Sequence[str], magnitude_groups: Sequence[str],
) -> dict[str, list[int]]:
    """Per-PMU indices restricted to the directly measured magnitudes.

    Perturbations representing instrument error apply only to these: voltage,
    current and apparent-power magnitudes.  Angles, ratios, impedances and
    sequence components are derived, and perturbing them independently would
    produce a sample whose parts contradict each other.
    """
    magnitude = {i for g in magnitude_groups for i in groups.get(g, [])}
    per_pmu = pmu_column_indices(feature_names, pmus)
    return {p: [i for i in idx if i in magnitude] for p, idx in per_pmu.items()}


def global_feature_index(
    feature_names: Sequence[str], pmus: Sequence[str], phases: Sequence[str]
) -> dict:
    """Column map needed to recompute the aggregate features after an ablation.

    ``severity``, ``pmu_max_dI`` and ``pmu_max_dV`` are defined over the set of
    PMUs.  When a PMU's telemetry is lost, they must be **re-derived from the
    survivors**: the dead sensor can no longer win the argmax, nor contribute
    to the mean.  Merely blanking its own columns would leave those three
    features still reporting it, which is not what a real outage looks like.

    Returns the index of ``severity``, the ``dfI``/``dfV`` columns grouped by
    PMU position, and the one-hot columns keyed by the position they encode.
    """
    position = {name: i for i, name in enumerate(feature_names)}
    df_i: dict[int, list[int]] = {}
    df_v: dict[int, list[int]] = {}
    for pos, pmu in enumerate(pmus):
        df_i[pos] = [position[f"dfI_{pmu}_{ph}"] for ph in phases
                     if f"dfI_{pmu}_{ph}" in position]
        df_v[pos] = [position[f"dfV_{pmu}_{ph}"] for ph in phases
                     if f"dfV_{pmu}_{ph}" in position]

    onehot_i, onehot_v = {}, {}
    for i, name in enumerate(feature_names):
        for prefix, target in (("pmu_max_dI_", onehot_i), ("pmu_max_dV_", onehot_v)):
            if name.startswith(prefix):
                suffix = name[len(prefix):]
                if suffix.isdigit():
                    target[int(suffix)] = i

    return {
        "severity": position.get("severity"),
        "dfI": df_i, "dfV": df_v,
        "onehot_dI": onehot_i, "onehot_dV": onehot_v,
        "n_pmus": len(pmus),
    }


def recompute_aggregates(raw: np.ndarray, ablated: int, index: dict) -> None:
    """Re-derive ``severity`` and the argmax indicators from the surviving PMUs.

    Operates in place on the **raw** matrix, and must be called before
    normalisation — the scalers were fitted on raw quantities.
    """
    survivors = [j for j in range(index["n_pmus"]) if j != ablated]

    if index["severity"] is not None:
        columns = [c for j in survivors for c in index["dfI"].get(j, [])]
        if columns:
            raw[:, index["severity"]] = raw[:, columns].mean(axis=1)

    for key, onehot in (("dfI", "onehot_dI"), ("dfV", "onehot_dV")):
        positions, totals = [], []
        for j in survivors:
            columns = index[key].get(j, [])
            if columns:
                positions.append(j)
                totals.append(raw[:, columns].sum(axis=1))
        if not totals:
            continue
        winner = np.asarray(positions)[np.stack(totals, axis=1).argmax(axis=1)]
        for pos, column in index[onehot].items():
            raw[:, column] = (winner == pos).astype(raw.dtype)


# ═══════════════════════════════════════════════════════════════════════════
#  METRICS
# ═══════════════════════════════════════════════════════════════════════════
def mean_std(values: Sequence[float]) -> tuple[float, float]:
    """Mean and standard deviation, ignoring NaN; ``(nan, nan)`` if empty."""
    clean = [v for v in values if v is not None and not np.isnan(v)]
    if not clean:
        return float("nan"), float("nan")
    return float(np.mean(clean)), float(np.std(clean))


class Report:
    """Accumulates lines for both the console and a text file."""

    def __init__(self) -> None:
        self.lines: list[str] = []

    def __call__(self, text: str = "") -> None:
        print(text)
        self.lines.append(text)

    def save(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(self.lines), encoding="utf-8")
        print(f"\n  Saved: {path}")
        return path


# ═══════════════════════════════════════════════════════════════════════════
#  PHYSICALLY CONSISTENT MEASUREMENT ERROR
# ═══════════════════════════════════════════════════════════════════════════
#: The four phasors a PMU actually measures, as ``(magnitude, angle)`` column
#: prefixes, grouped by the instrument chain that produces them.  Everything
#: else in the feature vector — superimposed quantities, ratios, apparent
#: impedances, sequence components, powers — is *computed* from these.
MEASURED_PHASORS = {
    "voltage": (("Vb_", "AngVb_"), ("Vf_", "AngVf_")),
    "current": (("Ib_", "AngIb_"), ("If_", "AngIf_")),
}


def raw_measurement_frame(df: "pd.DataFrame", target_col: str) -> "pd.DataFrame":
    """Strip a feature matrix back to the columns a PMU measures, plus the target.

    Needed before re-deriving features: :func:`src.features.build_features`
    concatenates its output onto the frame it is given, so passing a frame that
    already contains derived columns would produce duplicates.
    """
    from src.features import MEASUREMENT_PREFIXES

    keep = [c for c in df.columns if c.startswith(MEASUREMENT_PREFIXES)]
    return df[keep + [target_col]].copy()


def apply_tve(
    frame: "pd.DataFrame",
    pmus: Sequence[str],
    phases: Sequence[str],
    sigma: float,
    rng,
    mode: str = "independent",
    quantities: Sequence[str] = ("voltage", "current"),
) -> "pd.DataFrame":
    """Perturb the measured phasors with a complex multiplicative error.

    The perturbation is applied to the phasor itself,

    .. math:: \\tilde X' = \\tilde X\\,(1 + \\varepsilon), \\qquad
              \\varepsilon = \\varepsilon_r + j\\varepsilon_i,\\;
              \\varepsilon_r, \\varepsilon_i \\sim
              \\mathcal N\\!\\left(0, \\sigma^2/2\\right),

    so that :math:`\\mathbb E|\\varepsilon|^2 = \\sigma^2` and

    .. math:: \\frac{|\\tilde X' - \\tilde X|}{|\\tilde X|} = |\\varepsilon|.

    That ratio is precisely the **total vector error** of IEC/IEEE 60255-118-1, the
    quantity by which PMU accuracy is specified (the standard's compliance
    limit is 1 % TVE).  ``sigma`` is therefore not an abstract noise level: it
    is an instrument specification, and a single draw perturbs magnitude and
    angle jointly and consistently rather than treating them as independent.

    Modes differ only in how :math:`\\varepsilon` is shared:

    ``independent``
        One draw per phasor.  Errors average out across channels.
    ``correlated``
        Half the variance shared across all channels of one PMU, half
        independent — a common-mode error such as a reference or timing drift.
    ``bias``
        One draw per PMU, constant across all samples: a calibration error,
        which does not average out over repeated events.

    ``quantities`` restricts the perturbation to one instrument chain, which is
    what isolates the voltage transformers from the current transformers.
    """
    if sigma <= 0:
        return frame

    out = frame.copy()
    half = sigma / np.sqrt(2.0)
    n = len(out)

    def draw(shape) -> np.ndarray:
        return (rng.normal(0.0, half, size=shape)
                + 1j * rng.normal(0.0, half, size=shape))

    for pmu in pmus:
        common = draw(1)[0] if mode in ("correlated", "bias") else 0.0
        for quantity in quantities:
            for mag_prefix, ang_prefix in MEASURED_PHASORS[quantity]:
                for phase in phases:
                    mag_col = f"{mag_prefix}{pmu}_{phase}"
                    ang_col = f"{ang_prefix}{pmu}_{phase}"
                    if mag_col not in out.columns:
                        continue

                    magnitude = out[mag_col].to_numpy(dtype=float)
                    angle = np.deg2rad(out[ang_col].to_numpy(dtype=float))
                    phasor = magnitude * np.exp(1j * angle)

                    if mode == "independent":
                        epsilon = draw(n)
                    elif mode == "correlated":
                        # Two independent halves of the variance sum to sigma.
                        epsilon = common + draw(n)
                    elif mode == "bias":
                        # One offset per PMU, identical for every sample.
                        epsilon = common
                    else:
                        raise ValueError(f"Unknown noise mode {mode!r}.")

                    perturbed = phasor * (1.0 + epsilon)
                    out[mag_col] = np.abs(perturbed)
                    out[ang_col] = np.rad2deg(np.angle(perturbed))

    return out


def rebuild_features(
    frame: "pd.DataFrame", pmus: Sequence[str], pre: dict
) -> np.ndarray:
    """Re-derive the full feature matrix from (possibly perturbed) phasors.

    This is what makes a perturbation physically meaningful: every superimposed
    quantity, ratio, apparent impedance, sequence component, power and
    cross-PMU aggregate is recomputed from the perturbed phasors using the same
    code that built the training data.  Perturbing the derived columns directly
    instead would describe a measurement no instrument can produce — and would
    understate the error on the superimposed quantities, which are differences
    of large numbers and therefore amplify relative error by roughly
    :math:`|X| / |\\Delta X|`.
    """
    from config import EPS, PHASES
    from src.features import build_features

    derived = build_features(frame, pmus, eps=EPS, phases=PHASES, verbose=False)
    return build_feature_matrix(derived, pre)


# ═══════════════════════════════════════════════════════════════════════════
#  POOLED PER-BUS STATISTICS
# ═══════════════════════════════════════════════════════════════════════════
def switch_partner(bus: str) -> str | None:
    """The bus on the other side of a normally-closed switch, if any.

    These pairs are separated by a few milliohms, so no measurement-based
    method can distinguish them.  They dominate the tail of the per-bus recall
    and are highlighted wherever that tail is reported.
    """
    from config import SWITCH_PAIRS

    for a, b in SWITCH_PAIRS:
        if bus == a:
            return b
        if bus == b:
            return a
    return None


def pooled_predictions(
    cv_dir: Path, model: str, n_folds: int, baseline_dir: Path | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Predictions and labels for one model, pooled across every fold.

    Pooling is sound here because the fold test sets are disjoint: their union
    is each sample predicted exactly once, by the model that did not see it.
    """
    preds, truth = [], []
    for _, results in iter_folds(cv_dir, n_folds, baseline_dir):
        if model not in results:
            continue
        preds.append(np.asarray(results[model]["preds"]))
        truth.append(np.asarray(results[model]["y_true"]))
    if not preds:
        return np.array([], dtype=int), np.array([], dtype=int)
    return np.concatenate(preds), np.concatenate(truth)


def per_bus_recall(
    cv_dir: Path, model: str, classes: Sequence[str], n_folds: int
) -> "pd.DataFrame":
    """Per-bus recall: Top-1 accuracy conditioned on the true class.

    Recall rather than precision, because the operational question is "when a
    fault happens *here*, is it found?" — not "when the model says here, is it
    right?".

    Buses with no test sample get ``NaN`` rather than zero: never having been
    evaluated is not the same as never being found, and zero would drag the
    reported mean down with data that does not exist.

    Returned columns: ``bus``, ``recall``, ``support``, ``switch_partner``.
    Defined here rather than in a script so that the figure and the prose
    report the same numbers by construction instead of by convention.
    """
    preds, truth = pooled_predictions(cv_dir, model, n_folds)
    n_classes = len(classes)

    if preds.size == 0:
        return pd.DataFrame({
            "bus": list(classes),
            "recall": [float("nan")] * n_classes,
            "support": [0] * n_classes,
            "switch_partner": [switch_partner(str(b)) for b in classes],
        })

    support = np.bincount(truth, minlength=n_classes)
    hits = np.bincount(truth[preds == truth], minlength=n_classes)
    with np.errstate(invalid="ignore", divide="ignore"):
        recall = np.where(support > 0, hits / np.maximum(support, 1), np.nan)

    return pd.DataFrame({
        "bus": [str(b) for b in classes],
        "recall": recall,
        "support": support,
        "switch_partner": [switch_partner(str(b)) for b in classes],
    })
