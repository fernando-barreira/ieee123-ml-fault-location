"""Group-wise normalisation of the feature matrix.

The feature matrix mixes quantities that differ by many orders of magnitude and
that have very different distributions: voltages of a few thousand volts,
currents of a few hundred amperes, apparent powers in the megavolt-ampere
range, trigonometric features bounded in [-1, 1], and apparent impedances whose
tail is unbounded.  Fitting one scaler over all of them would let the
heavy-tailed columns dictate the scale of everything else.

Each group defined in :data:`config.FEATURE_GROUPS` therefore gets its own
scaler:

* ``RobustScaler`` (median / inter-quartile range) for **ratios** and
  **apparent impedances**.  ``Z_est = |ΔV| / |ΔI|`` explodes whenever the
  superimposed current at a distant PMU is near zero, which happens routinely
  for high-resistance faults.  One such sample would move the mean and inflate
  the standard deviation enough to squash every normal sample towards zero.
  The median and IQR ignore it.
* ``StandardScaler`` for everything else.

After scaling, values are clipped to :data:`config.CLIP_RANGE`.  This bounds
the residual outliers that survive the robust scaler without discarding the
samples themselves — a high-resistance fault is a *hard* sample, not a corrupt
one, and dropping it would bias the reported accuracy upwards.

Two rules that this module exists to enforce
--------------------------------------------
1. **Scalers are fitted on the training split only.**  Fitting on the full
   dataset leaks the test distribution into the transform.
2. **Categorical columns are one-hot encoded before scaling.**  ``pmu_max_dI``
   holds a PMU index; standardising it would assert that PMU 3 is "larger" than
   PMU 1.  See :func:`one_hot_categoricals`.

Tree ensembles do not use any of this.  Decision trees split on thresholds, so
a per-feature affine transform leaves them unchanged — but the post-scaling
clip is *not* affine, and would collapse every extreme apparent-impedance value
onto one threshold.  ``config.UNSCALED_MODELS`` lists the models that therefore
receive the raw matrix, in the hyper-parameter search and in the final training
alike.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler, StandardScaler


def one_hot_categoricals(
    df: pd.DataFrame,
    categorical: Sequence[str],
    n_categories: int,
) -> pd.DataFrame:
    """Replace index-valued columns with one-hot indicator columns.

    ``pmu_max_dI`` becomes ``pmu_max_dI_0 ... pmu_max_dI_{K-1}``.  The number of
    categories is passed explicitly rather than inferred, so that a split in
    which some PMU never happens to be the maximum still produces the full set
    of columns and every split has an identical schema.

    .. warning::
       The generated names end in ``_<digit>``.  Any code that identifies a
       column's PMU from its suffix must exclude these columns explicitly —
       ``pmu_max_dI_1`` refers to *position 1 in the PMU list*, not to bus 1.
    """
    out = df.copy()
    for col in categorical:
        if col not in out.columns:
            continue
        values = out[col].to_numpy(dtype=int)
        for k in range(n_categories):
            out[f"{col}_{k}"] = (values == k).astype(np.float32)
        out = out.drop(columns=[col])
    return out


def group_indices(
    feature_names: Sequence[str],
    groups: dict[str, tuple[str, ...]],
    exclude_prefixes: Sequence[str] = (),
) -> tuple[dict[str, list[int]], list[int]]:
    """Map each feature group to the column positions it covers.

    Columns matching none of the groups are collected into a ``"rest"`` group
    so that no column ever escapes normalisation.

    Parameters
    ----------
    exclude_prefixes
        Columns starting with any of these are left untouched — used for the
        one-hot indicator columns, which are already in [0, 1].

    Returns
    -------
    (groups, noise_safe_indices)
        ``noise_safe_indices`` lists the columns on which measurement noise may
        be injected.  It is matched against :data:`config.NOISE_SAFE_PREFIXES`
        directly, **not** taken from the groups above: which scaler a column
        needs and whether an instrument measures it independently are separate
        questions.  In particular the sequence magnitudes share a scaler group
        with the phase quantities but are exact linear functions of them, so
        perturbing both would produce physically inconsistent samples.
    """
    from config import NOISE_SAFE_PREFIXES

    def match(prefixes: tuple[str, ...]) -> list[int]:
        return [
            i
            for i, name in enumerate(feature_names)
            if any(name.startswith(p) for p in prefixes)
            and not any(name.startswith(x) for x in exclude_prefixes)
        ]

    resolved = {name: match(prefixes) for name, prefixes in groups.items()}

    claimed = {i for idx in resolved.values() for i in idx}
    excluded = {
        i
        for i, name in enumerate(feature_names)
        if any(name.startswith(x) for x in exclude_prefixes)
    }
    rest = sorted(set(range(len(feature_names))) - claimed - excluded)
    if rest:
        resolved["rest"] = rest

    return resolved, match(NOISE_SAFE_PREFIXES)


def fit_transform_groupwise(
    train: pd.DataFrame,
    *others: pd.DataFrame,
) -> tuple[np.ndarray, ...]:
    """Fit scalers on ``train`` and apply them to ``train`` and ``others``.

    Returns
    -------
    tuple of numpy.ndarray
        Scaled matrices in the same order as the inputs, ``float32``, followed
        by the list of noise-safe column indices as the last element.

    Notes
    -----
    ``float32`` is used because the downstream models are trained in
    ``float32``/``bfloat16``; keeping ``float64`` here would only double memory
    traffic for the 75 000-row cross-validation pool.
    """
    from config import (
        CATEGORICAL_FEATURES,
        CLIP_RANGE,
        FEATURE_GROUPS,
        ROBUST_SCALER_GROUPS,
    )

    names = list(train.columns)
    exclude = tuple(f"{c}_" for c in CATEGORICAL_FEATURES)
    groups, noise_safe = group_indices(names, FEATURE_GROUPS, exclude)

    matrices = [
        frame.to_numpy(dtype=np.float32, copy=True) for frame in (train, *others)
    ]

    for group_name, idx in groups.items():
        if not idx:
            continue
        scaler = (
            RobustScaler()
            if group_name in ROBUST_SCALER_GROUPS
            else StandardScaler()
        )
        matrices[0][:, idx] = scaler.fit_transform(matrices[0][:, idx]).astype(
            np.float32
        )
        for m in matrices[1:]:
            m[:, idx] = scaler.transform(m[:, idx]).astype(np.float32)

    lo, hi = CLIP_RANGE
    for m in matrices:
        np.clip(m, lo, hi, out=m)

    return (*matrices, noise_safe)


def infer_n_pmus(columns: Sequence[str]) -> int:
    """Number of PMUs represented in a feature matrix.

    Counted from the distinct PMU identifiers appearing in the column names.
    This is the value the ``pmu_max_*`` indicators must be one-hot encoded
    against, and getting it wrong is a silent, damaging failure: the encoding
    would produce a different number of columns per fold, so the folds would no
    longer share a schema.

    Two tempting shortcuts are both unsafe and are avoided here:

    * *Parsing the filename.* A rename breaks it, and the fallback then applies
      without warning.
    * *Taking* ``max(pmu_max_dI) + 1``. If some PMU never happens to be the
      most-affected one — entirely possible for a PMU far from every fault, and
      more likely in a small fold — that PMU's indicator column silently
      disappears.

    Raises
    ------
    ValueError
        If no per-PMU columns are found.
    """
    from src.features import extract_pmu_id

    categorical_prefixes = ("pmu_max_dI", "pmu_max_dV")
    pmus = {
        extract_pmu_id(col)
        for col in columns
        if not col.startswith(categorical_prefixes)
    }
    pmus.discard(None)
    if not pmus:
        raise ValueError(
            "No per-PMU feature columns found; cannot determine the number of "
            "PMUs for one-hot encoding."
        )
    return len(pmus)


def prepare_matrix(
    df: "pd.DataFrame", target_col: str
) -> tuple["pd.DataFrame", np.ndarray, int]:
    """Split off the target and one-hot encode the categorical indicators.

    Returns
    -------
    (features, targets, n_pmus)
        ``features`` still holds raw physical units — nothing is scaled here,
        because scalers must be fitted per fold on training data only.
    """
    from config import CATEGORICAL_FEATURES

    features = df.drop(columns=[target_col])
    targets = df[target_col].to_numpy()
    n_pmus = infer_n_pmus(list(features.columns))
    features = one_hot_categoricals(features, CATEGORICAL_FEATURES, n_pmus)
    return features, targets, n_pmus


def scale_splits(
    train: np.ndarray,
    others: "list[np.ndarray]",
    feature_names: Sequence[str],
) -> tuple[np.ndarray, "list[np.ndarray]", dict, "list[int]"]:
    """Fit group-wise scalers on ``train`` and apply them to every split.

    Returns the scaled training matrix, the scaled other matrices, the fitted
    scalers (persisted per fold so a saved model can be applied to new data),
    and the indices of the noise-safe columns.
    """
    from config import CATEGORICAL_FEATURES, CLIP_RANGE, FEATURE_GROUPS, ROBUST_SCALER_GROUPS

    exclude = tuple(f"{c}_" for c in CATEGORICAL_FEATURES)
    groups, noise_safe = group_indices(list(feature_names), FEATURE_GROUPS, exclude)

    matrices = [np.asarray(train, dtype=np.float32).copy()]
    matrices += [np.asarray(m, dtype=np.float32).copy() for m in others]

    scalers: dict = {}
    for name, idx in groups.items():
        if not idx:
            continue
        scaler = (
            RobustScaler() if name in ROBUST_SCALER_GROUPS else StandardScaler()
        )
        matrices[0][:, idx] = scaler.fit_transform(matrices[0][:, idx]).astype(
            np.float32
        )
        for m in matrices[1:]:
            m[:, idx] = scaler.transform(m[:, idx]).astype(np.float32)
        scalers[name] = scaler

    lo, hi = CLIP_RANGE
    for m in matrices:
        np.clip(m, lo, hi, out=m)

    return matrices[0], matrices[1:], scalers, noise_safe