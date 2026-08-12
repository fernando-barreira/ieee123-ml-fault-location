"""Central configuration for the IEEE 123-bus ML fault-location pipeline.

Every path, physical constant and experimental hyper-parameter used by the
pipeline lives here.  Scripts import from this module instead of hard-coding
values, so a single edit propagates to the whole pipeline.

Paths
-----
All paths are resolved relative to the repository root, so the code runs
unchanged on Windows, Linux and macOS.  Two locations depend on the user's
machine and can be overridden with environment variables:

===========================  ==================================================
``IEEE123_DSS_MASTER``       Full path to ``IEEE123Master.dss``.  The OpenDSS
                             test cases are redistributed under their own
                             licence and are therefore *not* vendored here.
``IEEE123_DATA_DIR``         Root for generated data (default ``./data``).
                             Useful when the ~2 GB raw dataset lives on
                             another drive.
===========================  ==================================================

Physical conventions used throughout the pipeline
-------------------------------------------------
* Voltages are OpenDSS bus voltages: **magnitude in volts, line-to-neutral**,
  angle in **degrees**, phase order ``(1, 2, 3) = (A, B, C)``.
* Currents are **branch** currents in amperes, angle in degrees, taken at the
  terminal of the monitored line that is connected to the PMU bus.  OpenDSS
  signs currents as *flowing into the element* at that terminal.
* Fault resistances ``Rf`` are in ohms and are applied per faulted phase.
* The feeder is a 4.16 kV line-to-line, 4-wire multi-grounded radial system.
"""

from __future__ import annotations

import os
from pathlib import Path

# ═══════════════════════════════════════════════════════════════════════════
#  1. PATHS
# ═══════════════════════════════════════════════════════════════════════════
REPO_ROOT = Path(__file__).resolve().parent

#: OpenDSS master file of the IEEE 123-bus test feeder.
#: Ships with the official OpenDSS installer under
#: ``IEEETestCases/123Bus/IEEE123Master.dss``.
DSS_MASTER = Path(
    os.environ.get(
        "IEEE123_DSS_MASTER",
        REPO_ROOT / "network" / "dss" / "IEEE123Master.dss",
    )
)

DATA_DIR = Path(os.environ.get("IEEE123_DATA_DIR", REPO_ROOT / "data"))
RAW_DIR = DATA_DIR / "raw"          # OpenDSS output, one row per simulation
INTERIM_DIR = DATA_DIR / "interim"  # graph, candidate set, splits
PROCESSED_DIR = DATA_DIR / "processed"  # engineered feature matrices
RESULTS_DIR = REPO_ROOT / "results"

# --- named artefacts -------------------------------------------------------
GRAPH_PKL = INTERIM_DIR / "graph_ieee123.gpickle"
CANDIDATES_PKL = INTERIM_DIR / "pmu_candidates.pkl"
SPLITS_PKL = INTERIM_DIR / "splits.pkl"
SPLITS_SUMMARY_CSV = INTERIM_DIR / "splits_summary.csv"

RAW_DATASET_CSV = RAW_DIR / "measurements_candidate_pmus.csv"
FEATURES_ALL_CSV = PROCESSED_DIR / "features_candidate_pmus.csv"


def features_csv(k: int) -> Path:
    """Path of the feature matrix restricted to a ``k``-PMU subset."""
    return PROCESSED_DIR / f"features_{k}pmu.csv"


FSNR_DIR = RESULTS_DIR / "fsnr"


def resolve_path(value: str | Path) -> Path:
    """Turn a command-line path into an absolute one.

    Relative paths are resolved against :data:`REPO_ROOT`, **not** against the
    current working directory.  ``--out data/raw/smoke.csv`` therefore means the
    same file whether the script is launched from the repository root, from
    ``scripts/`` or from an IDE with its own working directory.

    Absolute paths are returned untouched, so pointing at another drive still
    works.
    """
    path = Path(value).expanduser()
    return path if path.is_absolute() else (REPO_ROOT / path).resolve()


def ensure_dir(path: Path) -> Path:
    """Create ``path`` if needed and verify it can actually be written to.

    Directories are created here rather than at import time so that a failure
    is reported by the script that needs the directory, with the absolute path
    that failed — and so that merely importing :mod:`config` has no side
    effects on the file system.

    The write probe matters more than it looks: a long stage should fail in the
    first second, not after the setup phase.  On Windows the usual causes of a
    late failure are OneDrive folder protection, Windows Defender Controlled
    Folder Access, or a read-only attribute inherited from a copied folder —
    none of which show up until something is actually written.

    Raises
    ------
    SystemExit
        With an actionable message naming the absolute path and the likely
        causes.
    """
    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise SystemExit(
            f"Cannot create directory:\n  {path}\n"
            f"  {type(exc).__name__}: {exc}\n\n"
            "Common causes on Windows: the folder is synced by OneDrive, "
            "Windows Defender Controlled Folder Access is blocking it, or the "
            "parent folder is read-only.\n"
            "You can also place the data elsewhere by setting the "
            "IEEE123_DATA_DIR environment variable, e.g.\n"
            '  $env:IEEE123_DATA_DIR = "D:\\ic_data"'
        ) from exc

    probe = path / ".write_probe"
    try:
        probe.touch()
        probe.unlink()
    except OSError as exc:
        raise SystemExit(
            f"Directory exists but is not writable:\n  {path}\n"
            f"  {type(exc).__name__}: {exc}\n\n"
            "Set IEEE123_DATA_DIR to a writable location, e.g.\n"
            '  $env:IEEE123_DATA_DIR = "D:\\ic_data"'
        ) from exc
    return path


def ensure_parent(path: Path) -> Path:
    """Make sure the directory holding ``path`` exists and is writable."""
    ensure_dir(path.parent)
    return path


def ensure_writable_file(path: Path) -> Path:
    """Verify that ``path`` itself can be written, not just its directory.

    A writable directory does not imply a writable file: an existing file can
    carry the read-only attribute, be locked by an editor or a sync client, or
    have been copied from a source that was read-only — Windows preserves the
    attribute on copy.  Checking the directory alone misses all of these.

    This matters most for the consolidated outputs of long stages, which are
    written last.  Discovering the problem then costs the whole run.
    """
    ensure_dir(path.parent)
    if not path.exists():
        return path

    try:
        with open(path, "a"):
            pass
    except OSError as exc:
        raise SystemExit(
            f"File exists but cannot be written:\n  {path}\n"
            f"  {type(exc).__name__}: {exc}\n\n"
            "Common causes: the read-only attribute (often inherited when the "
            "file was copied), the file being open in an editor, or a sync "
            "client holding a lock.\n"
            "On Windows, clear the attribute with:\n"
            f'  attrib -R "{path}"'
        ) from exc
    return path


def write_json_atomic(path: Path, payload: object) -> Path:
    """Write JSON via a temporary file, then replace the target.

    Guarantees the destination is never left half-written if the process dies
    mid-write.  If the replacement fails — typically because the destination is
    locked — the temporary file is kept and its path returned, so the content is
    still recoverable rather than lost.
    """
    import json

    ensure_dir(path.parent)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "w") as fh:
        json.dump(payload, fh, indent=2, default=str)

    try:
        temporary.replace(path)
        return path
    except OSError as exc:
        print(
            f"WARNING: could not replace {path} ({type(exc).__name__}: {exc}).\n"
            f"         The content was written to {temporary} instead; rename "
            "it once the destination is free."
        )
        return temporary


# ═══════════════════════════════════════════════════════════════════════════
#  2. REPRODUCIBILITY
# ═══════════════════════════════════════════════════════════════════════════
SEED = 42


# ═══════════════════════════════════════════════════════════════════════════
#  3. FEEDER PHYSICS (immutable properties of the IEEE 123-bus test case)
# ═══════════════════════════════════════════════════════════════════════════
#: Nominal line-to-line voltage of the feeder [kV].
VBASE_LL_KV = 4.16

#: Bus that the voltage source is connected to.  Everything downstream of it
#: is the feeder proper.
SOURCE_BUS = "150"

#: Feeder head: first bus downstream of the substation voltage regulator.
#: A "single PMU at the substation" measures **here**, because this is where
#: the whole feeder current is observable.  Bus ``1`` is *not* the substation;
#: it is the first load bus one span downstream of ``149``.
SUBSTATION_BUS = "149"

#: Buses that exist in the ``.dss`` file only to model a device (regulator
#: windings, open-switch stubs) and have no independent electrical identity.
#: Mapping to ``None`` removes the node from the topology graph; mapping to
#: another name merges it into that node.
#:
#: * ``150`` / ``150r``  – source-side and regulated-side terminals of the
#:   substation regulator.  Removed: the source itself is not a fault location.
#: * ``9r``, ``25r``     – regulated terminals of the line regulators; merged
#:   into their parent buses ``9`` and ``25``.
#: * ``160`` / ``160r``  – terminals of the regulator on the 60-67 branch;
#:   merged into ``60``.
#: * ``61s``             – secondary of the 4.16/0.48 kV transformer at bus 61.
#:   Merged into ``61`` (the 480 V spot load is not a distribution fault site).
#: * ``610``             – 480 V load bus behind that transformer.  Removed.
#: * ``300_open``, ``94_open`` – dead-end stubs of the normally-open tie
#:   switches.  Removed.
BUS_ALIAS: dict[str, str | None] = {
    "150": None,
    "150r": None,
    "9r": "9",
    "25r": "25",
    "160": "60",
    "160r": "60",
    "61s": "61",
    "610": None,
    "300_open": None,
    "94_open": None,
}

#: Switches that are normally open.  They carry no current in the base
#: configuration, therefore they must **not** create an edge in the topology
#: graph: a fault "two hops away" through an open tie is electrically infinitely
#: far.  The parser also honours explicit ``Open Line.<name>`` commands found in
#: the ``.dss`` file; this set is a safety net for cases where the switch is
#: opened from a script instead.
NORMALLY_OPEN_SWITCHES: set[str] = {"sw7", "sw8"}

#: Buses excluded from the fault-location label space, for the same reason they
#: are excluded from the graph.  Derived from :data:`BUS_ALIAS` so the two can
#: never drift apart.
FICTITIOUS_BUSES: set[str] = set(BUS_ALIAS)

#: Number of candidate fault locations (classes) after removing the fictitious
#: buses.  Used as an assertion, not as a configuration knob.
EXPECTED_N_CLASSES = 122


# ═══════════════════════════════════════════════════════════════════════════
#  4. DISTRIBUTED GENERATION
# ═══════════════════════════════════════════════════════════════════════════
#: Photovoltaic units injected into the feeder, as ``(name, bus, kVA)``.
#: Rated power is also used as ``Pmpp`` (unity DC/AC ratio) and the units run
#: at unity power factor, i.e. they inject active power only.  This is the
#: standard "PV as negative load" model: during a fault the inverter behaves as
#: a current-limited source, so its main effect on the measurements is to
#: reduce and re-direct the pre-fault current seen by upstream PMUs.
DG_UNITS: list[tuple[str, str, float]] = [
    ("PV1", "114", 1000.0),
    ("PV2", "96", 300.0),
    ("PV3", "13", 75.0),
    ("PV4", "67", 75.0),
]

#: Global irradiance is drawn once per simulation from this range (per-unit of
#: 1 kW/m²), then each unit is perturbed by ±``DG_IRRADIANCE_JITTER`` to emulate
#: partial cloud cover across the feeder.
DG_IRRADIANCE_RANGE = (0.1, 1.0)
DG_IRRADIANCE_JITTER = 0.1
DG_IRRADIANCE_CLIP = (0.05, 1.0)


# ═══════════════════════════════════════════════════════════════════════════
#  5. MONTE-CARLO FAULT SIMULATION
# ═══════════════════════════════════════════════════════════════════════════
N_SIMULATIONS = 100_000

#: Uniform multiplier applied to every load, emulating the daily load curve.
LOAD_MULT_RANGE = (0.7, 1.3)

#: Fault-type mix.  Dominated by single-line-to-ground faults, matching the
#: distribution reported for real overhead distribution feeders
#: (~70-80 % SLG, ~15-20 % phase-to-phase / double-line-to-ground, ~5 % 3-phase).
#:
#: ``LLL`` – three-phase (ungrounded)
#: ``LL``  – phase-to-phase (ungrounded)
#: ``LLG`` – double line-to-ground
#: ``LG``  – single line-to-ground
#:
#: .. important::
#:    The **insertion order matters** and must not be changed.  ``random.choices``
#:    draws one uniform variate and maps it through the cumulative weights in
#:    population order, so reordering these entries changes which fault type a
#:    given RNG state produces — and because each type then consumes a different
#:    number of subsequent draws (LG: one, LL/LLG: two, LLL: none), the whole
#:    scenario stream diverges from the first iteration onwards.  This order
#:    reproduces the original campaign's sequence exactly.
FAULT_TYPE_WEIGHTS: dict[str, float] = {
    "LLL": 0.05,
    "LL": 0.10,
    "LLG": 0.10,
    "LG": 0.75,
}

#: Labels used by the first version of this pipeline, kept only so that older
#: datasets and analysis scripts can be mapped onto the current ones.  The
#: physics is unchanged — this is a renaming to the IEEE notation used in the
#: paper, not a change of fault model.
#:
#: Any downstream script that filters on ``tipo_falta == "FT"`` and similar must
#: be updated: datasets produced by this repository carry the new labels.
FAULT_TYPE_LEGACY_NAMES: dict[str, str] = {
    "FT": "LG",
    "FFT": "LLG",
    "FF": "LL",
    "FFF": "LLL",
}

#: Fault resistance is log-normal per fault type, in ohms.  Values are
#: ``(mu, sigma, cap)`` of ``lognormal(mu, sigma)`` truncated at ``cap``.
#:
#: Rationale: ground-return faults have the widest and highest resistance
#: (arcing through soil, tree contact, downed conductor on dry ground), while
#: metallic phase-to-phase and three-phase faults are close to bolted.
FAULT_RESISTANCE: dict[str, tuple[float, float, float]] = {
    "LG": (2.5, 0.8, 300.0),
    "LLG": (2.0, 0.8, 150.0),
    "LL": (0.5, 0.5, 30.0),
    "LLL": (0.3, 0.5, 10.0),
}

#: Power-flow solver settings.  Faulted networks with high ``Rf`` converge
#: slowly, hence the larger iteration budget once the fault object is added.
SOLVER_MAX_ITER_BASE = 100
SOLVER_MAX_ITER_FAULT = 300
SOLVER_TOLERANCE = 1e-4

#: Rows buffered in memory before being appended to the CSV.
CHECKPOINT_EVERY = 500


# ═══════════════════════════════════════════════════════════════════════════
#  6. PMU CANDIDATE POOL
# ═══════════════════════════════════════════════════════════════════════════
#: Size of the candidate pool searched by FSNR.  The pool is simulated once
#: with all candidates instrumented; every K-PMU scenario is then obtained by
#: column selection, so no re-simulation is needed when the placement changes.
N_CANDIDATES = 30

#: Buses forced into the candidate pool for physical reasons, with the reason
#: recorded for the paper.  Everything else is filled by farthest-first
#: traversal to spread the pool over the feeder.
FORCED_CANDIDATES: dict[str, str] = {
    SUBSTATION_BUS: "feeder head (single-PMU reference scenario)",
    "114": "point of common coupling of PV1 (1000 kVA)",
    "96": "point of common coupling of PV2 (300 kVA)",
    "13": "point of common coupling of PV3 (75 kVA); sectionalising switch",
    "67": "point of common coupling of PV4 (75 kVA)",
    "152": "sectionalising switch (pairs with bus 13)",
    "18": "sectionalising switch (pairs with bus 135)",
    "135": "sectionalising switch (pairs with bus 18)",
    "97": "sectionalising switch (pairs with bus 197)",
    "197": "sectionalising switch (pairs with bus 97)",
    "60": "regulator location and junction of the 60-67 branch",
}

#: PMU placement used by the first version of this study, obtained from
#: engineering judgement rather than from FSNR.  Kept so the search can be
#: benchmarked against it.
HEURISTIC_PMU_SET = ["1", "18", "49", "53", "63"]


# ═══════════════════════════════════════════════════════════════════════════
#  7. FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════════════════════
TARGET_COL = "barra_falta"
PHASES = ("f1", "f2", "f3")

#: Guard added to denominators and to arguments of ``log``.  Small relative to
#: any physically meaningful quantity (volts, amperes, ohms).
EPS = 1e-6

#: Features whose value depends on *which* PMUs are in the subset, and which
#: must therefore be recomputed (never copied) when a subset is extracted from
#: the full candidate feature matrix.
SUBSET_DEPENDENT_FEATURES = {"severity", "pmu_max_dI", "pmu_max_dV"}

#: Features that encode a PMU **index** rather than a magnitude.  They must be
#: one-hot encoded before any scaler is fitted, otherwise the scaler treats a
#: categorical label as an ordinal quantity.
CATEGORICAL_FEATURES = {"pmu_max_dI", "pmu_max_dV"}


def expected_n_features(n_pmus: int) -> int:
    """Number of feature columns produced for ``n_pmus`` PMUs.

    Breakdown per PMU: 24 raw measurement columns (4 phasors x magnitude and
    angle x 3 phases), 57 per-phase derived features (19 x 3 phases) and 9
    per-PMU features (current unbalance, six sequence magnitudes, two sequence
    ratios).  Three global features are shared by all PMUs.
    """
    return 90 * n_pmus + 3


# ═══════════════════════════════════════════════════════════════════════════
#  8. DATA SPLITS (frozen once, reused by every downstream phase)
# ═══════════════════════════════════════════════════════════════════════════
SPLIT_SIZES = {
    "holdout": 5_000,   # never touched until the very last sanity check
    "fsnr": 5_000,      # PMU placement search
    "optuna": 15_000,   # hyper-parameter search
    # cross-validation pool takes whatever remains (~75 000)
}
FSNR_VAL_FRACTION = 0.30
OPTUNA_VAL_FRACTION = 0.30
CV_N_SPLITS = 5


# ═══════════════════════════════════════════════════════════════════════════
#  9. NORMALISATION
# ═══════════════════════════════════════════════════════════════════════════
#: Column-name prefixes grouped by physical quantity.  Each group is scaled
#: with its own scaler so that, for example, the heavy tail of the apparent
#: impedance does not compress the dynamic range of the voltages.
#:
#: Every engineered column must fall into exactly one group.  The sequence
#: magnitudes belong with the phase quantities they are derived from
#: (``abs_dV*`` are volts, ``abs_dI*`` are amperes), and the sequence ratios
#: belong with the other ratios — they are quotients by ``|ΔI₁|``, which is
#: near zero for remote high-resistance faults and therefore heavy-tailed.
FEATURE_GROUPS: dict[str, tuple[str, ...]] = {
    "voltages": ("Vb_", "Vf_", "dfV_", "abs_dV0_", "abs_dV1_", "abs_dV2_"),
    "currents": ("Ib_", "If_", "dfI_", "abs_dI0_", "abs_dI1_", "abs_dI2_"),
    "angles": ("AngVb_", "AngVf_", "AngIb_", "AngIf_", "dfAngV_", "dfAngI_"),
    "trig": ("sin_", "cos_"),
    "ratios": (
        "ratioV_", "ratioI_", "logRV_", "logRI_",
        "ratio_dI2_dI1_", "ratio_dI0_dI1_",
    ),
    "impedance": ("Z_est_", "log_absZ_"),
    "power": ("Sb_", "Sf_", "dS_"),
    "globals": ("unbalance_I_", "severity"),
}

#: Groups scaled with ``RobustScaler`` (median / IQR).  Ratios and apparent
#: impedances are heavy-tailed: a near-zero superimposed current makes
#: ``Z_est = |ΔV| / |ΔI|`` explode, and a single such sample would dominate the
#: mean and standard deviation used by ``StandardScaler``.
ROBUST_SCALER_GROUPS = {"ratios", "impedance"}

#: Post-scaling clip, in scaled units.  Bounds the residual outliers that
#: survive the robust scaler without discarding the samples themselves.
CLIP_RANGE = (-20.0, 20.0)

#: Columns on which measurement noise may be injected during training.
#:
#: This is deliberately **not** derived from :data:`FEATURE_GROUPS`.  The two
#: answer different questions — which scaler suits a column's distribution, and
#: which columns a real instrument independently measures — and they only
#: coincide by accident.
#:
#: The set is the raw and superimposed voltage and current magnitudes plus the
#: apparent powers.  The sequence magnitudes ``abs_dV*`` and ``abs_dI*`` are
#: **excluded** even though they are magnitudes in volts and amperes and share
#: a scaler group with the phase quantities: they are *exact linear functions*
#: of the phase phasors.  Perturbing them independently, on top of perturbing
#: the phase quantities they are computed from, produces a feature vector whose
#: sequence components contradict its phase components — a sample no instrument
#: could ever produce.  The same argument excludes the angle, trigonometric and
#: impedance features.
#:
#: Noise is applied in the **raw** domain, before normalisation.  Applying it to
#: already-scaled columns changes the effective signal-to-noise ratio feature by
#: feature.
NOISE_SAFE_PREFIXES = (
    "Vb_", "Vf_", "dfV_",      # voltage magnitudes [V]
    "Ib_", "If_", "dfI_",      # current magnitudes [A]
    "Sb_", "Sf_", "dS_",       # apparent powers [VA]
)


# ═══════════════════════════════════════════════════════════════════════════
# 10. FSNR — FORWARD SELECTION WITH NEIGHBOURHOOD REFINEMENT
# ═══════════════════════════════════════════════════════════════════════════
FSNR_K_TARGET = 9        # forward path is grown to 9 for the marginal-value curve

#: Budget at which neighbourhood refinement is applied.  This is the study's
#: **operating point**, and it is deliberately smaller than the forward target.
#:
#: Forward selection is nested, so its length-K prefixes give the marginal value
#: of each additional sensor for free — that is what the forward path to 9 is
#: for.  Refinement is not nested: swapping a PMU at one budget says nothing
#: about the best placement at another.  Refining at 9 and then taking a
#: 5-prefix mixes the two and can produce a placement that neither stage
#: actually selected, which is exactly what happens when a swap lands on an
#: early position of the path.  Refinement therefore runs on the prefix of
#: length ``FSNR_REFINE_K``, and that refined set is the reported placement.
FSNR_REFINE_K = 5

FSNR_NEIGHBOR_RADIUS = 3  # hops
FSNR_MAX_REFINE_ROUNDS = 3

#: During the placement search the three global features are dropped.  They are
#: computed over the full candidate pool, so keeping them would leak
#: information from PMUs that are not in the subset being evaluated.  They are
#: recomputed from scratch for the final K-PMU datasets.
FSNR_DROP_SUBSET_DEPENDENT = True

#: Lightweight proxy MLP.  A reduced version of the final Deep Residual MLP:
#: the ranking of PMU subsets is what matters here, not the absolute accuracy.
FSNR_MLP = {
    "hidden": 256,
    "n_blocks": 2,
    "dropout": 0.3,
    "epochs": 200,
    "patience": 40,     # 3 500 training samples need a long patience to settle
    "min_epochs": 30,
    "rescue_epochs": 30,
    "rescue_threshold": 0.02,  # ~2.5x the 1/122 random-guess accuracy
    "lr": 1e-3,
    "weight_decay": 1e-3,
    "warmup_epochs": 5,
    "batch_size": 256,
    "train_noise": 0.005,
}

#: Proxy random forest.  Paired with the MLP so the aggregate score does not
#: favour placements that only suit one inductive bias.
FSNR_RF = {
    "n_estimators": 100,
    "max_features": 0.3,
    "min_samples_leaf": 5,
}

FOCAL_GAMMA = 1.5
LABEL_SMOOTHING = 0.02


# ═══════════════════════════════════════════════════════════════════════════
# 11. RUNTIME
# ═══════════════════════════════════════════════════════════════════════════
NUM_WORKERS = 0  # the FSNR pool (3 500 rows) does not justify worker processes


# ═══════════════════════════════════════════════════════════════════════════
# 12. MODEL ZOO
# ═══════════════════════════════════════════════════════════════════════════
#: Models compared in the study.  **This order is the execution order** of
#: stage 9 and the order in which results and hyper-parameters are listed.
#:
#: The neural models come first deliberately: they carry the paper's headline
#: numbers, and a fold that fails part-way is far more useful having produced
#: them than having produced the tree baselines.  Within the neural group the
#: Deep Residual MLP leads because it is the reference model of the study.
#:
#: Two gradient-based dense models, one attention-based tabular model and two
#: tree ensembles — chosen so the comparison spans genuinely different
#: inductive biases rather than variations of one.
MODELS = ("DeepResidualMLP", "MLP", "TabNet", "RF", "LightGBM")


def order_models(mapping: dict) -> dict:
    """Reorder a ``{model: ...}`` mapping to follow :data:`MODELS`.

    Python dictionaries preserve insertion order, and both ``best_hp.json`` and
    the console summaries are read by humans, so a stable order is worth
    enforcing — and enforcing it from a single source of truth, rather than
    letting each writer choose.  Keys not listed in :data:`MODELS` are appended
    in their original order rather than dropped: an unrecognised entry is
    someone's experiment, not garbage.
    """
    known = [m for m in MODELS if m in mapping]
    extra = [m for m in mapping if m not in MODELS]
    return {m: mapping[m] for m in known + extra}

#: Models trained by gradient descent through the shared training loop.
NEURAL_MODELS = frozenset({"MLP", "DeepResidualMLP", "TabNet"})

#: Models that consume the **unscaled** feature matrix.  Decision trees split
#: on thresholds, so any per-feature affine transform leaves them unchanged —
#: but the post-scaling clip at :data:`CLIP_RANGE` is *not* affine, and would
#: silently merge every extreme apparent-impedance value into one threshold.
#: Feeding trees the raw matrix keeps that information and, more importantly,
#: keeps the hyper-parameter search and the final training on identical inputs.
UNSCALED_MODELS = frozenset({"RF", "LightGBM"})

#: Human-readable names used in results tables and output filenames.
MODEL_DISPLAY_NAMES = {
    "MLP": "MLP Baseline",
    "DeepResidualMLP": "Deep Residual MLP",
    "TabNet": "TabNet",
    "RF": "Random Forest",
    "LightGBM": "LightGBM",
}


# ═══════════════════════════════════════════════════════════════════════════
# 13. HYPER-PARAMETER SEARCH (stage 8)
# ═══════════════════════════════════════════════════════════════════════════
OPTUNA_DIR = RESULTS_DIR / "optuna"
BEST_HP_JSON = OPTUNA_DIR / "best_hp.json"

N_TRIALS = 75
TPE_STARTUP_TRIALS = 15
PRUNER_STARTUP_TRIALS = 15
PRUNER_WARMUP_STEPS = 30

#: Training budget during the search.  Deliberately **identical for every
#: neural model**: if one architecture were given more epochs or a longer
#: patience than another, the comparison in the paper would measure the budget
#: rather than the architecture.  600 is a ceiling; with patience 60 the runs
#: are expected to stop well before it.
SEARCH_EPOCHS = 600
SEARCH_PATIENCE = 60
SEARCH_MIN_EPOCHS = 10
SEARCH_WARMUP_EPOCHS = 10


# ═══════════════════════════════════════════════════════════════════════════
# 14. CROSS-VALIDATED TRAINING (stage 9)
# ═══════════════════════════════════════════════════════════════════════════
CV_DIR = RESULTS_DIR / "cv"

#: Fraction of each fold's training pool held out for early stopping.  This is
#: **not** a tuning split: the hyper-parameters are already fixed by stage 8,
#: which ran on a disjoint pool.  It only decides when to stop.
INTERNAL_VAL_FRACTION = 0.15

CV_EPOCHS = 2000
CV_PATIENCE = 100
CV_WARMUP_EPOCHS = 10

LGB_N_ESTIMATORS = 3000
LGB_BAGGING_FREQ = 1
LGB_EARLY_STOPPING_ROUNDS = 150

#: Samples used to measure single-sample inference latency.
N_INFERENCE_RUNS = 100

#: DataLoader worker processes.  Windows spawns rather than forks, so workers
#: re-import the module; keep at 0 there if startup cost dominates.
DATALOADER_WORKERS = 4


# ═══════════════════════════════════════════════════════════════════════════
# 15. CLASSICAL BASELINES (stage 10)
# ═══════════════════════════════════════════════════════════════════════════
BASELINE_DIR = RESULTS_DIR / "baselines"

#: Score normalisation for the impedance baseline.
#:
#: ``True``  – relative error, ``Σ w·|Z_app/Z_path − 1|²`` (dimensionless).
#:             Compensates the magnitude bias introduced by the stochastic
#:             fault resistance, which under the absolute criterion
#:             systematically penalises distant buses.
#: ``False`` – absolute error, ``Σ w·|Z_app − Z_path|²`` (Ω²).
BASELINE_NORMALIZE_BY_PATH = True

#: How scores become pseudo-probabilities so that Top-3 and Top-5 are defined.
#: ``"softmax"`` uses ``p ∝ exp(−score/T)``; ``"rank"`` uses ``p ∝ 1/(rank+1)``.
BASELINE_CALIBRATION = "softmax"
BASELINE_SOFTMAX_TEMP = None  # None -> per-run median of the per-sample spread

#: Floor applied to conductors whose series impedance is numerically zero
#: (closed switches, regulator windings).  Without it, Dijkstra sees zero-cost
#: edges and the relative score divides by zero.  Small enough to be
#: physically negligible.
BASELINE_ZERO_IMPEDANCE_FLOOR = 1e-4

#: Clip on the observed apparent impedance.  A phase carrying no superimposed
#: current gives ``|ΔV|/|ΔI| -> ∞``; without a bound those samples dominate.
BASELINE_Z_APP_CLIP = 1e6


# ═══════════════════════════════════════════════════════════════════════════
# 16. POST-TRAINING ANALYSES (stages 11-18)
# ═══════════════════════════════════════════════════════════════════════════
ANALYSIS_DIR = RESULTS_DIR / "analysis"
FIGURE_DIR = RESULTS_DIR / "figures"

#: Hop distances reported in the error CDF.  ``F(0)`` is Top-1 accuracy by
#: construction, so the curve extends the headline metric rather than
#: replacing it.
HOP_CDF_LEVELS = (0, 1, 2, 3)

#: Bootstrap resamples for the confidence intervals.
N_BOOTSTRAP = 10_000
BOOTSTRAP_CI = (2.5, 97.5)

#: Fault-resistance bins for the conditional accuracy table [ohm].  Chosen so
#: the first bin is near-bolted and the last covers the high-resistance regime
#: where any phasor-based locator degrades.
RF_BINS = (0.0, 5.0, 15.0, 50.0, float("inf"))
RF_BIN_LABELS = ("[0,5)", "[5,15)", "[15,50)", ">=50")

#: Permutation repeats per feature group.  Permutation importance is expensive
#: (one full inference pass per repeat per group per fold) and its variance
#: across repeats is small relative to the variance across folds.
N_PERMUTATION_REPEATS = 3

#: Measurement-error levels swept in the robustness study, expressed as
#: **total vector error** (TVE) in the sense of IEC/IEEE 60255-118-1: the modulus
#: of the complex relative deviation of a measured phasor from its true value.
#: The standard's compliance limit for a class-P PMU is 1 %, so the sweep
#: brackets the specification rather than exploring an abstract range.
#:
#: The error is applied to the four *measured* phasors and every derived
#: feature is then recomputed, so a given TVE propagates the way it would in a
#: real instrument chain.  This matters: the superimposed quantities are
#: differences of large numbers, so a 1 % error on the voltage phasors becomes
#: roughly |V|/|dV| ~ 20 times larger on the superimposed voltage.
#: The grid is concentrated **below** the standard's 1 % limit, because that is
#: where deployed instruments actually sit: a compliant class-P PMU reports
#: 0.1-0.3 % TVE in steady state, and 1 % is the worst an instrument may be and
#: still pass. Sweeping 0.5-5 % measures badly out-of-specification hardware and
#: says little about a real installation. It also misses the transition
#: entirely: with errors propagated through the superimposed quantities the
#: degradation is a cliff rather than a slope, and a grid whose second point is
#: already past the cliff cannot locate it.
NOISE_SIGMAS = (0.0, 0.0005, 0.001, 0.002, 0.005, 0.01)
N_NOISE_RUNS = 3
ISOLATED_GROUP_SIGMA = 0.002

#: Instrument chains perturbed independently by the isolated-error scenario.
#: These are the two chains a substation actually has — voltage transformers
#: and current transformers.  Apparent power is *not* listed: it is computed
#: from both, so it has no error source of its own.
MEASURED_QUANTITIES = ("voltage", "current")

#: How a lost PMU is represented.
#:
#: ``"mean"``  – its columns are replaced by the training-set mean.  This is the
#:               imputation a deployed system would use, and it keeps the input
#:               inside the distribution the model was fitted on.
#: ``"zero"``  – its columns are set to zero in the raw domain, i.e. no signal.
#:               Physically interpretable but far outside the training
#:               distribution, so it overstates the degradation.
MISSING_PMU_MODE = "mean"

#: Per-bus recall above which a bus is considered reliably located.
PER_BUS_HEALTHY_THRESHOLD = 0.90
N_WORST_BUSES_PLOTTED = 30

#: Bus pairs joined by a normally-closed switch.  Electrically these are the
#: same node to within a few milliohms, so no measurement-based method can
#: separate them; they are highlighted in the per-bus figure.
SWITCH_PAIRS = (("13", "152"), ("18", "135"), ("97", "197"))