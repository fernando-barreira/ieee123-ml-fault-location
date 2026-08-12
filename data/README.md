# Data

Generated datasets are **not** versioned in this repository. The raw
measurement file is several gigabytes and the derived feature matrices are
larger still.

Everything here can be regenerated from scratch by running stages 1-4 of
the pipeline (see the main [README](../README.md)).

## Layout

```
data/
├── raw/          measurements_candidate_pmus.csv   ← stage 3 output
├── interim/      graph_ieee123.gpickle             ← stage 1
│                 pmu_candidates.pkl                ← stage 2
│                 splits.pkl, splits_summary.csv    ← stage 5
└── processed/    features_candidate_pmus.csv       ← stage 4
                  features_1pmu.csv                 ← stage 7
                  features_5pmu.csv
                  features_9pmu.csv
                  pmu_subsets.json
```

Override the root with the `IEEE123_DATA_DIR` environment variable if the
datasets live on another drive.

---

## `raw/measurements_candidate_pmus.csv`

One row per simulated fault scenario. 100 000 rows.

### Scenario columns

These describe the simulated ground truth. Only `barra_falta` is a modelling
target; the others are metadata used for stratified analysis. **None of them is
observable by a PMU**, so all except the target are dropped before training.

| Column | Unit | Description |
|---|---|---|
| `barra_falta` | — | **Target.** Faulted bus, one of 122 classes. |
| `tipo_falta` | — | `LG`, `LLG`, `LL` or `LLL`. |
| `fases` | — | Faulted phase nodes, e.g. `1`, `1.2`, `1.2.3`. |
| `impedancia_falta` | Ω | Fault resistance `Rf`, per faulted branch. |
| `fator_carga` | p.u. | Uniform load multiplier, 0.7-1.3. |
| `irradiancia_global` | p.u. | Global irradiance before per-unit jitter. |

### Distributed generation columns

Four PV units, indexed `k = 0…3`.

| Column | Unit | Description |
|---|---|---|
| `gd{k}_bus` | — | Point of common coupling. |
| `gd{k}_kw` | kW | Active power injected in this scenario. |
| `gd{k}_irradiancia` | p.u. | Irradiance at this unit. |

Rated powers: PV1 at bus 114 (1000 kVA), PV2 at bus 96 (300 kVA), PV3 at bus 13
(75 kVA), PV4 at bus 67 (75 kVA). All at unity power factor.

### Measurement columns

For every candidate PMU bus `B` and every phase `n ∈ {1, 2, 3}` (A, B, C):

| Column | Unit | Description |
|---|---|---|
| `Vb_{B}_f{n}` | V | Pre-fault voltage magnitude, line-to-neutral. |
| `AngVb_{B}_f{n}` | degrees | Pre-fault voltage angle. |
| `Vf_{B}_f{n}` | V | During-fault voltage magnitude. |
| `AngVf_{B}_f{n}` | degrees | During-fault voltage angle. |
| `Ib_{B}_f{n}` | A | Pre-fault current magnitude in the monitored branch. |
| `AngIb_{B}_f{n}` | degrees | Pre-fault current angle. |
| `If_{B}_f{n}` | A | During-fault current magnitude. |
| `AngIf_{B}_f{n}` | degrees | During-fault current angle. |

24 columns per PMU. All PMU sites are three-phase; a zero in these columns
means the solver returned fewer terminals than expected and should be
investigated rather than ignored.

**Sign and reference conventions.** Voltages are line-to-neutral, as returned
by OpenDSS. Currents are branch currents at the terminal of the monitored line
facing the PMU bus, positive *into* the line. Each PMU monitors one fixed
branch, chosen once from the topology as the line facing the source, and the
same branch is read before and during the fault — otherwise the superimposed
current `ΔĨ = Ĩ_fault − Ĩ_pre` would combine phasors measured on two different
conductors.

---

## `processed/features_*.csv`

Feature matrices produced by stages 4 and 7. Row order matches the raw file
exactly, which is what keeps the frozen splits valid.

Column count is `90K + 3` features plus the target, where `K` is the number of
PMUs: 93 for K = 1, 453 for K = 5, 813 for K = 9. After one-hot expansion of
the two categorical `pmu_max_*` indicators this becomes `90K + 1 + 2K`
columns — 461 for K = 5.

Per PMU and phase (19 columns): `ratioV`, `ratioI`, `logRV`, `logRI`, `dfV`,
`dfI`, `dfAngV`, `dfAngI`, `sin_dfAngV`, `cos_dfAngV`, `sin_dfAngI`,
`cos_dfAngI`, `Z_est`, `log_absZ`, `sin_Z_ang`, `cos_Z_ang`, `Sb`, `Sf`, `dS`.

Per PMU (9 columns): `unbalance_I`, `abs_dV0`, `abs_dV1`, `abs_dV2`, `abs_dI0`,
`abs_dI1`, `abs_dI2`, `ratio_dI2_dI1`, `ratio_dI0_dI1`.

Global (3 columns): `severity`, `pmu_max_dI`, `pmu_max_dV`.

The 24 raw measurement columns of each retained PMU are carried through as
well.

> **`pmu_max_dI` and `pmu_max_dV` are categorical.** They hold the *position*
> of a PMU in the placement list, not a magnitude and not a bus number. One-hot
> encode them before fitting any scaler (`src.scaling.one_hot_categoricals`).
> Note that the resulting names end in `_<digit>`, which collides with the
> per-PMU column naming — code that infers a column's PMU from its suffix must
> exclude them explicitly.

> **The three global features are subset-dependent.** They are defined over the
> set of PMUs present. When extracting a K-PMU scenario they must be recomputed
> over those K PMUs, never copied from a matrix built on a larger set — copying
> leaks information from PMUs the scenario does not have.
> `src.features.select_pmu_subset` does this correctly.

---

## `interim/splits.pkl`

Frozen positional indices for the four pools, their internal train/validation
splits, and the five cross-validation folds. Fold indices are **local to**
`cv_indices`: index the dataset with `cv_indices[fold_train]`.

The file records a SHA-256 fingerprint of the label column.
`src.splits.load_splits` verifies it and raises if the dataset does not match,
which prevents the most damaging silent failure in the pipeline — regenerating
the raw dataset and then reusing stale indices, which would scramble the
sample-to-label correspondence without any error.
