# Machine Learning for Bus-Level Fault Location in Active Distribution Networks

Reproducibility package for **"Machine Learning for Bus-Level Fault Location
in Active Distribution Networks"**.

The study identifies which of the 122 candidate buses of the IEEE 123-bus
distribution feeder is faulted, using synchronised voltage and current phasors
from a small number of PMUs, in a feeder that hosts four photovoltaic
distributed generators. The full pipeline — network model, Monte-Carlo
short-circuit campaign, physics-informed feature engineering, algorithmic PMU
placement, model training and evaluation — is contained here.

---

## What the problem is

A fault on a distribution feeder must be located before a crew can be
dispatched. Distribution feeders make this hard in ways transmission lines do
not: they are radial and heavily branched, so impedance-to-fault maps to many
candidate locations at once; they are unbalanced, with single- and two-phase
laterals; and distributed generation injects fault current from inside the
feeder, breaking the assumption of a single upstream source that classical
impedance-based methods rely on.

This work treats the problem as **122-class classification** from sparse
measurements. The inputs are the pre-fault and during-fault phasors at five
PMUs; the output is the faulted bus.

## What is in this repository

| Stage | Script | Produces |
|---|---|---|
| 0 | `scripts/00_check_setup.py` | Preflight check (no output) |
| 1 | `scripts/01_build_graph.py` | Feeder topology graph |
| 2 | `scripts/02_select_candidates.py` | Pool of 30 candidate PMU sites |
| 3 | `scripts/03_run_simulations.py` | 100 000 simulated fault scenarios |
| 4 | `scripts/04_build_features.py` | Physics-informed feature matrix |
| 5 | `scripts/05_make_splits.py` | Frozen data splits |
| 6 | `scripts/06_run_fsnr.py` | PMU placement (FSNR) |
| 7 | `scripts/07_build_pmu_subsets.py` | Training dataset for the 5-PMU operating point |
| 8 | `scripts/08_optuna_search.py` | Hyper-parameters per model |
| 9 | `scripts/09_cross_validate.py` | 5-fold results — the headline numbers |
| 10 | `scripts/10_run_baselines.py` | Classical baselines on the same folds |
| 11 | `scripts/11_analyze_hop_distance.py` | Hop-distance error and its CDF |
| 12 | `scripts/12_analyze_per_fault_type.py` | Accuracy by fault type and resistance |
| 13 | `scripts/13_bootstrap_ci.py` | Pooled bootstrap confidence intervals |
| 14 | `scripts/14_feature_importance.py` | Group necessity, sufficiency and Shapley |
| 15 | `scripts/15_robustness.py` | Missing PMUs and measurement noise |
| 16 | `scripts/16_holdout_check.py` | One-shot check on the sealed holdout |
| 17 | `scripts/17_make_figures.py` | The paper's figures |

Stages 11-17 are read-only analyses: they consume the predictions and models
saved by stages 9 and 10 and never retrain anything, so a figure cannot
disagree with the table beside it. Stage 16 evaluates the sealed holdout pool
exactly once and its result must not feed back into any earlier stage.

Stages 8-10 consume the stage-7 output (`data/processed/features_5pmu.csv` via
`config.features_csv(5)`) and the frozen splits, and nothing else: stage 7 is
the only producer of scenario datasets, so the PMU set the models are trained
on is by construction the one the placement search selected.

The remaining analyses — bootstrap confidence intervals, robustness sweeps,
permutation importance and the paper's figures — are released alongside the
final version of the paper.

Every stage reads its configuration from `config.py`. There are no absolute
paths in the code; the only machine-dependent setting is where the OpenDSS
test case lives.

---

## Installation

```bash
git clone https://github.com/fernando-barreira/ieee123-ml-fault-location.git
cd ieee123-ml-fault-location
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Python 3.10 or newer. A CUDA GPU is needed for stage 6 and for the neural
models; everything else runs on CPU.

### Network model

The IEEE 123-bus OpenDSS model is distributed by EPRI together with OpenDSS and
is **not** redistributed here. Install OpenDSS and point the pipeline at the
master file with the `IEEE123_DSS_MASTER` environment variable.

**Windows (PowerShell).** For the current session only:

```powershell
$env:IEEE123_DSS_MASTER = "C:\Program Files\OpenDSS\IEEETestCases\123Bus\IEEE123Master.dss"
```

Permanently, so every future session picks it up:

```powershell
[Environment]::SetEnvironmentVariable("IEEE123_DSS_MASTER", "C:\Program Files\OpenDSS\IEEETestCases\123Bus\IEEE123Master.dss", "User")
```

> The permanent form writes to the registry and reaches only processes started
> **afterwards**. If you run from an IDE, restart the IDE itself — a new
> terminal tab inherits its environment from the still-running IDE process and
> will not see the change. Verify with:
>
> ```powershell
> python -c "import os; print(os.environ.get('IEEE123_DSS_MASTER'))"
> ```

**Linux / macOS:**

```bash
export IEEE123_DSS_MASTER=/path/to/IEEETestCases/123Bus/IEEE123Master.dss
```

Every script also accepts `--dss <path>` directly, which overrides the variable.

Copying the `123Bus` folder into `network/dss/` works too and is what the
default path expects, but that directory is deliberately excluded by
`.gitignore`: the model is EPRI's to distribute, not this repository's.

Then confirm the installation:

```bash
python scripts/00_check_setup.py
```

It verifies package versions, the OpenDSS model and engine, GPU availability
and — the check that matters most before a multi-hour stage — that every output
directory is writable.

If the repository lives somewhere the process cannot write (OneDrive-synced
folders and Windows Defender Controlled Folder Access are the usual culprits),
put the datasets elsewhere:

```powershell
$env:IEEE123_DATA_DIR = "D:\ic_data"
```

Stage 3 additionally needs a working OpenDSS engine through
`py_dss_interface`. Stages 4-7 do not: they operate on the CSV produced by
stage 3, so the results can be reproduced from the published dataset without
installing the solver.

### Data

Generated datasets are not versioned — the raw measurement file alone is
several gigabytes. See [`data/README.md`](data/README.md) for more information
and the column schema.

---

## Running the pipeline

```bash
python scripts/00_check_setup.py                # seconds; run this first
python scripts/01_build_graph.py --report
python scripts/02_select_candidates.py --plot
python scripts/03_run_simulations.py            # hours; resumable with --resume
python scripts/04_build_features.py
python scripts/05_make_splits.py
python scripts/06_run_fsnr.py                   # hours on one GPU; resumable
python scripts/07_build_pmu_subsets.py
python scripts/08_optuna_search.py             # 75 trials x 5 models; resumable
python scripts/09_cross_validate.py            # the reported results
python scripts/10_run_baselines.py             # must run after stage 9
python scripts/11_analyze_hop_distance.py
python scripts/12_analyze_per_fault_type.py
python scripts/13_bootstrap_ci.py
python scripts/14_feature_importance.py       # add --shapley for the additive decomposition
python scripts/15_robustness.py
python scripts/16_holdout_check.py             # once, and only once
python scripts/17_make_figures.py              # needs 11 and 15
```

Every script accepts `--help`. To check that the code runs before committing to
a full campaign:

```bash
python scripts/03_run_simulations.py -n 2000 --out data/raw/smoke.csv
python scripts/04_build_features.py --raw data/raw/smoke.csv --out data/processed/smoke_features.csv
```

---

## Method

### 1. Test system

The IEEE 123-bus feeder is a 4.16 kV radial system with voltage regulators,
shunt capacitors, normally-open tie switches and a mix of single-, two- and
three-phase laterals. After removing the source bus, the regulator secondaries
(`150r`, `9r`, `25r`, `160r`), the 480 V transformer secondary (`610`), the
switch-intermediate node `61s`, bus `160` (electrically identical to `60`
through an in-line regulator) and the dead-end terminals of the normally-open
switches Sw7 and Sw8, **122 physically meaningful buses** remain. These are the
classes of the classifier. The removal map lives in `config.BUS_ALIAS` and is
applied by a single parser (`src/dss_parser.py`), so the graph, the candidate
pool and the label space cannot disagree.

### 2. Simulation

Each of the 100 000 scenarios draws a global irradiance (0.1-1.0 p.u., with
per-unit jitter across the four PV units), a uniform load multiplier
(0.7-1.3 p.u.), a fault type (75 % LG, 10 % LLG, 10 % LL, 5 % LLL), a fault bus
compatible with that type, and a fault resistance from a type-calibrated
log-normal distribution. The network is solved **twice** — before and during
the fault — so that superimposed quantities are available downstream.

Each PMU monitors one fixed branch: its upstream line, chosen once from the
feeder topology as the incident conductor facing the source (the feeder head,
which has no upstream line, falls back to its largest-current branch). This
matters: if the monitored branch were re-selected independently before and
during the fault, the superimposed current `ΔĨ = Ĩ_fault − Ĩ_pre` would
subtract phasors measured on two different conductors.

### 3. Features

453 physics-informed features for five PMUs (`config.expected_n_features(K) =
90K + 3`), expanding to 461 columns once the two categorical
"which PMU saw the largest change" indicators are one-hot encoded:

* **superimposed phasors** `ΔṼ`, `ΔĨ`, computed as true complex differences —
  by superposition these isolate the pure-fault network from the load flow,
  which is what lets one model span the whole load range;
* **per-unit sag and surge** and their logarithms;
* **superimposed apparent impedance** `ΔṼ/ΔĨ`, magnitude and angle — the basis
  of distance protection;
* **Fortescue sequence components** of the superimposed phasors; zero sequence
  exists only with a ground return path, so `ΔI₀` separates grounded from
  ungrounded faults and its decay with distance differs from the
  positive-sequence decay;
* **apparent power** before, during and the difference;
* **cross-PMU aggregates**: mean superimposed current and the index of the most
  affected PMU.

Angles are exported as sine and cosine as well as degrees, so no model has to
learn that −179° and +179° are adjacent.

### 4. PMU placement (FSNR)

Forward Selection with Neighbourhood Refinement over a 30-bus candidate pool.
Forward selection grows the placement one PMU at a time; refinement then swaps
each selected PMU with candidates within three hops on the topology graph.

Each placement is scored by the **geometric mean** of the validation accuracy
of two lightweight proxies — a residual MLP and a random forest. Two proxies
with opposite inductive biases are used deliberately: since the paper compares
gradient-based against tree-based models, letting either one choose the sensor
locations would build the conclusion into the experimental setup. The geometric
mean penalises placements that suit only one family.

Because forward selection is nested, the length-K prefix of its path is the
K-PMU placement, so a single run yields the marginal-value curve across sensor
budgets for free — at proxy level, and that is where the choice of operating
point is settled. Averaged over the PMUs they add, the second four sensors are
worth about half of the first four, which is what justifies five as the
operating point for the rest of the study. Re-training the full model stack at
other budgets would add cost without adding evidence to that argument, so
stage 7 produces only the five-PMU dataset by default (`--k 1,5,9` remains
available for follow-up work).

K = 1 is the one placement not taken from the forward path: it is pinned to the
feeder head (bus 149), because a one-PMU deployment in practice means
instrumenting the substation.

### Expect the selected buses to vary slightly

Re-running the placement search — on a different machine, after regenerating the
dataset, or with a different seed — can return a placement that differs from the
one reported in the paper by one or two buses. This is a property of the problem,
not a defect, and it is worth understanding before treating any single bus list
as canonical.

The reason is that the proxy score is estimated on 1 500 validation samples, so
its standard error at the observed accuracy levels is around one percentage
point, while the margin between the best and second-best candidate at each
forward step is typically a few tenths of a point. Several candidates are
therefore statistically indistinguishable at every step, and an arbitrarily small
perturbation can reorder them. Two further sources compound this: the feeder
contains genuinely interchangeable measurement points — buses 1 and 149 sit at
opposite terminals of the same conductor, and each normally-closed switch joins a
pair of buses that are electrically almost identical — and, as described in the
previous section, the candidate pool itself can differ from the original study's.

For a bit-for-bit reproduction of a published placement, use the placement file
shipped with the results rather than re-running the search, or pass the bus list
directly to stage 7:

```bash
python scripts/07_build_pmu_subsets.py --k 5 --manual 5=1,42,63,91,135
```


### 5. Models

Five models spanning genuinely different inductive biases: two dense
gradient-based networks (a plain MLP and a residual stack), an attention-based
tabular model (TabNet), and two tree ensembles (random forest, LightGBM).

Hyper-parameters are searched per model on the `optuna` pool (stage 8) and then
held fixed through cross-validation (stage 9). Every neural model gets the
*same* epoch budget, patience and stopping rule — otherwise the comparison would
partly measure the training schedule rather than the architecture. Tree
ensembles receive the unscaled feature matrix in both stages: splitting
thresholds are invariant to the affine part of the scaling, but the
post-scaling clip is not, so tuning on clipped inputs and training on raw ones
would not evaluate the model that was selected.

Alongside Top-1/3/5 accuracy, stage 9 records the **hop distance** of every
prediction. For a fault locator that is the operationally meaningful error:
predicting an adjacent bus sends the crew to the right span, while predicting a
bus on a different lateral sends them somewhere else entirely.

Two learning-free baselines (stage 10) are evaluated on exactly the same fold
test sets. The impedance baseline answers "is learning necessary?"; the purely
topological one answers "is the model just memorising the feeder map?". Keeping
them separate is what lets the paper decompose the two.

### 6. Data splits

Four disjoint pools, frozen once and reused by every phase:

| Pool | Size | Used by |
|---|---|---|
| `fsnr` | 5 000 | PMU placement search |
| `optuna` | 15 000 | hyper-parameter search |
| `cv` | 75 000 | stratified 5-fold cross-validation (reported results) |
| `holdout` | 5 000 | untouched until the final sanity check |

All splits are stratified on the fault bus. The splits file records a hash of
the label column, and loading it against a different dataset raises rather than
silently scrambling the sample-to-label correspondence.

---

## Repository layout

```
config.py               all paths, physical constants and hyper-parameters
src/
  dss_parser.py         single OpenDSS text parser (topology + phase counts)
  graph.py              topology graph, hop distances, coverage statistics
  pmu_candidates.py     candidate pool selection
  simulation.py         PV insertion, fault application, PMU measurement
  features.py           feature engineering and K-PMU subset extraction
  scaling.py            group-wise normalisation, one-hot encoding
  splits.py             frozen pools and cross-validation folds
  models.py             architectures, focal loss, class weights
  training.py           shared training loop, metrics, instrumentation
  baselines.py          impedance and topological reference methods
  analysis.py           shared loaders for the post-training analyses
  plotting.py           figure style shared by every figure
  placement/
    proxies.py          proxy MLP and random forest, focal loss
    fsnr.py             forward selection and neighbourhood refinement
scripts/                numbered entry points, one per stage
tools/                  auditing utilities (not part of the pipeline)
data/                   generated datasets (not versioned)
network/dss/            OpenDSS model (not redistributed)
results/                metrics, logs and placement artefacts
```

---

## Reproducibility notes

* Every stochastic component is seeded from `config.SEED = 42`.
* Relative paths passed on the command line resolve against the **repository
  root**, not the working directory, so `--out data/raw/smoke.csv` means the
  same file wherever the script is launched from. Absolute paths are honoured
  as given.
* Importing `config` has no side effects; directories are created by the script
  that needs them, and their writability is probed before any long stage
  starts.
* Stages 3 and 6 are the long ones and both are resumable.
* The splits are deliberately immutable. `scripts/05_make_splits.py` refuses to
  overwrite an existing splits file without `--force`, because doing so
  invalidates every result derived from it.
* Scalers are fitted on training data only, per feature group; ratios and
  apparent impedances use a robust scaler because a near-zero superimposed
  current makes `ΔV/ΔI` heavy-tailed.
* Training-time noise augmentation is applied in the raw domain before
  normalisation. Applying it to already-normalised columns changes the
  effective signal-to-noise ratio per feature and degrades every model.
* The set of columns that receive training noise is defined separately from the
  scaler groups (`config.NOISE_SAFE_PREFIXES`). Sequence magnitudes are scaled
  alongside the phase quantities but are never perturbed: they are exact
  linear functions of those quantities, so perturbing both would produce a
  sample whose sequence components contradict its phase components.
* The robustness study (stage 15) uses a stricter model still. It perturbs only
  the four phasors a PMU measures, with a complex multiplicative error whose
  modulus is the total vector error of IEC/IEEE 60255-118-1, and then **re-derives
  every feature** from the perturbed phasors. This is what makes the sweep
  physically realisable, and it is not a cosmetic difference: superimposed
  quantities are differences of large numbers, so a 1 % TVE on the voltage
  phasors appears roughly an order of magnitude larger on the superimposed
  voltage and larger still on its sequence components. Perturbing the derived
  columns directly would hide that amplification entirely.

## Tools
```bash
python tools/class_distribution.py        # class balance, and where it comes from
python tools/paper_claims.py              # numbers quoted in the paper's prose
python tools/rebuild_best_hp.py           # reconstruct best_hp.json from stage 8
python tools/sensor_budget_table.py       # sensor-budget table (reads stage 6)
python tools/training_epochs.py           # where training peaked and where it stopped
```

The `scripts/` directory contains the main pipeline of the project. Each script
performs a stage of the workflow and may generate artefacts consumed by later
pipeline stages. 

`tools/` are auxiliary utilities. They only read the artefacts produced by
the main pipeline to generate tables, summaries, or other outputs
that facilitate data inspection and interpretation. They do not produce any
artefact that is consumed by the scripts, so they can be executed whenever the
required pipeline outputs already exist.

## Known physical limits of the approach

These are properties of the problem, not defects of the models, and are
reported as such in the paper:

* **Switch-adjacent bus pairs** (13/152, 18/135, 97/197) are separated by a
  closed switch of a few milliohms. Their phasor signatures are identical to
  within solver tolerance from anywhere on the feeder. No model can separate
  them, which is why hop-distance error is reported alongside exact accuracy.
* **High-resistance ground faults** produce a superimposed current barely above
  normal load variation. Accuracy degrades sharply above roughly 50 Ω.
* **Quasi-steady-state phasors only.** Travelling waves, CT saturation and the
  DC decaying component are outside the model. The results describe a
  phasor-based locator, not a transient-based one.
* **Detection is assumed.** The pipeline solves localisation *given* that a
  fault occurred.

---

## Citing

If you use this code or the associated dataset, please cite the paper (see
`CITATION.cff`).

## Licence

Code released under the MIT Licence — see [`LICENSE`](LICENSE).

The IEEE 123-bus test feeder is provided by the IEEE PES Test Feeder Working
Group; OpenDSS is developed by EPRI. Neither is redistributed here.

## Acknowledgements

Developed as an undergraduate research project in the Department of Electrical
Engineering, Universidade Estadual Paulista "Júlio de Mesquita Filho" (UNESP),
Bauru, Brazil, under the supervision of Prof. Dr. José Alfredo Covolan Ulson.

The authors thank EPRI and the IEEE PES Test Feeder Working Group for making
OpenDSS and the IEEE 123-bus benchmark freely available.
