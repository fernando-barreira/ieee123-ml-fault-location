r"""Stage 14 — feature-group importance: necessity, sufficiency and Shapley.

Three measures of what each physical feature group contributes.

**Necessity — permute one group.**  Shuffle every column of one group across
samples, destroying its association with the label while preserving its
marginal distribution.  This asks: *what does the model lose if this class of
measurement is corrupted?*  With redundant groups it understates unique value,
because the information survives in the correlated groups that were left
intact.

**Sufficiency — permute every group except one.**  The complement: keep one
group and corrupt all others.  This asks: *how far does this group get on its
own?*  Together with necessity it brackets the group's role — a group that is
neither necessary nor sufficient is genuinely redundant, while one that is
sufficient but not necessary is duplicated elsewhere.

**Grouped Shapley values (optional, expensive).**  The only measure here that
*decomposes*: Shapley values satisfy

.. math:: \sum_i \varphi_i = v(N) - v(\varnothing),

so the contributions sum exactly to the accuracy the full feature set buys over
a fully corrupted one.  Necessity and sufficiency do not add up to anything and
must never be presented as if they did.  Shapley achieves this by averaging a
group's marginal contribution over **every** coalition of the others, which is
also why it costs :math:`2^{|G|}` model evaluations — 512 for the nine groups
used here.  Enable with ``--shapley``; the script prints a time estimate first.

For brevity, the paper presents only the grouped Shapley values, but this script
can compute all three measures cited above for future analysis.

A caveat that applies to all three, and that belongs in any write-up: permuting
a group breaks physical consistency.  A shuffled voltage group paired with an
unshuffled current group describes a network state that cannot exist.  These
measures therefore quantify *reliance* — how much the fitted model leans on a
group — not sensitivity to any perturbation an instrument could produce.  The
robustness study answers that second question instead, and does so with a
physically realisable error model.

Usage
-----
.. code-block:: bash

    python scripts/14_feature_importance.py
    python scripts/14_feature_importance.py --shapley --shapley-subsample 3000
    python scripts/14_feature_importance.py --models "Deep Residual MLP" --repeats 5
"""

from __future__ import annotations

import os
import warnings

# Must be set BEFORE scikit-learn or joblib is imported, and must live in the
# environment rather than only in this process's filter list: RandomForest's
# predict_proba fans out to joblib workers, and a filter registered at runtime
# does not reach them. loky propagates the parent environment to its children,
# so PYTHONWARNINGS is what actually silences them.
#
# The notice being suppressed is scikit-learn's `sklearn.utils.parallel.delayed`
# advisory. It is harmless, but it fires once per parallel batch: in an earlier
# run it buried 86 lines of results under 6 500 lines of noise, and output
# nobody can read is output nobody checks.
os.environ.setdefault("PYTHONWARNINGS", "ignore::UserWarning")
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import argparse
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config
from src.analysis import (
    Report,
    build_feature_matrix,
    load_fold_indices,
    load_fold_model,
    load_preprocessing,
    load_scalers,
    mean_std,
    model_input,
    predict_probs,
)


def permute(raw: np.ndarray, columns: list[int], rng) -> np.ndarray:
    """Shuffle the given columns across samples, as one block.

    A single row permutation is applied to the whole block so that the
    perturbed sample receives a coherent set of values from one other sample
    rather than an incoherent mixture — which matters for the one-hot argmax
    indicators, where an independent shuffle per column could encode two
    winners or none.
    """
    out = raw.copy()
    order = rng.permutation(len(raw))
    out[:, columns] = raw[order][:, columns]
    return out


def permute_many(raw: np.ndarray, blocks: list[list[int]], rng) -> np.ndarray:
    """Shuffle several groups, each with its own independent permutation."""
    out = raw.copy()
    for columns in blocks:
        if columns:
            order = rng.permutation(len(raw))
            out[:, columns] = raw[order][:, columns]
    return out


class Scorer:
    """Top-1 accuracy of one model under a given set of permuted groups.

    Results are cached by coalition, because the Shapley computation revisits
    the same coalition from many orderings and a model evaluation here is the
    dominant cost.
    """

    def __init__(self, kind, model, name, raw, y, groups, scalers, pre, device,
                 seed):
        self.kind, self.model, self.name = kind, model, name
        self.raw, self.y = raw, y
        self.groups = groups
        self.scalers, self.pre, self.device = scalers, pre, device
        self.seed = seed
        self.cache: dict[frozenset, float] = {}
        self.n_calls = 0

    def value(self, kept: frozenset) -> float:
        """Accuracy when only the groups in ``kept`` retain their values."""
        if kept in self.cache:
            return self.cache[kept]

        permuted_blocks = [idx for g, idx in self.groups.items() if g not in kept]
        # Seeded from the coalition itself, so the same coalition yields the
        # same perturbation wherever it is reached and the Shapley increments
        # are differences between comparable quantities.
        rng = np.random.default_rng(self.seed + hash(kept) % 10_000_019)
        raw = (self.raw if not permuted_blocks
               else permute_many(self.raw, permuted_blocks, rng))
        probs = predict_probs(
            self.kind, self.model,
            model_input(self.kind, self.name, raw, self.scalers, self.pre),
            self.device,
        )
        score = float((probs.argmax(axis=1) == self.y).mean())
        self.cache[kept] = score
        self.n_calls += 1
        return score


def necessity_and_sufficiency(scorer: Scorer, repeats: int) -> dict[str, dict]:
    """Accuracy when one group is corrupted, and when only it survives.

    ``repeats`` re-draws the permutation to average out the shuffle; the
    Shapley path uses one draw per coalition instead, relying on the average
    over coalitions.
    """
    names = list(scorer.groups)
    full = frozenset(names)
    baseline = scorer.value(full)
    floor = scorer.value(frozenset())

    out: dict[str, dict] = {
        "_baseline": {"top1": baseline},
        "_floor": {"top1": floor},
    }
    for group in names:
        without, only = [], []
        for r in range(repeats):
            # Fresh scorers per repeat would defeat the cache, so vary the seed
            # and bypass it for these two coalitions only.
            rng = np.random.default_rng(scorer.seed + 1000 * r + hash(group) % 997)
            raw_without = permute(scorer.raw, scorer.groups[group], rng)
            blocks = [idx for g, idx in scorer.groups.items() if g != group]
            raw_only = permute_many(scorer.raw, blocks, rng)
            for raw, sink in ((raw_without, without), (raw_only, only)):
                probs = predict_probs(
                    scorer.kind, scorer.model,
                    model_input(scorer.kind, scorer.name, raw, scorer.scalers,
                                scorer.pre),
                    scorer.device,
                )
                sink.append(float((probs.argmax(axis=1) == scorer.y).mean()))

        out[group] = {
            "top1_permuted": float(np.mean(without)),
            "top1_permuted_std": float(np.std(without)),
            "necessity": baseline - float(np.mean(without)),
            "top1_only": float(np.mean(only)),
            "sufficiency": float(np.mean(only)) - floor,
            "n_features": len(scorer.groups[group]),
        }
    return out


def grouped_shapley(scorer: Scorer) -> dict[str, float]:
    r"""Exact Shapley value of every group, by enumerating all coalitions.

    .. math::
        \varphi_i = \sum_{S \subseteq N \setminus \{i\}}
            \frac{|S|!\,(n-|S|-1)!}{n!}\,\bigl[v(S \cup \{i\}) - v(S)\bigr]

    Exact enumeration is used rather than Monte-Carlo sampling because with
    nine groups the full lattice is 512 coalitions — the same order as a decent
    sampled estimate, and without its variance.

    The efficiency property :math:`\sum_i \varphi_i = v(N) - v(\varnothing)`
    holds by construction and is checked by the caller: it is the reason to pay
    for this measure at all, since necessity and sufficiency do not decompose.
    """
    from itertools import combinations
    from math import factorial

    names = list(scorer.groups)
    n = len(names)
    values: dict[str, float] = {g: 0.0 for g in names}

    for group in names:
        others = [g for g in names if g != group]
        for size in range(n):
            weight = factorial(size) * factorial(n - size - 1) / factorial(n)
            for coalition in combinations(others, size):
                base = frozenset(coalition)
                values[group] += weight * (
                    scorer.value(base | {group}) - scorer.value(base)
                )
    return values


def permutation_groups(pre: dict, report) -> dict[str, list[int]]:
    """Feature groups to permute, covering **every** column.

    The scaler groups stored by stage 9 deliberately exclude the one-hot
    ``pmu_max_dI`` / ``pmu_max_dV`` indicators, because those must not be
    scaled.  Reusing that grouping unchanged would silently exclude them from
    the importance analysis too — and they are not incidental: they encode
    which PMU saw the largest change, which is the model's coarsest and most
    direct localisation cue.

    They are permuted as a single block sharing one row permutation, so each
    perturbed sample receives a coherent set of indicators taken from one other
    sample rather than an incoherent mixture that could encode two winners or
    none.

    The total is checked against ``n_features`` so that any column left
    unaccounted for is reported rather than quietly skipped.
    """
    groups = {g: list(idx) for g, idx in pre["groups"].items() if idx}

    categorical = [
        i for i, name in enumerate(pre["feature_names"])
        if name.startswith(("pmu_max_dI", "pmu_max_dV"))
    ]
    if categorical:
        groups["argmax PMU"] = categorical

    covered = sorted({i for idx in groups.values() for i in idx})
    n_features = pre["n_features"]
    report("  groups: " + ", ".join(f"{g}({len(i)})" for g, i in groups.items()))
    report(f"  coverage: {len(covered)} of {n_features} feature columns")

    if len(covered) != n_features:
        missing = sorted(set(range(n_features)) - set(covered))
        report(f"  WARNING: {len(missing)} column(s) belong to no group and "
               f"will not be permuted: "
               f"{[pre['feature_names'][i] for i in missing[:8]]}")
    return groups


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--dataset", type=config.resolve_path,
                        default=config.features_csv(5))
    parser.add_argument("--cv-dir", type=config.resolve_path, default=config.CV_DIR)
    parser.add_argument("--out-dir", type=config.resolve_path,
                        default=config.ANALYSIS_DIR / "importance")
    # Deep Residual MLP against Random Forest by default: the paper's claim is
    # about inductive families, so the importance profiles worth contrasting
    # are a dense-linear model against an axis-aligned one, not two dense
    # models that differ only in depth.
    parser.add_argument("--models", type=str,
                        default="Deep Residual MLP,Random Forest")
    parser.add_argument("--repeats", type=int,
                        default=config.N_PERMUTATION_REPEATS)
    parser.add_argument("--seed", type=int, default=config.SEED)
    parser.add_argument(
        "--shapley", action="store_true",
        help="Also compute exact grouped Shapley values. These decompose "
             "additively, unlike the other two measures, but cost 2^n model "
             "evaluations per fold.",
    )
    parser.add_argument(
        "--shapley-subsample", type=int, default=0,
        help="Evaluate Shapley on this many test samples per fold (0 = all). "
             "The tree ensembles are slow enough that the full fold is often "
             "impractical; the ranking is stable well below the full set.",
    )
    parser.add_argument(
        "--shapley-folds", type=int, default=0,
        help="Restrict Shapley to the first N folds (0 = all).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config.ensure_dir(args.out_dir)
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    report = Report()

    report("=" * 78)
    report("  PERMUTATION IMPORTANCE BY PHYSICAL FEATURE GROUP")
    report("=" * 78)
    report(f"  device: {device} | repeats per group: {args.repeats}")

    pre = load_preprocessing(args.cv_dir)
    fold_indices = load_fold_indices(args.cv_dir)
    groups = permutation_groups(pre, report)
    encoder = pre["target_encoder"]

    df = pd.read_csv(args.dataset)
    raw_all = build_feature_matrix(df, pre)
    y_all = encoder.transform(df[config.TARGET_COL].to_numpy())

    per_fold: dict[str, dict[int, dict]] = {m: {} for m in models}
    shapley: dict[str, dict[int, dict]] = {m: {} for m in models}
    start = time.time()

    if args.shapley:
        n_coalitions = 2 ** len(groups)
        report(f"\n  Shapley enabled: {n_coalitions} coalitions per model per "
               f"fold ({len(groups)} groups).")
        if args.shapley_subsample:
            report(f"  Shapley evaluated on {args.shapley_subsample} samples "
                   "per fold.")

    for fold in range(pre["n_folds"]):
        scalers_path = Path(args.cv_dir) / f"fold_{fold}" / "scalers.pkl"
        if not scalers_path.exists():
            report(f"\n  fold {fold}: not trained, skipped")
            continue
        scalers = load_scalers(args.cv_dir, fold)
        test_idx = fold_indices[fold]["test"]
        raw_test, y_test = raw_all[test_idx], y_all[test_idx]
        report(f"\n  -- fold {fold} | n_test={len(test_idx)} --")

        for name in models:
            try:
                kind, model = load_fold_model(name, args.cv_dir, fold, pre, device)
            except (FileNotFoundError, KeyError) as exc:
                report(f"    {name}: not available ({exc}), skipped")
                continue

            scorer = Scorer(kind, model, name, raw_test, y_test, groups,
                            scalers, pre, device, args.seed + fold)
            t0 = time.time()
            per_fold[name][fold] = necessity_and_sufficiency(scorer, args.repeats)
            entry = per_fold[name][fold]
            report(f"    {name}: intact Top-1 = {entry['_baseline']['top1']:.4f}, "
                   f"all-permuted floor = {entry['_floor']['top1']:.4f} "
                   f"({time.time() - t0:.0f}s)")

            run_shapley = args.shapley and (
                not args.shapley_folds or fold < args.shapley_folds
            )
            if run_shapley:
                if args.shapley_subsample and args.shapley_subsample < len(test_idx):
                    rng = np.random.default_rng(args.seed + fold)
                    pick = rng.choice(len(test_idx), args.shapley_subsample,
                                      replace=False)
                    sub = Scorer(kind, model, name, raw_test[pick], y_test[pick],
                                 groups, scalers, pre, device, args.seed + fold)
                else:
                    sub = scorer

                # One evaluation, timed, to estimate the full cost before
                # committing to 2^n of them.
                probe = time.time()
                sub.value(frozenset(list(groups)[:1]))
                per_eval = time.time() - probe
                estimate = per_eval * 2 ** len(groups) / 60.0
                report(f"      Shapley: ~{per_eval:.2f}s per coalition, "
                       f"~{estimate:.0f} min for this fold")

                t0 = time.time()
                values = grouped_shapley(sub)
                total = sum(values.values())
                span = sub.value(frozenset(groups)) - sub.value(frozenset())
                shapley[name][fold] = {
                    "values": values, "sum": total, "span": span,
                    "n_evaluations": sub.n_calls,
                }
                report(f"      Shapley done in {(time.time() - t0) / 60:.1f} min "
                       f"| sum={total:.4f} vs v(N)-v(0)={span:.4f} "
                       f"| residual={abs(total - span):.2e}")

            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

    # ── aggregate ─────────────────────────────────────────────────────────
    report("\n" + "=" * 78)
    report("  NECESSITY AND SUFFICIENCY (mean +/- std over folds)")
    report("=" * 78)
    report("    necessity   = Top-1 lost when this group alone is permuted")
    report("    sufficiency = Top-1 gained over the all-permuted floor when "
           "this group alone survives")

    summary: dict[str, dict] = {}
    rows = []
    for name in models:
        folds = per_fold[name]
        if not folds:
            continue
        base_mean, base_std = mean_std(
            [d["_baseline"]["top1"] for d in folds.values()]
        )
        floor_mean, _ = mean_std([d["_floor"]["top1"] for d in folds.values()])
        report(f"\n  {name}  (intact {base_mean:.4f} +/- {base_std:.4f}, "
               f"floor {floor_mean:.4f})")
        report(f"    {'Group':<14}{'n feat':>7}{'necessity':>20}"
               f"{'sufficiency':>14}{'nec/feat':>11}")
        report("    " + "-" * 66)

        entries = {}
        for group in groups:
            necessity = [d[group]["necessity"] for d in folds.values() if group in d]
            sufficiency = [d[group]["sufficiency"] for d in folds.values() if group in d]
            if not necessity:
                continue
            nec, nec_std = mean_std(necessity)
            suf, suf_std = mean_std(sufficiency)
            entries[group] = {
                "necessity": nec, "necessity_std": nec_std,
                "sufficiency": suf, "sufficiency_std": suf_std,
                "n_features": len(groups[group]),
            }
            rows.append({"model": name, "group": group,
                         "n_features": len(groups[group]),
                         "necessity_mean": nec, "necessity_std": nec_std,
                         "sufficiency_mean": suf, "sufficiency_std": suf_std,
                         "necessity_per_feature": nec / max(len(groups[group]), 1)})

        for group, e in sorted(entries.items(), key=lambda kv: -kv[1]["necessity"]):
            report(f"    {group:<14}{e['n_features']:>7}"
                   f"{e['necessity']:>13.4f}+/-{e['necessity_std']:<6.4f}"
                   f"{e['sufficiency']:>14.4f}"
                   f"{e['necessity'] / max(e['n_features'], 1):>11.5f}")
        summary[name] = {"baseline": (base_mean, base_std),
                         "floor": floor_mean, "groups": entries}

    total_necessity = {
        name: sum(e["necessity"] for e in summary[name]["groups"].values())
        for name in summary
    }
    for name, total in total_necessity.items():
        span = summary[name]["baseline"][0] - summary[name]["floor"]
        report(f"\n  {name}: necessities sum to {total:.3f} against an available "
               f"{span:.3f}; they overlap by {total / max(span, 1e-9):.1f}x and "
               "must not be presented as an additive decomposition.")

    # ── Shapley ───────────────────────────────────────────────────────────
    shapley_rows = []
    if args.shapley and any(shapley.values()):
        report("\n" + "=" * 78)
        report("  GROUPED SHAPLEY VALUES (additive by construction)")
        report("=" * 78)
        for name in models:
            folds = shapley[name]
            if not folds:
                continue
            report(f"\n  {name}  ({len(folds)} fold(s), "
                   f"{list(folds.values())[0]['n_evaluations']} evaluations each)")
            report(f"    {'Group':<14}{'Shapley':>18}{'share':>10}")
            report("    " + "-" * 44)
            values = {
                g: mean_std([f["values"][g] for f in folds.values()])
                for g in groups
            }
            total = sum(v[0] for v in values.values())
            for group, (mean, std) in sorted(values.items(), key=lambda kv: -kv[1][0]):
                report(f"    {group:<14}{mean:>11.4f}+/-{std:<6.4f}"
                       f"{mean / max(total, 1e-9):>10.1%}")
                shapley_rows.append({"model": name, "group": group,
                                     "shapley_mean": mean, "shapley_std": std,
                                     "share": mean / max(total, 1e-9)})
            spans = [f["span"] for f in folds.values()]
            residual = max(abs(f["sum"] - f["span"]) for f in folds.values())
            report(f"    {'TOTAL':<14}{total:>11.4f}        "
                   f"  v(N)-v(0) = {np.mean(spans):.4f}")
            report(f"    efficiency check: largest residual {residual:.2e} "
                   "(exact enumeration, so this is float error only)")

    for filename, payload in (
        ("necessity_sufficiency_per_fold.pkl", per_fold),
        ("shapley_per_fold.pkl", shapley),
        ("summary.pkl", summary),
    ):
        with open(args.out_dir / filename, "wb") as fh:
            pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)
    if rows:
        pd.DataFrame(rows).to_csv(args.out_dir / "importance.csv", index=False)
    if shapley_rows:
        pd.DataFrame(shapley_rows).to_csv(args.out_dir / "shapley.csv", index=False)

    report(f"\n  elapsed: {time.time() - start:.0f}s")
    report.save(args.out_dir / "importance.txt")


if __name__ == "__main__":
    main()