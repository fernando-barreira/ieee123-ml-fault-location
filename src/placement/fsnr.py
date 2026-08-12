"""FSNR — forward selection with neighbourhood refinement for PMU placement.

Optimal PMU placement for fault location is a combinatorial problem:
:math:`\\binom{30}{5} = 142\\,506` subsets from the candidate pool alone, each
requiring a model to be trained.  FSNR is a two-stage greedy heuristic:

**Stage 1 — forward selection.**
    Start from the empty set.  At each step, try adding every remaining
    candidate, score each resulting subset, and keep the best.  Cost is
    :math:`\\mathcal{O}(K \\cdot |C|)` evaluations instead of
    :math:`\\binom{|C|}{K}`.

    A useful side effect: the order in which PMUs are added *is* the ranking of
    marginal value.  The K-PMU placement is the length-K prefix of the forward
    path, so a single run yields every scenario from K=1 to K=9 with no extra
    computation, and the scenarios are nested — which is what makes the
    "how much does the K-th PMU buy?" curve meaningful.

**Stage 2 — neighbourhood refinement.**
    Greedy forward selection is myopic: an early choice that looked best on its
    own can become redundant once later PMUs are added.  Refinement revisits
    each selected PMU and tries swapping it for buses within
    :data:`config.FSNR_NEIGHBOR_RADIUS` hops.  Restricting swaps to a
    topological neighbourhood is the physical part of the heuristic: a PMU's
    contribution is dominated by the region it observes, so the useful
    alternatives to a given site are its electrical neighbours, not arbitrary
    buses elsewhere on the feeder.

Every evaluated subset is cached by its sorted membership, so subsets reached
by different paths are scored once.  The cache is written to disk after every
accepted move, making a run resumable — it takes hours on a single GPU.
"""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Callable

import networkx as nx
import numpy as np
import pandas as pd


def subset_key(pmus: list[str]) -> str:
    """Canonical cache key: membership only, order-independent."""
    return " ".join(sorted(str(p) for p in pmus))


def features_for_subset(
    feature_to_pmu: dict[str, str | None],
    pmus: list[str],
    drop_columns: set[str],
) -> list[str]:
    """Columns available to a placement consisting of ``pmus``.

    Per-PMU columns of unselected PMUs are excluded, and the columns in
    ``drop_columns`` (the subset-dependent globals) are excluded outright:
    during the search they are still computed over the full candidate pool, so
    keeping them would let a subset benefit from PMUs it does not contain.
    """
    selected = set(pmus)
    return [
        col
        for col, pmu in feature_to_pmu.items()
        if col not in drop_columns and (pmu is None or pmu in selected)
    ]


def map_columns_to_pmus(
    columns: list[str], target_col: str, global_features: set[str]
) -> dict[str, str | None]:
    """``{column: pmu_id or None}`` for every feature column.

    ``None`` marks a global feature.  The target column is excluded.
    """
    from src.features import extract_pmu_id

    mapping: dict[str, str | None] = {}
    for col in columns:
        if col == target_col:
            continue
        if col in global_features:
            mapping[col] = None
            continue
        mapping[col] = extract_pmu_id(col)
    return mapping


class ProgressStore:
    """Resumable on-disk record of an FSNR run.

    Writes three artefacts to ``output_dir``:

    ``fsnr_progress.json``
        Evaluation cache plus the current state; read on start-up to resume.
    ``fsnr_log.txt``
        Human-readable, timestamped record of accepted moves.
    ``fsnr_log.csv``
        One row per evaluation — the audit trail behind the placement table in
        the paper.
    """

    def __init__(self, output_dir: str | Path) -> None:
        self.dir = Path(output_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.rows: list[dict] = []

    def load(self) -> dict:
        path = self.dir / "fsnr_progress.json"
        if not path.exists():
            return {}
        with open(path) as fh:
            return json.load(fh)

    def save(self, state: dict, message: str | None = None) -> None:
        with open(self.dir / "fsnr_progress.json", "w") as fh:
            json.dump(state, fh, indent=2, default=str)
        if message:
            with open(self.dir / "fsnr_log.txt", "a") as fh:
                fh.write(f"[{datetime.now().isoformat(timespec='seconds')}] {message}\n")

    def record(self, row: dict) -> None:
        self.rows.append(row)

    def flush_csv(self) -> Path:
        path = self.dir / "fsnr_log.csv"
        pd.DataFrame(self.rows).to_csv(path, index=False)
        return path


def forward_selection(
    candidates: list[str],
    k_target: int,
    evaluate: Callable[[list[str]], dict],
    cache: dict[str, dict],
    store: ProgressStore,
) -> tuple[list[str], list[dict]]:
    """Grow the placement one PMU at a time, maximising the aggregate score.

    Parameters
    ----------
    candidates
        Candidate pool.
    k_target
        Number of PMUs to select.  Prefixes of the resulting path are the
        placements for smaller K.
    evaluate
        ``subset -> {"geom", "mlp", "rf", "n_features"}``.
    cache
        Evaluation cache, mutated in place.
    store
        Progress store, for resumability and logging.

    Returns
    -------
    (selected, history)
        ``history`` records, per iteration, the PMU added, the resulting
        placement and the scores of *every* candidate tried — the raw material
        for the marginal-value curve.
    """
    selected: list[str] = []
    history: list[dict] = []

    print(f"\n[forward selection] K={k_target}, |candidates|={len(candidates)}")

    for step in range(k_target):
        print(f"\n  -- iteration {step + 1}/{k_target} (|S| = {len(selected)}) --")
        remaining = [c for c in candidates if c not in selected]
        scores: dict[str, dict] = {}

        for i, candidate in enumerate(remaining, start=1):
            trial = selected + [candidate]
            key = subset_key(trial)

            if key in cache:
                result, elapsed, tag = cache[key], 0.0, "cached"
            else:
                t0 = time.time()
                result = evaluate(trial)
                elapsed = time.time() - t0
                cache[key] = result
                tag = f"{elapsed:.0f}s"

            scores[candidate] = result
            print(
                f"    [{i:>2d}/{len(remaining)}] +{candidate:>4s}  "
                f"MLP={result['mlp']:.4f}  RF={result['rf']:.4f}  "
                f"geom={result['geom']:.4f}  n_feat={result['n_features']}  ({tag})"
            )
            store.record(
                {
                    "phase": "forward",
                    "iteration": step + 1,
                    "subset": " ".join(trial),
                    "candidate_added": candidate,
                    "n_features": result["n_features"],
                    "score_mlp": result["mlp"],
                    "score_rf": result["rf"],
                    "score_geom": result["geom"],
                    "elapsed_s": round(elapsed, 2),
                    "accepted": False,
                }
            )

        best = max(scores, key=lambda c: scores[c]["geom"])
        selected.append(best)
        _mark_accepted(store.rows, phase="forward", iteration=step + 1,
                       candidate_added=best)

        history.append(
            {
                "iteration": step + 1,
                "added": best,
                "selected_after": list(selected),
                "score_geom": scores[best]["geom"],
                "score_mlp": scores[best]["mlp"],
                "score_rf": scores[best]["rf"],
                "all_candidate_scores": {c: r["geom"] for c, r in scores.items()},
            }
        )

        store.save(
            {
                "phase": "forward",
                "cache": cache,
                "n_evaluations": len(cache),
                "forward_history": history,
                "current_selected": selected,
            },
            f"iteration {step + 1}: added PMU {best} "
            f"(geom={scores[best]['geom']:.4f})",
        )
        print(f"  -> added PMU {best}   geom={scores[best]['geom']:.4f}   S={selected}")

    return selected, history


def neighborhood_refinement(
    selected: list[str],
    graph: nx.Graph,
    candidates: list[str],
    evaluate: Callable[[list[str]], dict],
    cache: dict[str, dict],
    store: ProgressStore,
    current_score: float,
    radius: int,
    max_rounds: int,
) -> tuple[list[str], float, list[dict]]:
    """Swap each selected PMU with nearby candidates while the score improves.

    Neighbourhoods come from breadth-first search on the topology graph, i.e.
    from real conductor connections.  Bus labels in the IEEE 123 model are not
    ordered along the feeder — bus 97 and bus 197 are one switch apart while
    bus 96 and bus 97 are on different laterals — so any neighbourhood derived
    from label arithmetic would be meaningless.

    Terminates when a full round produces no improvement or after
    ``max_rounds``.  Since every accepted swap strictly increases the score and
    the number of subsets is finite, termination is guaranteed.
    """
    print(
        f"\n[neighbourhood refinement] start={selected}  radius={radius} hops  "
        f"score={current_score:.4f}"
    )

    history: list[dict] = []
    candidate_set = set(candidates)
    rounds = 0
    improved = True

    while improved and rounds < max_rounds:
        improved = False
        rounds += 1
        print(f"\n  -- round {rounds} --  current score = {current_score:.4f}")

        for position, current_pmu in enumerate(list(selected)):
            if current_pmu not in graph:
                continue

            distances = nx.single_source_shortest_path_length(
                graph, current_pmu, cutoff=radius
            )
            neighbours = [
                bus
                for bus in distances
                if bus in candidate_set and bus not in selected
            ]
            if not neighbours:
                continue

            print(
                f"\n    PMU[{position}] = {current_pmu}: "
                f"testing {len(neighbours)} neighbour(s)"
            )
            best_swap: str | None = None
            best_score = current_score

            for neighbour in neighbours:
                trial = [neighbour if p == current_pmu else p for p in selected]
                key = subset_key(trial)

                if key in cache:
                    result, elapsed, tag = cache[key], 0.0, "cached"
                else:
                    t0 = time.time()
                    result = evaluate(trial)
                    elapsed = time.time() - t0
                    cache[key] = result
                    tag = f"{elapsed:.0f}s"

                marker = "*" if result["geom"] > current_score else " "
                print(
                    f"      {marker} {current_pmu} -> {neighbour}: "
                    f"MLP={result['mlp']:.4f}  RF={result['rf']:.4f}  "
                    f"geom={result['geom']:.4f}  ({tag})"
                )
                store.record(
                    {
                        "phase": "refinement",
                        "iteration": rounds,
                        "subset": " ".join(trial),
                        "swap_from": current_pmu,
                        "swap_to": neighbour,
                        "n_features": result["n_features"],
                        "score_mlp": result["mlp"],
                        "score_rf": result["rf"],
                        "score_geom": result["geom"],
                        "elapsed_s": round(elapsed, 2),
                        "accepted": False,
                    }
                )

                if result["geom"] > best_score:
                    best_score, best_swap = result["geom"], neighbour

            if best_swap is not None:
                selected[selected.index(current_pmu)] = best_swap
                _mark_accepted(
                    store.rows,
                    phase="refinement",
                    iteration=rounds,
                    swap_from=current_pmu,
                    swap_to=best_swap,
                )
                history.append(
                    {
                        "round": rounds,
                        "replaced": current_pmu,
                        "with": best_swap,
                        "score_before": current_score,
                        "score_after": best_score,
                        "selected_after": list(selected),
                    }
                )
                current_score = best_score
                improved = True
                store.save(
                    {
                        "phase": "refinement",
                        "cache": cache,
                        "n_evaluations": len(cache),
                        "refinement_history": history,
                        "current_selected": selected,
                    },
                    f"round {rounds}: {current_pmu} -> {best_swap} "
                    f"(geom={current_score:.4f})",
                )
                print(
                    f"    accepted: {current_pmu} -> {best_swap}  "
                    f"geom={current_score:.4f}"
                )

        if not improved:
            print(f"\n  round {rounds}: no swap improved the score; converged.")

    return selected, current_score, history


def _mark_accepted(rows: list[dict], **match) -> None:
    """Flag the most recent log row matching ``match`` as the accepted move."""
    for row in reversed(rows):
        if all(row.get(k) == v for k, v in match.items()):
            row["accepted"] = True
            return


def geometric_mean(a: float, b: float) -> float:
    """Aggregate score of the two proxies; see the module docstring."""
    return float(np.sqrt(max(a, 0.0) * max(b, 0.0)))
