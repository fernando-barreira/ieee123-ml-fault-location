"""Stage 7 — extract the scenario dataset for the operating point (K = 5 PMUs).

Takes the candidate-pool feature matrix from stage 4 and the placement from
stage 6 and writes the dataset the models are trained on.

Scope
-----
The paper's operating point is **five PMUs**, and the comparison across sensor
budgets is settled at *proxy* level inside stage 6: the FSNR log already reports
the score of every prefix of the forward path, so the marginal value of each
additional sensor is available without retraining anything.  Averaged over the
PMUs they add, the second four sensors are worth roughly half of the first four,
which is what justifies stopping at five.  Re-training the full model stack at
K = 1 and K = 9 would add cost without adding evidence to that argument, so only
K = 5 is produced by default.

Other budgets remain one flag away — ``--k 1,5,9`` — for follow-up experiments
where the full models, rather than the proxies, need to be compared across
sensor counts.

How the placement for each K is determined
------------------------------------------
``K >= 2``
    The first K buses of the FSNR **forward path**.  Because forward selection
    grows the set one PMU at a time, its prefixes are exactly the placements it
    would have chosen for smaller budgets, and the scenarios are nested — which
    is what makes an accuracy-versus-K curve a fair marginal-value curve rather
    than a comparison of unrelated placements.

``K == 1``
    Pinned to the feeder head (:data:`config.SUBSTATION_BUS`), not taken from
    the forward path.  A single-PMU deployment in practice means "instrument
    the substation": it is the one location that already has metering, requires
    no new communications infrastructure and sees the whole feeder current.
    Reporting a single-PMU baseline at some interior bus would describe a
    deployment nobody would build.

What is copied and what is recomputed
-------------------------------------
Per-PMU features are copied verbatim — each is a function of that PMU's own
phasors, so removing other PMUs cannot change them.  The three aggregate
features (``severity``, ``pmu_max_dI``, ``pmu_max_dV``) are **recomputed** over
the retained PMUs only; copying them would let the scenario benefit from
information gathered by the other 25 candidates.

Row order is preserved, so the frozen splits from stage 5 remain valid.

This stage is the **only** producer of scenario datasets.  Every downstream
consumer — the hyper-parameter search, cross-validated training and the
evaluation suite — reads ``config.features_csv(K)``
(``data/processed/features_{K}pmu.csv``), which guarantees that the PMUs they
train on are exactly the ones the placement search selected.

Usage
-----
.. code-block:: bash

    python scripts/07_build_pmu_subsets.py                  # K = 5
    python scripts/07_build_pmu_subsets.py --k 1,5,9        # follow-up study
    python scripts/07_build_pmu_subsets.py --k 5 --manual 5=1,29,42,80,97
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config
from src.features import select_pmu_subset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--dataset", type=config.resolve_path, default=config.FEATURES_ALL_CSV)
    parser.add_argument(
        "--placement",
        type=config.resolve_path,
        default=config.FSNR_DIR / "pmus_fsnr.json",
        help="pmus_fsnr.json written by stage 6.",
    )
    parser.add_argument("--out-dir", type=config.resolve_path, default=config.PROCESSED_DIR)
    parser.add_argument(
        "--k",
        type=str,
        default="5",
        help="Comma-separated PMU budgets. Defaults to the paper's operating "
             "point (5). Use e.g. --k 1,5,9 for a follow-up comparison across "
             "sensor counts at full-model level.",
    )
    parser.add_argument(
        "--manual",
        action="append",
        default=[],
        metavar="K=BUS,BUS,...",
        help="Override the placement for one K. Repeatable.",
    )
    return parser.parse_args()


def parse_manual(entries: list[str]) -> dict[int, list[str]]:
    """Parse ``--manual 5=1,48,63,83,105`` into ``{5: [...]}``."""
    out: dict[int, list[str]] = {}
    for entry in entries:
        if "=" not in entry:
            raise SystemExit(f"--manual expects K=bus,bus,...; got {entry!r}")
        k_str, buses = entry.split("=", 1)
        out[int(k_str)] = [b.strip() for b in buses.split(",") if b.strip()]
    return out


def load_placement(path: Path) -> tuple[list[str], dict[int, list[str]]]:
    """Read the forward path and any budget-specific refined placement.

    Returns
    -------
    (forward_path, refined)
        ``forward_path`` is the order in which FSNR added PMUs; its length-K
        prefix is the placement at budget K.  ``refined`` maps a budget to the
        placement that neighbourhood refinement produced at that budget, when
        the search recorded one.

    The distinction matters whenever refinement moved a bus that sits early in
    the forward path: the refined set is then *not* the prefix, and taking the
    prefix would report a placement the search did not select.  The refined
    entry wins at its own budget; every other budget falls back to the prefix.
    """
    with open(path) as fh:
        result = json.load(fh)

    forward_path = _forward_path_from(result, path)

    refined: dict[int, list[str]] = {}
    refine_k = result.get("refine_k")
    selected = result.get("refined_placement") or result.get("selected_pmus")
    if refine_k and selected and len(selected) == int(refine_k):
        refined[int(refine_k)] = [str(b) for b in selected]

    return forward_path, refined


def _forward_path_from(result: dict, path: Path) -> list[str]:
    """Extract the order of additions, tolerating every schema ever written."""
    raw = result.get("forward_path")

    # Cumulative form: each entry is itself a list of buses.
    if raw and all(isinstance(entry, (list, tuple)) for entry in raw):
        print(
            f"  Note: {path.name} stores the forward path in cumulative form; "
            "converting to the order of additions."
        )
        converted, seen = [], set()
        for cumulative in raw:
            for bus in cumulative:
                if str(bus) not in seen:
                    seen.add(str(bus))
                    converted.append(str(bus))
        return converted

    if not raw:
        history = result.get("forward_history") or []
        if history:
            def order_key(record: dict) -> int:
                for key in ("iteration", "iter", "step", "k"):
                    if key in record:
                        return int(record[key])
                return 0

            raw = [record["added"] for record in sorted(history, key=order_key)]

    if not raw:
        raise SystemExit(
            f"{path} contains no forward path. Rerun stage 6, or supply every "
            "placement with --manual."
        )

    flat = [str(bus) for bus in raw]
    malformed = [bus for bus in flat if not bus.replace("_", "").isalnum()]
    if malformed:
        raise SystemExit(
            f"{path} yielded bus names that are not plausible identifiers: "
            f"{malformed[:3]}. The file's schema is not the one this script "
            "expects; check it or use --manual."
        )
    return flat


def main() -> None:
    args = parse_args()
    config.ensure_dir(args.out_dir)
    budgets = sorted({int(k.strip()) for k in args.k.split(",") if k.strip()})
    manual = parse_manual(args.manual)

    needs_forward = any(k not in manual and k != 1 for k in budgets)
    forward_path: list[str] = []
    refined: dict[int, list[str]] = {}
    if needs_forward:
        forward_path, refined = load_placement(args.placement)
        print(f"FSNR forward path ({len(forward_path)}): {forward_path}")
        for k, buses in sorted(refined.items()):
            prefix = forward_path[:k]
            changed = sorted(set(prefix) ^ set(buses), key=lambda b: int(b))
            print(f"FSNR refined placement at K={k}: {buses}"
                  + (f"  (differs from the prefix: {changed})" if changed else
                     "  (same as the prefix)"))

    placements: dict[int, tuple[list[str], str]] = {}
    for k in budgets:
        if k in manual:
            buses, source = manual[k], "manual"
        elif k == 1:
            buses, source = [config.SUBSTATION_BUS], "feeder head (pinned)"
        elif k in refined:
            buses, source = refined[k], "FSNR refined (operating point)"
        else:
            if k > len(forward_path):
                raise SystemExit(
                    f"K={k} exceeds the forward path length "
                    f"({len(forward_path)}). Rerun stage 6 with --k {k}, or use "
                    f"--manual {k}=..."
                )
            buses, source = forward_path[:k], "FSNR forward prefix (unrefined)"
        if len(buses) != k:
            raise SystemExit(f"K={k} resolved to {len(buses)} bus(es): {buses}")
        placements[k] = (buses, source)

    print(f"\nReading {args.dataset}")
    df = pd.read_csv(args.dataset)
    print(f"  shape: {df.shape}")
    present = [c for c in config.SUBSET_DEPENDENT_FEATURES if c in df.columns]
    print(f"  aggregate features present in the input: {present}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, dict] = {}

    for k in budgets:
        buses, source = placements[k]
        print(f"\n-- K={k} [{source}]: {buses} --")
        subset = select_pmu_subset(
            df,
            buses,
            target_col=config.TARGET_COL,
            global_features=config.SUBSET_DEPENDENT_FEATURES,
            phases=config.PHASES,
            verbose=True,
        )

        expected = config.expected_n_features(k)
        actual = subset.shape[1] - 1
        if actual != expected:
            print(
                f"    WARNING: {actual} feature columns, expected {expected}."
            )

        out_path = config.features_csv(k) if args.out_dir == config.PROCESSED_DIR \
            else args.out_dir / f"features_{k}pmu.csv"
        subset.to_csv(out_path, index=False)
        print(f"    saved: {out_path}")

        manifest[str(k)] = {
            "pmus": buses,
            "source": source,
            "rows": int(subset.shape[0]),
            "cols": int(subset.shape[1]),
            "csv": str(out_path),
        }

    manifest_path = args.out_dir / "pmu_subsets.json"
    with open(manifest_path, "w") as fh:
        json.dump(
            {
                "input_dataset": str(args.dataset),
                "method": "copy per-PMU features, recompute aggregates",
                "k1_rule": (
                    f"K=1 is pinned to bus {config.SUBSTATION_BUS} (feeder "
                    "head), not taken from the FSNR path"
                ),
                "refined_placements": {str(k): v for k, v in refined.items()},
                "forward_path": forward_path,
                "subsets": manifest,
            },
            fh,
            indent=2,
        )

    print("\n" + "=" * 70)
    for k in budgets:
        info = manifest[str(k)]
        print(f"  K={k:<2d} {info['pmus']}")
        print(f"       {info['rows']:,} x {info['cols']}  ->  {info['csv']}")
    print(f"\n  manifest: {manifest_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()