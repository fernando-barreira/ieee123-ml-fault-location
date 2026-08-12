"""Stage 3 — Monte-Carlo short-circuit campaign on the IEEE 123-bus feeder.

Runs :data:`config.N_SIMULATIONS` independent fault scenarios.  Each one draws
an operating point (irradiance, load level), solves the unfaulted network to
capture the pre-fault phasors, applies a fault, solves again, and records what
every candidate PMU would have measured.

Runtime is dominated by the 200 000 power-flow solutions (two per scenario);
expect few hours.  Results are appended to disk every
:data:`config.CHECKPOINT_EVERY` rows, and ``--resume`` continues an interrupted
run.

Usage
-----
.. code-block:: bash

    python scripts/03_run_simulations.py
    python scripts/03_run_simulations.py -n 2000 --out data/raw/smoke.csv
    python scripts/03_run_simulations.py --resume
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config
from src.graph import load_graph
from src.pmu_candidates import load_candidates
from src.simulation import (
    apply_fault,
    build_branch_map,
    buses_by_phase_count,
    choose_fault_phases,
    draw_fault_resistance,
    insert_pv,
    read_bus_current,
    record_pmu_measurements,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--dss", type=config.resolve_path, default=config.DSS_MASTER)
    parser.add_argument("--candidates", type=config.resolve_path, default=config.CANDIDATES_PKL)
    parser.add_argument("--graph", type=config.resolve_path, default=config.GRAPH_PKL)
    parser.add_argument("--out", type=config.resolve_path, default=config.RAW_DATASET_CSV)
    parser.add_argument(
        "-n", "--n-simulations", type=int, default=config.N_SIMULATIONS
    )
    parser.add_argument("--seed", type=int, default=config.SEED)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Append to an existing output file instead of overwriting it.",
    )
    return parser.parse_args()


def validate_pmu_buses(dss, buses: list[str]) -> list[str]:
    """Keep only buses that exist in the solved circuit and are three-phase.

    OpenDSS is the authority here: the static parser is a cross-check, but the
    solver knows how many nodes are actually energised at each bus.
    """
    valid = []
    for bus in buses:
        try:
            dss.circuit.set_active_bus(bus)
            n_nodes = dss.bus.num_nodes
        except Exception as exc:  # pragma: no cover - depends on OpenDSS build
            print(f"  {bus}: could not activate ({exc}) -- dropped")
            continue
        if n_nodes < 3:
            print(f"  {bus}: only {n_nodes} phase(s) -- dropped")
            continue
        valid.append(bus)
    return valid


def main() -> None:
    args = parse_args()

    # Checked before anything expensive: a stage that runs for hours must fail
    # in the first second if it cannot write its output, not after the setup
    # phase has already burned the OpenDSS solves.
    config.ensure_parent(args.out)
    print(f"Output file: {args.out}")

    try:
        import py_dss_interface
    except ImportError:  # pragma: no cover
        raise SystemExit(
            "py_dss_interface is required for this stage.\n"
            "  pip install py_dss_interface\n"
            "It also needs a working OpenDSS installation."
        )

    py_rng = random.Random(args.seed)
    np_rng = np.random.RandomState(args.seed)

    pmu_buses = [str(b) for b in load_candidates(args.candidates)["candidates"]]
    print(f"Candidate PMU sites ({len(pmu_buses)}): {pmu_buses}")

    from src.baselines import compile_circuit

    circuit = str(args.dss)

    # ── nominal solve: validate the PMU sites and fix the monitored branches ──
    print("\nNominal solve (no fault, no PV) for setup...")
    dss = compile_circuit(py_dss_interface.DSS(), args.dss)

    pmu_buses = validate_pmu_buses(dss, pmu_buses)
    print(f"Three-phase PMU sites confirmed: {len(pmu_buses)}")

    import networkx as nx

    graph = load_graph(args.graph)
    anchor = (
        config.SUBSTATION_BUS
        if config.SUBSTATION_BUS in graph
        else min(graph.nodes(), key=lambda b: int(b) if b.isdigit() else 10**9)
    )
    hops_to_source = nx.single_source_shortest_path_length(graph, anchor)

    branch_map = build_branch_map(dss, pmu_buses, hops_to_source)
    print("\nMonitored branch per PMU (upstream line, fixed for the whole run):")
    for bus in pmu_buses:
        branch = branch_map[bus]
        print(f"  bus {bus:>5s} -> Line.{branch.line_name} "
              f"(terminal offset {branch.terminal_offset}, "
              f"{branch.n_phases} phases)")

    fault_types = list(config.FAULT_TYPE_WEIGHTS)
    fault_weights = [config.FAULT_TYPE_WEIGHTS[t] for t in fault_types]

    # Re-seed so the main loop is independent of how many probing calls the
    # setup phase happened to make.
    py_rng = random.Random(args.seed)
    np_rng = np.random.RandomState(args.seed)

    append_mode = args.resume and args.out.exists()
    n_written = sum(1 for _ in open(args.out)) - 1 if append_mode else 0
    if append_mode:
        print(f"\nResuming: {n_written} row(s) already in {args.out}")

    buffer: list[dict] = []
    n_diverged = 0
    n_skipped = 0
    t_start = time.time()

    print(f"\nRunning {args.n_simulations} simulations...")
    for i in range(args.n_simulations):
        dss.text("clear")
        dss.text(f"compile [{circuit}]")

        # ── operating point ───────────────────────────────────────────────
        irradiance_global = py_rng.uniform(*config.DG_IRRADIANCE_RANGE)
        dg_state = []
        for name, bus, kva in config.DG_UNITS:
            jitter = py_rng.uniform(
                -config.DG_IRRADIANCE_JITTER, config.DG_IRRADIANCE_JITTER
            )
            irradiance = min(
                config.DG_IRRADIANCE_CLIP[1],
                max(config.DG_IRRADIANCE_CLIP[0], irradiance_global + jitter),
            )
            insert_pv(dss, name, bus, kva, irradiance, config.VBASE_LL_KV)
            dg_state.append((bus, kva, irradiance))

        load_mult = py_rng.uniform(*config.LOAD_MULT_RANGE)
        dss.text(f"set maxiterations={config.SOLVER_MAX_ITER_BASE}")
        dss.text(f"set tolerance={config.SOLVER_TOLERANCE}")
        dss.text(f"set loadmult={load_mult}")
        dss.text("solve")
        if not dss.solution.converged:
            # The pre-fault snapshot comes from this solution, so an
            # unconverged operating point would supply meaningless reference
            # phasors to every superimposed quantity of the scenario.  The
            # scenario is discarded rather than recorded with a bad baseline.
            n_diverged += 1
            continue

        # ── pre-fault snapshot ────────────────────────────────────────────
        v_pre, i_pre = {}, {}
        for bus in pmu_buses:
            dss.circuit.set_active_bus(bus)
            v_pre[bus] = list(dss.bus.vmag_angle)
            i_pre[bus] = read_bus_current(dss, bus, branch_map)

        # ── draw the fault ────────────────────────────────────────────────
        by_phases = buses_by_phase_count(dss, config.FICTITIOUS_BUSES)
        fault_type = py_rng.choices(fault_types, weights=fault_weights, k=1)[0]

        # A fault can only involve phases that exist at the bus.
        if fault_type == "LLL":
            pool = by_phases[3]
        elif fault_type in ("LL", "LLG"):
            pool = by_phases[2] + by_phases[3]
        else:
            pool = by_phases[1] + by_phases[2] + by_phases[3]
        if not pool:
            n_skipped += 1
            continue

        fault_bus = py_rng.choice(pool)
        dss.circuit.set_active_bus(fault_bus)
        available_phases = [str(n) for n in dss.bus.nodes]
        phases = choose_fault_phases(fault_type, available_phases, py_rng)
        rf = draw_fault_resistance(fault_type, config.FAULT_RESISTANCE, np_rng)

        dss.text(f"set maxiterations={config.SOLVER_MAX_ITER_FAULT}")
        dss.text(f"set tolerance={config.SOLVER_TOLERANCE}")
        apply_fault(dss, fault_bus, fault_type, phases, rf)
        dss.text("solve")
        if not dss.solution.converged:
            n_diverged += 1
            continue

        # ── record ────────────────────────────────────────────────────────
        row = {
            "tipo_falta": fault_type,
            "fases": ".".join(phases),
            "barra_falta": fault_bus,
            "impedancia_falta": rf,
            "fator_carga": load_mult,
            "irradiancia_global": irradiance_global,
        }
        for k, (bus, kva, irradiance) in enumerate(dg_state):
            row[f"gd{k}_bus"] = bus
            row[f"gd{k}_kw"] = kva * irradiance
            row[f"gd{k}_irradiancia"] = irradiance

        for bus in pmu_buses:
            dss.circuit.set_active_bus(bus)
            record_pmu_measurements(
                row,
                bus,
                v_pre=v_pre[bus],
                v_fault=list(dss.bus.vmag_angle),
                i_pre=i_pre[bus],
                i_fault=read_bus_current(dss, bus, branch_map),
            )
        buffer.append(row)

        if len(buffer) >= config.CHECKPOINT_EVERY:
            _flush(buffer, args.out, append=append_mode or n_written > 0)
            n_written += len(buffer)
            buffer.clear()
            rate = n_written / max(time.time() - t_start, 1e-9)
            print(f"  {n_written:>7,} rows written  ({rate:.1f} rows/s)")

    if buffer:
        _flush(buffer, args.out, append=append_mode or n_written > 0)
        n_written += len(buffer)

    elapsed = time.time() - t_start
    print(f"\nDone in {elapsed / 60:.1f} min")
    print(f"  rows written        : {n_written:,}")
    print(f"  non-converged solves: {n_diverged:,}")
    print(f"  skipped (no bus)    : {n_skipped:,}")

    df = pd.read_csv(args.out)
    print(f"\n{args.out}")
    print(f"  shape   : {df.shape}")
    print(f"  classes : {df['barra_falta'].nunique()}")
    print(f"  fault type mix:\n{df['tipo_falta'].value_counts(normalize=True)}")


def _flush(rows: list[dict], path: Path, append: bool) -> None:
    """Append buffered rows to the CSV."""
    frame = pd.DataFrame(rows)
    frame.to_csv(
        path,
        index=False,
        mode="a" if append else "w",
        header=not append,
    )


if __name__ == "__main__":
    main()
