"""Stage 0 — preflight check.

Verifies, in a few seconds, everything the pipeline needs before any long stage
is launched: package versions, the OpenDSS model and engine, GPU availability,
and — the failure this exists to catch — that every output directory is
actually writable.

Run it after cloning and any time something behaves oddly.

Usage
-----
.. code-block:: bash

    python scripts/00_check_setup.py
"""

from __future__ import annotations

import os
import platform
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config

OK, WARN, FAIL = "[ ok ]", "[warn]", "[FAIL]"
_problems: list[str] = []
_warnings: list[str] = []


def fail(message: str) -> None:
    print(f"{FAIL} {message}")
    _problems.append(message)


def warn(message: str) -> None:
    print(f"{WARN} {message}")
    _warnings.append(message)


def check_python() -> None:
    print("\n--- interpreter ---")
    print(f"{OK} Python {platform.python_version()} on {platform.system()}")
    if sys.version_info < (3, 10):
        fail("Python 3.10 or newer is required.")
    print(f"{OK} executable: {sys.executable}")
    if sys.prefix == sys.base_prefix:
        warn("Not running inside a virtual environment.")


def check_packages() -> None:
    print("\n--- packages ---")
    required = ["numpy", "pandas", "sklearn", "networkx"]
    optional = {
        "torch": "stage 6 and the neural models",
        "py_dss_interface": "stage 3 (simulation)",
        "matplotlib": "figures",
        "optuna": "stage 8 (hyper-parameter search)",
        "lightgbm": "stage 9 (tree-based models)",
    }

    for name in required:
        try:
            module = __import__(name)
            print(f"{OK} {name} {getattr(module, '__version__', '?')}")
        except ImportError:
            fail(f"{name} is missing (pip install -r requirements.txt)")

    for name, purpose in optional.items():
        try:
            module = __import__(name)
            print(f"{OK} {name} {getattr(module, '__version__', '?')}")
        except ImportError:
            warn(f"{name} is missing — needed for {purpose}")


def check_gpu() -> None:
    print("\n--- gpu ---")
    try:
        import torch
    except ImportError:
        return

    build = torch.version.cuda
    print(f"{OK} torch {torch.__version__} "
          f"(CUDA build: {build if build else 'none — CPU-only wheel'})")

    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"{OK} CUDA available: {name} ({total:.1f} GiB)")
        if not torch.cuda.is_bf16_supported():
            warn(
                "This GPU does not support bfloat16. The FT-Transformer needs "
                "BF16: its attention softmax over ~450 tokens overflows the "
                "FP16 range."
            )
        return

    # No CUDA at run time. The two causes need very different fixes, and the
    # distinguishing signal is whether the installed wheel was built with CUDA
    # at all.
    if build is None:
        warn(
            "A CPU-only build of PyTorch is installed, so the GPU cannot be "
            "used no matter what hardware is present.\n"
            "         On Windows, `pip install torch` pulls the CPU wheel from "
            "PyPI by default.\n"
            "         Reinstall from the CUDA index (pick the build matching "
            "your driver at https://pytorch.org/get-started/locally/):\n"
            "           pip uninstall -y torch\n"
            "           pip install torch --index-url "
            "https://download.pytorch.org/whl/cu126"
        )
    else:
        warn(
            f"PyTorch was built with CUDA {build} but no device is visible at "
            "run time.\n"
            "         Check that `nvidia-smi` works and reports a driver "
            "recent enough for this build; on laptops, confirm the discrete "
            "GPU is not disabled in the power profile."
        )

    if sys.version_info >= (3, 13):
        warn(
            f"Python {platform.python_version()} is very new. CUDA wheels for "
            "PyTorch, LightGBM and pytorch-tabnet often lag by several months, "
            "and pip silently falls back to whatever build does exist.\n"
            "         If the CUDA install above fails to find a wheel, create "
            "the environment on Python 3.12 instead:\n"
            "           py -3.12 -m venv .venv\n"
            "           .venv\\Scripts\\activate\n"
            "           pip install -r requirements.txt"
        )


def check_network_model() -> None:
    print("\n--- network model ---")

    # Distinguishing "the variable is unset" from "the variable points
    # somewhere wrong" matters: the two have completely different fixes, and
    # the second is easy to mistake for the first after a permanent
    # SetEnvironmentVariable that the current process has not picked up.
    override = os.environ.get("IEEE123_DSS_MASTER")
    if override:
        print(f"{OK} IEEE123_DSS_MASTER = {override}")
    else:
        print("       IEEE123_DSS_MASTER is not set in this process; "
              "falling back to the repository default")

    if not config.DSS_MASTER.exists():
        lines = [
            f"OpenDSS master file not found:\n         {config.DSS_MASTER}",
        ]
        if override:
            parent = Path(override).parent
            lines.append(
                "         The variable IS set, so the path itself is wrong. "
                f"Its folder {'exists' if parent.exists() else 'does not exist'}"
                f": {parent}"
            )
            if parent.exists():
                siblings = sorted(p.name for p in parent.glob("*.dss"))[:6]
                lines.append(f"         .dss files found there: "
                             f"{siblings or 'none'}")
        else:
            lines.append(
                "         Set it for THIS session:\n"
                '           $env:IEEE123_DSS_MASTER = "C:\\Program Files\\'
                'OpenDSS\\IEEETestCases\\123Bus\\IEEE123Master.dss"'
            )
            lines.append(
                "         Or permanently (writes to the registry):\n"
                '           [Environment]::SetEnvironmentVariable('
                '"IEEE123_DSS_MASTER", "<path>", "User")\n'
                "         A permanent setting only reaches processes started "
                "AFTERWARDS. Restart the IDE, not just the terminal tab: the "
                "terminal inherits its environment from the IDE process."
            )
        fail("\n".join(lines))
        return
    print(f"{OK} {config.DSS_MASTER}")

    try:
        from src.dss_parser import parse_dss

        model = parse_dss(config.DSS_MASTER)
    except Exception as exc:  # noqa: BLE001 - report anything the parser hits
        fail(f"Could not parse the model: {type(exc).__name__}: {exc}")
        return

    print(f"{OK} {len(model.lines)} Line objects parsed")
    if model.unparsed:
        warn(f"{len(model.unparsed)} Line object(s) had no Bus1/Bus2: "
             f"{model.unparsed[:5]}")

    three_phase = model.three_phase_buses()
    print(f"{OK} {len(three_phase)} three-phase buses")
    if len(three_phase) < config.N_CANDIDATES:
        fail(
            f"Only {len(three_phase)} three-phase buses, but the candidate "
            f"pool needs {config.N_CANDIDATES}."
        )

    try:
        from src.graph import build_graph, validate_graph

        stats = validate_graph(
            build_graph(config.DSS_MASTER, model=model),
            expected_nodes=config.EXPECTED_N_CLASSES,
        )
    except Exception as exc:  # noqa: BLE001
        fail(f"Topology graph invalid: {type(exc).__name__}: {exc}")
        return

    print(f"{OK} graph: {stats['n_nodes']} buses, {stats['n_edges']} conductors, "
          f"radial={stats['is_tree']}, diameter={stats['diameter_hops']} hops")
    if not stats["is_tree"]:
        warn("The graph contains a loop; check the normally-open tie switches.")


def check_engine() -> None:
    print("\n--- opendss engine ---")
    try:
        import py_dss_interface
    except ImportError:
        warn("py_dss_interface not installed — stage 3 cannot run.")
        return
    if not config.DSS_MASTER.exists():
        return
    try:
        dss = py_dss_interface.DSS()
        dss.text("clear")
        dss.text(f"compile [{config.DSS_MASTER}]")
        dss.text("solve")
        if dss.solution.converged:
            print(f"{OK} engine solved the base case "
                  f"({len(dss.circuit.buses_names)} buses)")
        else:
            fail("The engine compiled the model but the base case diverged.")
    except Exception as exc:  # noqa: BLE001
        fail(f"OpenDSS engine failed: {type(exc).__name__}: {exc}")


def check_directories() -> None:
    print("\n--- directories ---")
    override = os.environ.get("IEEE123_DATA_DIR")
    print(f"{OK} repository root: {config.REPO_ROOT}")
    print(f"{OK} data root      : {config.DATA_DIR}"
          f"{' (from IEEE123_DATA_DIR)' if override else ''}")

    for label, path in (
        ("raw", config.RAW_DIR),
        ("interim", config.INTERIM_DIR),
        ("processed", config.PROCESSED_DIR),
        ("results", config.RESULTS_DIR),
    ):
        try:
            config.ensure_dir(path)
            print(f"{OK} writable: {label:<10s} {path}")
        except SystemExit as exc:
            fail(str(exc).splitlines()[0] + f" ({label})")

    print(f"{OK} free space check: run manually if the disk is tight — the raw "
          "dataset for 100 000 scenarios is several GB.")


def check_artefacts() -> None:
    print("\n--- pipeline artefacts ---")
    stages = [
        ("1  graph", config.GRAPH_PKL),
        ("2  candidates", config.CANDIDATES_PKL),
        ("3  raw dataset", config.RAW_DATASET_CSV),
        ("4  features", config.FEATURES_ALL_CSV),
        ("5  splits", config.SPLITS_PKL),
        ("6  placement", config.FSNR_DIR / "pmus_fsnr.json"),
        ("7  K=5 dataset", config.features_csv(5)),
    ]
    for label, path in stages:
        if path.exists():
            size = path.stat().st_size
            unit = f"{size / 1024**2:.1f} MiB" if size > 1024**2 else f"{size / 1024:.0f} KiB"
            print(f"{OK} stage {label:<16s} present ({unit})")
        else:
            print(f"       stage {label:<16s} not yet produced")


def main() -> None:
    import argparse

    argparse.ArgumentParser(
        description=(
            "Preflight check: package versions, OpenDSS model and engine, GPU, "
            "and writability of every output directory. Exits non-zero if a "
            "blocking problem is found."
        )
    ).parse_args()

    print("=" * 70)
    print("IEEE 123-bus ML fault location — preflight check")
    print("=" * 70)

    check_python()
    check_packages()
    check_gpu()
    check_directories()
    check_network_model()
    check_engine()
    check_artefacts()

    print("\n" + "=" * 70)
    if _problems:
        print(f"{len(_problems)} blocking problem(s):")
        for message in _problems:
            print(f"  - {message.splitlines()[0]}")
    if _warnings:
        print(f"{len(_warnings)} warning(s) — stages that need them will fail:")
        for message in _warnings:
            print(f"  - {message.splitlines()[0]}")
    if not _problems and not _warnings:
        print("Everything checks out. Start with scripts/01_build_graph.py")
    print("=" * 70)
    sys.exit(1 if _problems else 0)


if __name__ == "__main__":
    main()