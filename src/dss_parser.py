"""Static parser for the OpenDSS master file of the IEEE 123-bus feeder.

This module reads ``IEEE123Master.dss`` as *text* and extracts the two pieces
of information the rest of the pipeline needs:

1. the **line list** — which buses are physically connected by a conductor and
   how long that conductor is;
2. the **phase configuration of every bus** — how many phases terminate there
   and which ones (A, B, C).

Both are recovered from a single pass so that the graph builder, the candidate
selector and the analysis scripts can never disagree about the topology.  An
earlier version of this project parsed the file twice with slightly different
regular expressions and produced two incompatible topology files; keeping one
parser is the fix.

Why parse the text at all, when OpenDSS itself is available?
------------------------------------------------------------
The solver gives the *solved* network, which requires a working OpenDSS
installation and a converged power flow.  Topology-only tasks (graph distances,
candidate pre-filtering, sanity checks in the paper) must be reproducible
without running the solver — for instance by a reviewer who only downloaded the
published datasets.  Where both routes exist, the OpenDSS answer is treated as
authoritative and this parser as the cross-check.

Notes on the IEEE 123 model
---------------------------
* Every conductor, including switches, is declared as a ``Line`` object.  A
  switch is flagged with ``Switch=y`` and given a token length.
* Normally-open tie switches terminate on dead-end stub buses (``300_open``,
  ``94_open``).  Removing those buses removes the tie, which is what we want:
  an open tie carries no current, so the far side of it is not electrically
  adjacent no matter how close it is geographically.
* Voltage regulators are ``Transformer`` objects between a bus and its
  ``r``-suffixed twin (``9``/``9r``, ``25``/``25r``, ``150``/``150r``,
  ``160``/``160r``).  They are not parsed as lines; the twins are merged back
  into their parent bus through :data:`config.BUS_ALIAS`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

# ── regular expressions ────────────────────────────────────────────────────
_RE_NEW_LINE = re.compile(r"^\s*new\s+line\.(\S+)", re.IGNORECASE)
_RE_BUS1 = re.compile(r"\bbus1\s*=\s*([\w.]+)", re.IGNORECASE)
_RE_BUS2 = re.compile(r"\bbus2\s*=\s*([\w.]+)", re.IGNORECASE)
_RE_LENGTH = re.compile(r"\blength\s*=\s*([\d.eE+-]+)", re.IGNORECASE)
_RE_PHASES = re.compile(r"\bphases\s*=\s*(\d+)", re.IGNORECASE)
_RE_SWITCH = re.compile(r"\bswitch\s*=\s*(y|yes|true)\b", re.IGNORECASE)
_RE_OPEN_CMD = re.compile(r"^\s*open\s+line\.(\S+)", re.IGNORECASE)


@dataclass(frozen=True)
class DssLine:
    """One ``Line`` object of the ``.dss`` file, after alias resolution.

    Attributes
    ----------
    name
        Object name, lower-cased (``l115``, ``sw7``, ...).
    bus1, bus2
        Canonical bus names after :data:`config.BUS_ALIAS` is applied, or
        ``None`` if the terminal sits on a bus that is removed from the model.
    raw_bus1, raw_bus2
        Bus names exactly as written in the file, including the ``.1.2.3``
        node list.  Needed to recover the phase configuration.
    length
        Conductor length in the units declared by the model (miles for the
        IEEE 123 case).  ``0.0`` for switches and for lines with no ``Length``.
    n_phases
        Value of the ``Phases`` property.
    is_switch
        ``True`` when the object is flagged ``Switch=y``.
    is_open
        ``True`` when the switch is normally open, either because an explicit
        ``Open Line.<name>`` command appears in the file or because the name is
        listed in :data:`config.NORMALLY_OPEN_SWITCHES`.
    """

    name: str
    bus1: str | None
    bus2: str | None
    raw_bus1: str
    raw_bus2: str
    length: float
    n_phases: int
    is_switch: bool
    is_open: bool


@dataclass
class DssModel:
    """Everything the pipeline needs to know about the feeder's topology."""

    lines: list[DssLine] = field(default_factory=list)
    #: ``{bus: {1, 2, 3}}`` — the set of phase nodes present at each bus.
    bus_nodes: dict[str, set[int]] = field(default_factory=dict)
    #: Names of ``Line`` objects that could not be parsed, for diagnostics.
    unparsed: list[str] = field(default_factory=list)

    def bus_phase_count(self) -> dict[str, int]:
        """``{bus: number of phases}``.

        A bus is three-phase when all of A, B and C terminate on it.  This is
        the hard physical requirement for a PMU site in this study: the
        symmetrical-component features need all three phase voltages and
        currents, and a two-phase lateral simply does not provide them.
        """
        return {bus: len(nodes) for bus, nodes in self.bus_nodes.items()}

    def three_phase_buses(self) -> list[str]:
        """Buses where all three phases are present, sorted numerically."""
        out = [b for b, n in self.bus_phase_count().items() if n >= 3]
        return sorted(out, key=_bus_sort_key)

    def electrical_edges(self) -> list[tuple[str, str, float]]:
        """``(bus1, bus2, length)`` for every **closed** conductor.

        Open switches and terminals on removed buses are dropped, so the result
        is the set of edges that can actually carry fault current.
        """
        edges = []
        for ln in self.lines:
            if ln.is_open:
                continue
            if ln.bus1 is None or ln.bus2 is None or ln.bus1 == ln.bus2:
                continue
            edges.append((ln.bus1, ln.bus2, ln.length))
        return edges


def _bus_sort_key(bus: str) -> tuple[int, str]:
    """Sort bus names numerically when possible, alphabetically otherwise."""
    return (int(bus), "") if bus.isdigit() else (10**9, bus)


def _canonical(bus_token: str, alias: dict[str, str | None]) -> str | None:
    """Strip the node list from a bus token and resolve it through ``alias``.

    ``"13.1.2.3"`` -> ``"13"``.  Returns ``None`` for buses that the alias map
    removes from the model.
    """
    bus = bus_token.split(".")[0].strip().lower()
    return alias.get(bus, bus)


def _nodes_of(bus_token: str, declared_phases: int) -> set[int]:
    """Phase nodes present at a bus terminal.

    OpenDSS allows two notations.  ``Bus1=13.1.2.3`` lists the nodes
    explicitly; ``Bus1=13`` with ``Phases=3`` means "the first three nodes",
    i.e. A, B and C.  Reading the explicit node list where available is what
    makes single- and two-phase laterals detectable: a lateral written as
    ``Bus1=1.2`` is a two-phase (A-B) tap, not a three-phase bus.
    """
    parts = bus_token.split(".")
    if len(parts) > 1:
        nodes = {int(p) for p in parts[1:] if p.isdigit()}
        return {n for n in nodes if n in (1, 2, 3)}
    return set(range(1, min(declared_phases, 3) + 1))


def parse_dss(
    dss_path: str | Path,
    alias: dict[str, str | None] | None = None,
    normally_open: set[str] | None = None,
) -> DssModel:
    """Parse the OpenDSS master file into a :class:`DssModel`.

    Parameters
    ----------
    dss_path
        Path to ``IEEE123Master.dss``.
    alias
        Bus-renaming map.  Defaults to :data:`config.BUS_ALIAS`.
    normally_open
        Names of switches to treat as open regardless of what the file says.
        Defaults to :data:`config.NORMALLY_OPEN_SWITCHES`.

    Raises
    ------
    FileNotFoundError
        If ``dss_path`` does not exist.  The OpenDSS test cases are not
        redistributed with this repository; see the README.
    """
    from config import BUS_ALIAS, NORMALLY_OPEN_SWITCHES

    alias = BUS_ALIAS if alias is None else alias
    normally_open = (
        NORMALLY_OPEN_SWITCHES if normally_open is None else normally_open
    )
    normally_open = {s.lower() for s in normally_open}

    dss_path = Path(dss_path)
    if not dss_path.exists():
        raise FileNotFoundError(
            f"OpenDSS master file not found: {dss_path}\n"
            "Set the IEEE123_DSS_MASTER environment variable or copy the file "
            "into network/dss/ (see README, section 'Network model')."
        )

    model = DssModel()
    raw_lines: list[tuple[str, str, str, float, int, bool]] = []
    opened_by_command: set[str] = set()

    text = dss_path.read_text(encoding="utf-8", errors="replace")
    for physical_line in text.splitlines():
        # OpenDSS comments start with '!' and run to the end of the line.
        stmt = physical_line.split("!", 1)[0].strip()
        if not stmt:
            continue

        open_cmd = _RE_OPEN_CMD.match(stmt)
        if open_cmd:
            opened_by_command.add(open_cmd.group(1).lower())
            continue

        m_name = _RE_NEW_LINE.match(stmt)
        if not m_name:
            continue

        name = m_name.group(1).lower()
        m1, m2 = _RE_BUS1.search(stmt), _RE_BUS2.search(stmt)
        if not (m1 and m2):
            # A Line without both terminals cannot contribute an edge; record
            # it instead of dropping it silently, because a missing edge
            # corrupts every hop distance computed downstream.
            model.unparsed.append(name)
            continue

        m_len = _RE_LENGTH.search(stmt)
        m_ph = _RE_PHASES.search(stmt)
        raw_lines.append(
            (
                name,
                m1.group(1),
                m2.group(1),
                float(m_len.group(1)) if m_len else 0.0,
                int(m_ph.group(1)) if m_ph else 3,
                bool(_RE_SWITCH.search(stmt)),
            )
        )

    for name, rb1, rb2, length, n_ph, is_switch in raw_lines:
        b1 = _canonical(rb1, alias)
        b2 = _canonical(rb2, alias)
        is_open = name in normally_open or name in opened_by_command

        model.lines.append(
            DssLine(
                name=name,
                bus1=b1,
                bus2=b2,
                raw_bus1=rb1,
                raw_bus2=rb2,
                length=length,
                n_phases=n_ph,
                is_switch=is_switch,
                is_open=is_open,
            )
        )

        # Phase configuration is accumulated even across open switches: the bus
        # physically has those phases whether or not the tie is closed.
        for raw, canon in ((rb1, b1), (rb2, b2)):
            if canon is None:
                continue
            model.bus_nodes.setdefault(canon, set()).update(_nodes_of(raw, n_ph))

    return model
