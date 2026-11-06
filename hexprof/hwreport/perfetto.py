"""Perfetto-ready copies of AI Hub's optrace files (``hexprof hw perfetto``).

AI Hub's optrace (``*_profile_trace.json``) is Chrome-trace JSON: process / thread names, one complete
event per HTP op on its thread, memory counters. Perfetto (ui.perfetto.dev) opens that format directly, but
it reads timestamps as µs, and these files carry HTP clock cycles (about 2000 per µs): opened as they are,
every op looks ~2000 times longer than it ran. The copy written here has the same events with ``ts`` /
``dur`` converted to µs, the conversion found the way the hardware report finds it (the trace span matched
against the same graph's QHAS run, in cycles, µs or ns), and with process / thread ids renumbered so that
Perfetto keeps every HTP thread on its own lane (it treats id 0, and a thread id equal to its process id,
specially). Names, nesting and counters are unchanged.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any, Sequence

from hexprof.hwreport import jsonstream
from hexprof.hwreport import optrace as OT
from hexprof.hwreport import qhas as QH


def factor_to_us(trace: Path, qhas: Path | None) -> tuple[float | None, str]:
    """Multiply the trace's timestamps by this to get µs (None: unit not identified), and why."""
    rep = OT.load(trace, bins=10)
    kq = QH.overall_kpis(QH.load_overall(qhas)) if qhas else {}
    OT.calibrate(rep, kq.get("timeline_cycles"), kq.get("time_us"))
    if rep.unit != "µs" or not rep.raw_span or not rep.span:
        return None, rep.unit_note
    return rep.span / rep.raw_span, rep.unit_note


def _renumber(e: dict[str, Any]) -> None:
    """Perfetto treats pid / tid 0 as the idle task and a tid equal to its pid as the main thread, which
    merges HTP threads 0 and 1 of process 1 into one lane. Give every process and thread a distinct id
    that is neither (names come from the trace's own metadata events, so nothing visible changes)."""
    if "pid" in e:
        pid = int(e["pid"])
        e["pid"] = pid + 1
        if "tid" in e:
            e["tid"] = (pid + 1) * 1000 + int(e["tid"]) + 1


def convert(trace: Path, out: Path, factor: float, note: str, compress: bool = False) -> Path:
    """Stream the events into ``out`` with ts / dur in µs."""
    prefix = "item" if jsonstream.top_level_is_list(trace) else "traceEvents.item"
    out.parent.mkdir(parents=True, exist_ok=True)
    opener: Any = gzip.open if compress else open
    with opener(out, "wt", encoding="utf-8") as f:
        f.write('{"displayTimeUnit": "ns", "otherData": ' + json.dumps({"converted_by": "hexprof hw perfetto",
                                                                       "source": str(trace), "note": note}))
        f.write(', "traceEvents": [\n')
        first = True
        for e in jsonstream.items(trace, prefix):
            if not isinstance(e, dict):
                continue
            _renumber(e)
            if "ts" in e:
                e["ts"] = float(e["ts"]) * factor
            if "dur" in e:
                e["dur"] = float(e["dur"]) * factor
            f.write(("" if first else ",\n") + json.dumps(e, separators=(",", ":")))
            first = False
        f.write("\n]}\n")
    return out


def export(paths: Sequence[str | Path], out_dir: str | Path | None = None, compress: bool = False) -> list[Path]:
    """One Perfetto-ready trace per profiled graph found under ``paths``."""
    from hexprof.hwreport import report

    graphs = [g for g in report.discover(paths) if g.optrace]
    if not graphs:
        raise FileNotFoundError("no optrace (*profile_trace*.json) files under " + ", ".join(map(str, paths)))
    dest = Path(out_dir) if out_dir else (Path(paths[0]) if Path(paths[0]).is_dir() else Path(paths[0]).parent) / "perfetto"
    written = []
    for g in graphs:
        factor, note = factor_to_us(g.optrace, g.qhas)
        if factor is None:
            print(f"{g.label}: SKIPPED, time unit not identified ({note})")
            continue
        out = dest / f"{g.label}.perfetto.json{'.gz' if compress else ''}"
        convert(g.optrace, out, factor, note, compress)
        size = out.stat().st_size / 1e6
        print(f"{g.label}: {out} ({size:.1f} MB; timestamps × {factor:.6g} -> µs)")
        written.append(out)
    print(f"\nOpen a file at https://ui.perfetto.dev (Open trace file, or drag and drop). The trace stays in your "
          f"browser: Perfetto's UI processes it locally.")
    return written
