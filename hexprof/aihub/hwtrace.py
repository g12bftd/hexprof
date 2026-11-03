"""Hardware metrics from AI Hub profile jobs (optrace + QHAS) and operator classification.

AI Hub profile jobs on the QNN/HTP path can carry two extra artifacts:

* ``PROFILE_TRACE`` – QNN HTP optrace, exported as JSON (Chrome-trace style).
* ``QHAS_SUMMARY`` – the QNN HTP Analysis Summary JSON (HMX/HVX utilisation,
  VTCM use, DRAM traffic, dominant path...).

Qualcomm documents neither JSON schema publicly and the field set depends on the
QAIRT version, so this module does *not* hard-code key names. It

1. walks the JSON and lists every record collection it finds (``hexprof hw fields``),
2. maps keys to canonical metrics by name pattern (overridable with ``field_map``),
3. classifies each operator against the measured ceilings:

   * compute-bound  – HMX or HVX utilisation >= ``util_threshold``
   * memory-bound   – achieved DRAM bandwidth >= ``bw_fraction`` x B (the measured ceiling)
   * overhead       – both low (sync, layout changes, dispatch)
   * unknown        – the trace lacks the fields needed to decide

Run ``hexprof hw fields`` on your first Gemma 4 / SigLIP 2 job to confirm which
fields exist before trusting the classification.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

# canonical metric -> list of (all-of) regex fragments matched against lower-cased key paths
DEFAULT_PATTERNS: dict[str, list[tuple[str, ...]]] = {
    "name": [("^name$",), ("op_?name",), ("node_?name",), ("^op$",), ("layer_?name",)],
    "op_type": [("op_?type",), ("^type$",), ("package",)],
    "hmx_util": [("hmx", "util"), ("hmx", "busy"), ("hmx", "active"), ("hmx", "usage")],
    "hvx_util": [("hvx", "util"), ("hvx", "busy"), ("hvx", "active"), ("hvx", "usage")],
    "vtcm_bytes": [("vtcm",)],
    "dram_read_bytes": [("(ddr|dram)", "read")],
    "dram_write_bytes": [("(ddr|dram)", "writ")],
    "dram_bytes": [("(ddr|dram)", "(byte|traffic|total|bw|bandwidth)")],
    "cycles": [("cycle",)],
    "time": [("(time|duration|dur)$",), ("(time|duration)_?(us|ns|ms)",), ("^dur$",)],
}


def load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8", errors="replace"))


# --------------------------------------------------------------------------- discovery


@dataclass
class RecordSet:
    path: str
    records: list[dict]

    @property
    def keys(self) -> list[str]:
        seen: dict[str, None] = {}
        for r in self.records[:200]:
            for k in _flat_keys(r):
                seen.setdefault(k, None)
        return list(seen)


def _flat_keys(d: dict, prefix: str = "") -> Iterable[str]:
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            yield from _flat_keys(v, key + ".")
        else:
            yield key


def _flat_get(d: dict, dotted: str) -> Any:
    cur: Any = d
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def find_record_sets(obj: Any, path: str = "$", min_len: int = 2) -> list[RecordSet]:
    """All lists of dicts in the JSON (e.g. traceEvents, per-op tables)."""
    found: list[RecordSet] = []
    if isinstance(obj, list):
        dicts = [x for x in obj if isinstance(x, dict)]
        if len(dicts) >= min_len and len(dicts) >= 0.8 * len(obj):
            found.append(RecordSet(path, dicts))
        for i, x in enumerate(obj[:50]):
            if isinstance(x, (dict, list)):
                found.extend(find_record_sets(x, f"{path}[{i}]", min_len))
    elif isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, (dict, list)):
                found.extend(find_record_sets(v, f"{path}.{k}", min_len))
    return found


def describe(path: str | Path) -> dict[str, Any]:
    """What a trace file contains: top-level keys and every record collection."""
    obj = load_json(path)
    sets = find_record_sets(obj)
    return {
        "file": str(path),
        "top_level_keys": list(obj.keys()) if isinstance(obj, dict) else f"list[{len(obj)}]",
        "record_sets": [
            {"path": s.path, "count": len(s.records), "keys": s.keys, "canonical": match_fields(s.keys)}
            for s in sorted(sets, key=lambda s: -len(s.records))
        ],
    }


def match_fields(keys: Iterable[str], field_map: dict[str, str] | None = None) -> dict[str, str]:
    """Map canonical metric names to the first matching key."""
    keys = list(keys)
    out: dict[str, str] = {}
    for canon, alternatives in DEFAULT_PATTERNS.items():
        for frags in alternatives:
            hit = next(
                (k for k in keys if all(re.search(f, k.lower()) for f in frags) and k not in out.values()),
                None,
            )
            if hit:
                out[canon] = hit
                break
    if field_map:
        out.update(field_map)
    return out


# --------------------------------------------------------------------------- normalisation


def _num(v: Any) -> float | None:
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        m = re.match(r"^\s*([-+]?\d+(\.\d+)?([eE][-+]?\d+)?)\s*(%)?", v)
        if m:
            return float(m.group(1))
    return None


def _time_to_us(key: str, value: float) -> float:
    k = key.lower()
    if k.endswith("ns") or "_ns" in k:
        return value / 1000.0
    if k.endswith("ms") or "_ms" in k:
        return value * 1000.0
    return value  # Chrome traces and AI Hub per-layer timings are microseconds


def _util(v: float | None) -> float | None:
    if v is None:
        return None
    return v / 100.0 if v > 1.0 else v


def normalize(record_set: RecordSet, field_map: dict[str, str] | None = None) -> list[dict[str, Any]]:
    """Per-op rows with canonical fields; Chrome-trace events are aggregated by name."""
    fm = match_fields(record_set.keys, field_map)
    is_chrome = "ph" in record_set.keys and "dur" in record_set.keys
    rows: dict[str, dict[str, Any]] = {}
    for i, r in enumerate(record_set.records):
        if is_chrome and r.get("ph") not in ("X", None):
            continue
        name = _flat_get(r, fm["name"]) if "name" in fm else None
        name = str(name) if name is not None else f"op_{i}"
        row = rows.setdefault(name, {"name": name, "count": 0})
        row["count"] += 1
        if "op_type" in fm and row.get("op_type") is None:
            row["op_type"] = _flat_get(r, fm["op_type"])
        for canon in ("cycles", "vtcm_bytes", "dram_read_bytes", "dram_write_bytes", "dram_bytes"):
            if canon in fm and (v := _num(_flat_get(r, fm[canon]))) is not None:
                row[canon] = row.get(canon, 0.0) + v if canon != "vtcm_bytes" else max(row.get(canon, 0.0), v)
        for canon in ("hmx_util", "hvx_util"):
            if canon in fm and (v := _util(_num(_flat_get(r, fm[canon])))) is not None:
                row.setdefault(f"_{canon}", []).append(v)
        if "time" in fm and (v := _num(_flat_get(r, fm["time"]))) is not None:
            row["time_us"] = row.get("time_us", 0.0) + _time_to_us(fm["time"], v)
        # Chrome-trace args often hold the counters.
        if is_chrome and isinstance(r.get("args"), dict):
            afm = match_fields(list(_flat_keys(r["args"])), None)
            for canon in ("hmx_util", "hvx_util"):
                if canon in fm:  # already read through the flattened "args.*" key
                    continue
                if canon in afm and (v := _util(_num(_flat_get(r["args"], afm[canon])))) is not None:
                    row.setdefault(f"_{canon}", []).append(v)
            for canon in ("dram_read_bytes", "dram_write_bytes", "dram_bytes", "cycles"):
                if canon in fm:
                    continue
                if canon in afm and (v := _num(_flat_get(r["args"], afm[canon]))) is not None:
                    row[canon] = row.get(canon, 0.0) + v
    out = []
    for row in rows.values():
        for canon in ("hmx_util", "hvx_util"):
            vals = row.pop(f"_{canon}", None)
            if vals:
                row[canon] = sum(vals) / len(vals)
        if "dram_bytes" not in row and ("dram_read_bytes" in row or "dram_write_bytes" in row):
            row["dram_bytes"] = row.get("dram_read_bytes", 0.0) + row.get("dram_write_bytes", 0.0)
        out.append(row)
    return out


def best_record_set(path: str | Path) -> RecordSet | None:
    """Pick the collection that looks most like a per-op table."""
    sets = find_record_sets(load_json(path))
    scored = []
    for s in sets:
        fm = match_fields(s.keys)
        score = sum(1 for k in ("hmx_util", "hvx_util", "dram_bytes", "dram_read_bytes", "time", "cycles") if k in fm)
        scored.append((score + ("name" in fm), len(s.records), s))
    if not scored:
        return None
    scored.sort(key=lambda t: (t[0], t[1]), reverse=True)
    return scored[0][2]


def merge_aihub_timing(rows: list[dict[str, Any]], per_layer: list[dict[str, Any]]) -> None:
    """Fill missing per-op time from AI Hub's per-layer profile (matched by name)."""
    by_name = {str(p["name"]): p for p in per_layer if p.get("name") is not None}
    for r in rows:
        p = by_name.get(r["name"])
        if p is None:
            continue
        if r.get("time_us") is None and p.get("execution_time_us") is not None:
            r["time_us"] = float(p["execution_time_us"])
        if r.get("cycles") is None and p.get("execution_cycles") is not None:
            r["cycles"] = float(p["execution_cycles"])
        r.setdefault("op_type", p.get("type"))
        r.setdefault("compute_unit", p.get("compute_unit"))


# --------------------------------------------------------------------------- classification


def classify(
    rows: list[dict[str, Any]],
    bandwidth_gbps: float | None,
    util_threshold: float = 0.6,
    bw_fraction: float = 0.7,
) -> list[dict[str, Any]]:
    """Label each op compute-bound / memory-bound / overhead / unknown."""
    for r in rows:
        util = max((r.get("hmx_util") or 0.0), (r.get("hvx_util") or 0.0))
        have_util = r.get("hmx_util") is not None or r.get("hvx_util") is not None
        bw = None
        if r.get("dram_bytes") is not None and r.get("time_us"):
            bw = r["dram_bytes"] / (r["time_us"] * 1e-6) / 1e9  # GB/s
            r["achieved_gbps"] = bw
        bw_known = bw is not None and bool(bandwidth_gbps)
        if have_util and util >= util_threshold:
            r["bound"] = "compute"
        elif bw_known and bw >= bw_fraction * bandwidth_gbps:  # type: ignore[operator]
            r["bound"] = "memory"
        elif have_util and bw_known:
            r["bound"] = "overhead"
        elif have_util:
            # Low utilisation but no DRAM figure (or no ceiling): cannot rule out memory-bound.
            r["bound"] = "low-util (no DRAM data)"
        else:
            r["bound"] = "unknown"
    return rows


def class_summary(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    total = sum(r.get("time_us") or 0.0 for r in rows) or 1.0
    out: dict[str, dict[str, float]] = {}
    for r in rows:
        b = out.setdefault(r.get("bound", "unknown"), {"ops": 0, "time_us": 0.0})
        b["ops"] += 1
        b["time_us"] += r.get("time_us") or 0.0
    for b in out.values():
        b["time_share"] = b["time_us"] / total
    return out
