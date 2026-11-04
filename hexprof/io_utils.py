"""Small helpers for result directories and report files."""

from __future__ import annotations

import datetime as _dt
import json
import math
import statistics
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import yaml


def run_dir(base: str | Path | None, prefix: str) -> Path:
    """Create ``base`` (or ``runs/<prefix>-<timestamp>``) and return it."""
    if base is None:
        stamp = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        base = Path("runs") / f"{prefix}-{stamp}"
    path = Path(base)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating,)):
        obj = float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    return obj


def write_json(path: str | Path, data: Any) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(data), indent=2) + "\n", encoding="utf-8")
    return path


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_yaml(path: str | Path, data: Any) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(_jsonable(data), sort_keys=False), encoding="utf-8")
    return path


def summarize(values: Iterable[float]) -> dict[str, float | int | None]:
    """median / min / max / mean / stdev (sample) of a sequence."""
    vals = [float(v) for v in values if v is not None]
    if not vals:
        return {"n": 0, "median": None, "min": None, "max": None, "mean": None, "stdev": None}
    return {
        "n": len(vals),
        "median": statistics.median(vals),
        "min": min(vals),
        "max": max(vals),
        "mean": statistics.fmean(vals),
        "stdev": statistics.stdev(vals) if len(vals) > 1 else 0.0,
    }


def fmt(value: Any, digits: int = 2) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def markdown_table(headers: Sequence[str], rows: Sequence[Sequence[Any]], digits: int = 2) -> str:
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    for row in rows:
        out.append("| " + " | ".join(fmt(c, digits) for c in row) + " |")
    return "\n".join(out)
