"""Stream items out of very large JSON files (QHAS / optrace can be several GB)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

try:  # C-backed streaming parser: constant memory, fast
    import ijson  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - exercised only without ijson
    ijson = None


def _walk(obj: Any, parts: list[str]) -> Iterator[Any]:
    if not parts:
        yield obj
        return
    head, rest = parts[0], parts[1:]
    if head == "item":
        if isinstance(obj, list):
            for x in obj:
                yield from _walk(x, rest)
    elif isinstance(obj, dict) and head in obj:
        yield from _walk(obj[head], rest)


_cache: dict[str, Any] = {}


def items(path: str | Path, prefix: str) -> Iterator[Any]:
    """Yield every value at ``prefix`` (ijson syntax, e.g. ``data.qnn_op_types.data.item``)."""
    if ijson is not None:
        with open(path, "rb") as f:
            yield from ijson.items(f, prefix, use_float=True)
        return
    key = str(Path(path).resolve())  # fallback: whole file in memory (install ijson for big files)
    if key not in _cache:
        _cache.clear()
        _cache[key] = json.loads(Path(path).read_text(encoding="utf-8", errors="replace"))
    yield from _walk(_cache[key], prefix.split(".") if prefix else [])


def first(path: str | Path, prefix: str, default: Any = None) -> Any:
    for x in items(path, prefix):
        return x
    return default


def top_level_is_list(path: str | Path) -> bool:
    with open(path, "rb") as f:
        while True:
            ch = f.read(1)
            if not ch:
                return False
            if not ch.isspace():
                return ch == b"["
