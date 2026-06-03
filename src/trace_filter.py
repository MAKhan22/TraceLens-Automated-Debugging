"""CLI trace id filtering (--trace accepts one id or comma-separated list)."""

from __future__ import annotations


def parse_trace_ids(trace_arg: str | None) -> frozenset[str] | None:
    """
    Parse --trace value.

    Examples:
        "imdb" -> frozenset({"imdb"})
        "amazon,hackernews,imdb" -> frozenset({...})
        "amazon, hackernews" -> same
    """
    if not trace_arg or not str(trace_arg).strip():
        return None
    ids: set[str] = set()
    for part in str(trace_arg).replace(" ", ",").split(","):
        tid = part.strip()
        if tid:
            ids.add(tid)
    return frozenset(ids) if ids else None


def trace_selected(trace_id: str, selected: frozenset[str] | None) -> bool:
    return selected is None or trace_id in selected
