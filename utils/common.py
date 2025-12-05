"""Shared utility helpers used across modules."""

from __future__ import annotations

from typing import Any, Iterable, List


def flatten_search_terms(values: Iterable[Any]) -> List[str]:
    """Normalize LLM outputs into a flat list of search keywords."""
    flattened: List[str] = []
    if not values:
        return flattened

    def _append(term: str) -> None:
        cleaned = term.strip()
        if cleaned:
            flattened.append(cleaned)

    for item in values:
        if isinstance(item, str):
            _append(item)
            continue
        if isinstance(item, dict):
            line = item.get("line")
            if isinstance(line, str):
                _append(line)
            for term in item.get("terms", []):
                if isinstance(term, str):
                    _append(term)
            continue
        if isinstance(item, Iterable):
            for term in item:
                if isinstance(term, str):
                    _append(term)
    return flattened
