from __future__ import annotations

"""
radcopilot.rag.startup_index

Startup hook for RadCopilot RAG initialization.

This module is intentionally simple and safe:
- it rebuilds the in-memory retrieval index from the persistent RAG library
- it does not fail if the library file does not exist yet
- it returns a JSON-serializable summary that the launcher can log

Expected caller:
    from radcopilot.rag.startup_index import build_startup_index
    result = build_startup_index(config)
"""

from pathlib import Path
from typing import Any, Protocol

from .library import get_library_path, rebuild_index


class ConfigLike(Protocol):
    """Minimal config contract expected by build_startup_index()."""

    base_dir: Path


def build_startup_index(config: ConfigLike | None = None) -> dict[str, Any]:
    """
    Rebuild the in-memory retrieval index from the persistent library.

    This is safe to call during startup even when no library file exists yet.
    In that case, the result will simply report zero records.
    """
    summary = rebuild_index(config=config)
    summary.setdefault("ok", True)
    summary.setdefault("count", 0)
    summary.setdefault("library_file", str(get_library_path(config=config)))
    summary.setdefault("modalities", {})
    return summary


__all__ = ["build_startup_index"]
