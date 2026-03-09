from __future__ import annotations

"""
radcopilot.services.logging_service

Shared local logging utilities for RadCopilot Local.

Goals:
- standard-library only
- JSONL append-only operational logging
- consistent event shape across server, report, RAG, and benchmark layers
- safe exception logging with bounded traceback size
- helpers for reading recent entries and lightweight summaries

This module is designed to replace ad hoc JSONL append helpers spread across
other modules as the refactor progresses.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
import threading
import traceback
from typing import Any, Iterable, Mapping, Protocol


DEFAULT_LOG_FILE = "radcopilot_errors.jsonl"
DEFAULT_MAX_DETAIL_CHARS = 4000
DEFAULT_MAX_TRACEBACK_CHARS = 12000
DEFAULT_RECENT_LIMIT = 100
DEFAULT_SOURCE = "radcopilot"


class ConfigLike(Protocol):
    """Minimal config contract expected from the runtime config."""

    base_dir: Path
    log_file: Path


@dataclass(slots=True)
class LogRecord:
    """Canonical log record used throughout the modular codebase."""

    type: str
    detail: str = ""
    source: str = DEFAULT_SOURCE
    level: str = "INFO"
    ts: str = ""
    path: str = ""
    event: str = ""
    context: dict[str, Any] = field(default_factory=dict)
    traceback: str = ""

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "ts": self.ts or utc_now(),
            "type": str(self.type or "EVENT"),
            "level": normalize_level(self.level),
            "source": str(self.source or DEFAULT_SOURCE),
        }
        if self.event:
            payload["event"] = str(self.event)
        if self.path:
            payload["path"] = str(self.path)
        if self.detail:
            payload["detail"] = truncate_text(self.detail, DEFAULT_MAX_DETAIL_CHARS)
        if self.context:
            payload["context"] = sanitize_json(self.context)
        if self.traceback:
            payload["traceback"] = truncate_text(self.traceback, DEFAULT_MAX_TRACEBACK_CHARS)
        return payload


@dataclass(slots=True)
class LogSummary:
    """Simple summary of a log file."""

    file: str
    count: int
    levels: dict[str, int]
    types: dict[str, int]
    latest_ts: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "file": self.file,
            "count": self.count,
            "levels": dict(self.levels),
            "types": dict(self.types),
            "latest_ts": self.latest_ts,
        }


_LOG_LOCKS: dict[str, threading.RLock] = {}
_LOCKS_GUARD = threading.RLock()


def get_log_path(config: ConfigLike | None = None, path: str | Path | None = None) -> Path:
    """Resolve the log file path used by the application."""
    if path is not None:
        return Path(path).expanduser().resolve()
    if config is not None:
        return Path(config.log_file).expanduser().resolve()
    return Path(DEFAULT_LOG_FILE).resolve()


def log_event(
    type: str,
    detail: str = "",
    *,
    config: ConfigLike | None = None,
    path: str | Path | None = None,
    source: str = DEFAULT_SOURCE,
    level: str = "INFO",
    event: str = "",
    route_path: str = "",
    context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Append a structured event to the JSONL log file and return the record."""
    record = LogRecord(
        type=type,
        detail=detail,
        source=source,
        level=level,
        event=event,
        path=route_path,
        context=dict(context or {}),
    ).to_dict()
    append_jsonl(record, config=config, path=path)
    return record


def log_exception(
    exc: BaseException,
    *,
    type: str = "EXCEPTION",
    detail: str = "",
    config: ConfigLike | None = None,
    path: str | Path | None = None,
    source: str = DEFAULT_SOURCE,
    level: str = "ERROR",
    event: str = "",
    route_path: str = "",
    context: Mapping[str, Any] | None = None,
    include_traceback: bool = True,
) -> dict[str, Any]:
    """Append an exception-shaped log record and return the normalized record."""
    parts = [detail.strip()] if detail.strip() else []
    parts.append(f"{type(exc).__name__}: {exc}")
    tb = traceback.format_exc() if include_traceback else ""
    record = LogRecord(
        type=type,
        detail=" | ".join(parts),
        source=source,
        level=level,
        event=event,
        path=route_path,
        context=dict(context or {}),
        traceback=tb,
    ).to_dict()
    append_jsonl(record, config=config, path=path)
    return record


def append_jsonl(
    payload: Mapping[str, Any] | LogRecord,
    *,
    config: ConfigLike | None = None,
    path: str | Path | None = None,
) -> Path:
    """Append one JSON record to the log file safely."""
    log_path = get_log_path(config=config, path=path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    record = payload.to_dict() if isinstance(payload, LogRecord) else sanitize_json(dict(payload))
    record.setdefault("ts", utc_now())
    record.setdefault("level", "INFO")
    record.setdefault("source", DEFAULT_SOURCE)
    record.setdefault("type", "EVENT")

    line = json.dumps(record, ensure_ascii=False) + "\n"
    lock = _get_lock(log_path)
    with lock:
        with log_path.open("a", encoding="utf-8") as fh:
            fh.write(line)
    return log_path


def append_many(
    payloads: Iterable[Mapping[str, Any] | LogRecord],
    *,
    config: ConfigLike | None = None,
    path: str | Path | None = None,
) -> Path:
    """Append many JSONL log records in one locked file write."""
    log_path = get_log_path(config=config, path=path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []
    for item in payloads:
        record = item.to_dict() if isinstance(item, LogRecord) else sanitize_json(dict(item))
        record.setdefault("ts", utc_now())
        record.setdefault("level", "INFO")
        record.setdefault("source", DEFAULT_SOURCE)
        record.setdefault("type", "EVENT")
        lines.append(json.dumps(record, ensure_ascii=False))

    if not lines:
        return log_path

    lock = _get_lock(log_path)
    with lock:
        with log_path.open("a", encoding="utf-8") as fh:
            fh.write("\n".join(lines) + "\n")
    return log_path


def read_recent(
    *,
    config: ConfigLike | None = None,
    path: str | Path | None = None,
    limit: int = DEFAULT_RECENT_LIMIT,
    levels: Iterable[str] | None = None,
    types: Iterable[str] | None = None,
    source: str | None = None,
) -> list[dict[str, Any]]:
    """Return recent log entries in chronological order."""
    log_path = get_log_path(config=config, path=path)
    if not log_path.exists() or not log_path.is_file():
        return []

    limit = max(1, min(5000, int(limit or DEFAULT_RECENT_LIMIT)))
    wanted_levels = {normalize_level(x) for x in levels} if levels else None
    wanted_types = {str(x).strip() for x in types if str(x).strip()} if types else None
    wanted_source = str(source).strip() if source else None

    try:
        raw_lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception:
        return []

    results: list[dict[str, Any]] = []
    for line in reversed(raw_lines):
        line = line.strip()
        if not line:
            continue
        try:
            item = json.loads(line)
        except Exception:
            continue
        if not isinstance(item, dict):
            continue

        level = normalize_level(str(item.get("level") or "INFO"))
        item_type = str(item.get("type") or "EVENT")
        item_source = str(item.get("source") or DEFAULT_SOURCE)

        if wanted_levels and level not in wanted_levels:
            continue
        if wanted_types and item_type not in wanted_types:
            continue
        if wanted_source and item_source != wanted_source:
            continue

        item["level"] = level
        item["type"] = item_type
        item["source"] = item_source
        results.append(item)
        if len(results) >= limit:
            break

    results.reverse()
    return results


def summarize_logs(
    *,
    config: ConfigLike | None = None,
    path: str | Path | None = None,
) -> dict[str, Any]:
    """Return a small aggregate summary of the current log file."""
    log_path = get_log_path(config=config, path=path)
    entries = read_recent(config=config, path=path, limit=5000)
    levels: dict[str, int] = {}
    types_count: dict[str, int] = {}
    latest_ts = ""

    for item in entries:
        level = normalize_level(str(item.get("level") or "INFO"))
        item_type = str(item.get("type") or "EVENT")
        levels[level] = levels.get(level, 0) + 1
        types_count[item_type] = types_count.get(item_type, 0) + 1
        ts = str(item.get("ts") or "")
        if ts and ts > latest_ts:
            latest_ts = ts

    summary = LogSummary(
        file=str(log_path),
        count=len(entries),
        levels=levels,
        types=types_count,
        latest_ts=latest_ts,
    )
    return summary.to_dict()


def count_lines(*, config: ConfigLike | None = None, path: str | Path | None = None) -> int:
    """Return the number of non-empty lines in the log file."""
    log_path = get_log_path(config=config, path=path)
    if not log_path.exists() or not log_path.is_file():
        return 0
    try:
        with log_path.open("r", encoding="utf-8", errors="replace") as fh:
            return sum(1 for line in fh if line.strip())
    except Exception:
        return 0


def utc_now() -> str:
    """Return an ISO-8601 UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


def normalize_level(value: str) -> str:
    """Normalize log level text."""
    value = str(value or "INFO").strip().upper()
    return value if value in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"} else "INFO"


def truncate_text(value: Any, max_chars: int) -> str:
    """Return a bounded string suitable for JSONL storage."""
    text = str(value or "")
    if len(text) <= max_chars:
        return text
    return text[: max(0, max_chars - 3)] + "..."


def sanitize_json(value: Any) -> Any:
    """Coerce common Python values into JSON-safe equivalents."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(k): sanitize_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [sanitize_json(v) for v in value]
    if hasattr(value, "to_dict"):
        try:
            return sanitize_json(value.to_dict())
        except Exception:
            return str(value)
    return str(value)


def _get_lock(path: Path) -> threading.RLock:
    key = str(path)
    with _LOCKS_GUARD:
        lock = _LOG_LOCKS.get(key)
        if lock is None:
            lock = threading.RLock()
            _LOG_LOCKS[key] = lock
        return lock


__all__ = [
    "DEFAULT_LOG_FILE",
    "DEFAULT_MAX_DETAIL_CHARS",
    "DEFAULT_MAX_TRACEBACK_CHARS",
    "DEFAULT_RECENT_LIMIT",
    "DEFAULT_SOURCE",
    "ConfigLike",
    "LogRecord",
    "LogSummary",
    "append_jsonl",
    "append_many",
    "count_lines",
    "get_log_path",
    "log_event",
    "log_exception",
    "normalize_level",
    "read_recent",
    "sanitize_json",
    "summarize_logs",
    "truncate_text",
    "utc_now",
]
