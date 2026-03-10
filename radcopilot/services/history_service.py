from __future__ import annotations

"""
radcopilot.services.history_service

Local history persistence for RadCopilot.

Why this module exists
----------------------
The current config surface already exposes UI history settings, but the visible
services package does not yet provide a dedicated history implementation. This
module fills that gap with a standard-library-only local persistence layer that:

- stores report history on disk in JSON
- respects config.ui.save_history and config.ui.max_history_items when present
- supports append, upsert, list, get, delete, clear, export, and search
- is safe for local workstation use with atomic writes
- remains tolerant of older or partial payloads from the UI/server layer

Design goals
------------
- no third-party dependencies
- JSON-serializable inputs/outputs
- conservative trimming so very large reports do not bloat local history
- flexible enough to work with both the modular AppConfig object and simpler
  launcher-style configs
"""

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import tempfile
import threading
from typing import Any, Iterable, Mapping, MutableMapping, Protocol, Sequence
import uuid


DEFAULT_HISTORY_FILENAME = "report_history.json"
DEFAULT_HISTORY_SUBDIR = "history"

MAX_TITLE_CHARS = 160
MAX_MODE_CHARS = 40
MAX_TEMPLATE_ID_CHARS = 80
MAX_MODALITY_CHARS = 80
MAX_MODEL_NAME_CHARS = 120
MAX_TEXT_CHARS = 20_000
MAX_REPORT_CHARS = 40_000
MAX_METADATA_ITEMS = 100
MAX_TAGS = 20
MAX_TAG_CHARS = 40
MAX_SEARCH_QUERY_CHARS = 200
MAX_LIST_LIMIT = 500
DEFAULT_EXPORT_FILENAME_PREFIX = "radcopilot_history"

_HISTORY_LOCK = threading.RLock()


class ConfigLike(Protocol):
    """Minimal duck-typed config contract used by this module."""

    data_dir: Path

    class _UI(Protocol):
        save_history: bool
        max_history_items: int

    ui: _UI


@dataclass(slots=True)
class HistoryEntry:
    id: str
    created_at: str
    updated_at: str
    title: str
    mode: str = "report"
    template_id: str = ""
    modality: str = ""
    model: str = ""
    findings: str = ""
    impression: str = ""
    report: str = ""
    differential: str = ""
    guidelines: str = ""
    benchmark: str = ""
    transcript: str = ""
    notes: str = ""
    source: str = "local"
    starred: bool = False
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["summary"] = build_history_summary(payload)
        payload["search_text"] = build_search_text(payload)
        return payload


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def list_history(
    *,
    config: ConfigLike | Any | None = None,
    history_file: str | Path | None = None,
    limit: int | None = None,
    offset: int = 0,
    query: str = "",
    starred_only: bool = False,
    tags: Sequence[str] | None = None,
) -> dict[str, Any]:
    """
    Return history items as a JSON-serializable payload.

    The newest item is first.
    """
    items = _read_history_dicts(config=config, history_file=history_file)
    filtered = _filter_history_items(
        items,
        query=query,
        starred_only=starred_only,
        tags=tags,
    )

    total = len(filtered)
    safe_offset = max(0, int(offset or 0))
    safe_limit = _coerce_limit(limit)
    page = filtered[safe_offset : safe_offset + safe_limit]

    return {
        "ok": True,
        "count": len(page),
        "total": total,
        "offset": safe_offset,
        "limit": safe_limit,
        "items": page,
        "history_file": str(resolve_history_file(config=config, history_file=history_file)),
        "history_enabled": history_enabled(config),
    }


def get_history_entry(
    entry_id: str,
    *,
    config: ConfigLike | Any | None = None,
    history_file: str | Path | None = None,
) -> dict[str, Any]:
    """Fetch a single history item by ID."""
    normalized_id = str(entry_id or "").strip()
    if not normalized_id:
        return {"ok": False, "error": "Missing history entry id."}

    items = _read_history_dicts(config=config, history_file=history_file)
    for item in items:
        if item.get("id") == normalized_id:
            return {
                "ok": True,
                "item": item,
                "history_file": str(resolve_history_file(config=config, history_file=history_file)),
            }

    return {"ok": False, "error": f"History entry not found: {normalized_id}"}


def append_history_entry(
    entry: Mapping[str, Any] | None = None,
    *,
    config: ConfigLike | Any | None = None,
    history_file: str | Path | None = None,
    trim_to_limit: bool = True,
    **kwargs: Any,
) -> dict[str, Any]:
    """
    Append a new history entry.

    Entry may be provided as a mapping plus/or keyword arguments.
    """
    if not history_enabled(config):
        return {
            "ok": True,
            "saved": False,
            "reason": "History saving is disabled.",
            "item": None,
            "count": len(_read_history_dicts(config=config, history_file=history_file)),
        }

    raw = _merge_entry_input(entry, kwargs)
    item = normalize_history_entry(raw)

    items = _read_history_dicts(config=config, history_file=history_file)
    items.insert(0, item)

    if trim_to_limit:
        items = _apply_history_limit(items, config=config)

    _write_history_dicts(items, config=config, history_file=history_file)

    return {
        "ok": True,
        "saved": True,
        "item": item,
        "count": len(items),
        "history_file": str(resolve_history_file(config=config, history_file=history_file)),
    }


def upsert_history_entry(
    entry: Mapping[str, Any] | None = None,
    *,
    config: ConfigLike | Any | None = None,
    history_file: str | Path | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """
    Insert or replace a history entry by ID.

    If the incoming payload has no ID, a new one is generated and inserted.
    """
    if not history_enabled(config):
        return {
            "ok": True,
            "saved": False,
            "reason": "History saving is disabled.",
            "item": None,
            "count": len(_read_history_dicts(config=config, history_file=history_file)),
        }

    raw = _merge_entry_input(entry, kwargs)
    item = normalize_history_entry(raw)

    items = _read_history_dicts(config=config, history_file=history_file)
    replaced = False
    new_items: list[dict[str, Any]] = []

    for existing in items:
        if existing.get("id") == item["id"]:
            replaced = True
            merged = dict(existing)
            merged.update(item)
            merged["updated_at"] = _utc_now()
            merged = normalize_history_entry(merged)
            new_items.append(merged)
        else:
            new_items.append(existing)

    if not replaced:
        new_items.insert(0, item)

    new_items = _deduplicate_items(new_items)
    new_items = _apply_history_limit(new_items, config=config)
    _write_history_dicts(new_items, config=config, history_file=history_file)

    return {
        "ok": True,
        "saved": True,
        "replaced": replaced,
        "item": item,
        "count": len(new_items),
        "history_file": str(resolve_history_file(config=config, history_file=history_file)),
    }


def delete_history_entry(
    entry_id: str,
    *,
    config: ConfigLike | Any | None = None,
    history_file: str | Path | None = None,
) -> dict[str, Any]:
    """Delete a history entry by ID."""
    normalized_id = str(entry_id or "").strip()
    if not normalized_id:
        return {"ok": False, "error": "Missing history entry id."}

    items = _read_history_dicts(config=config, history_file=history_file)
    kept = [item for item in items if item.get("id") != normalized_id]

    if len(kept) == len(items):
        return {"ok": False, "error": f"History entry not found: {normalized_id}"}

    _write_history_dicts(kept, config=config, history_file=history_file)
    return {
        "ok": True,
        "deleted": True,
        "id": normalized_id,
        "count": len(kept),
        "history_file": str(resolve_history_file(config=config, history_file=history_file)),
    }


def clear_history(
    *,
    config: ConfigLike | Any | None = None,
    history_file: str | Path | None = None,
) -> dict[str, Any]:
    """Clear all history."""
    path = resolve_history_file(config=config, history_file=history_file)
    _write_history_dicts([], config=config, history_file=path)
    return {"ok": True, "cleared": True, "count": 0, "history_file": str(path)}


def toggle_star(
    entry_id: str,
    starred: bool | None = None,
    *,
    config: ConfigLike | Any | None = None,
    history_file: str | Path | None = None,
) -> dict[str, Any]:
    """Toggle or explicitly set the starred state of one history item."""
    normalized_id = str(entry_id or "").strip()
    if not normalized_id:
        return {"ok": False, "error": "Missing history entry id."}

    items = _read_history_dicts(config=config, history_file=history_file)
    updated_item: dict[str, Any] | None = None

    for item in items:
        if item.get("id") != normalized_id:
            continue
        next_value = (not bool(item.get("starred"))) if starred is None else bool(starred)
        item["starred"] = next_value
        item["updated_at"] = _utc_now()
        item.update(normalize_history_entry(item))
        updated_item = item
        break

    if updated_item is None:
        return {"ok": False, "error": f"History entry not found: {normalized_id}"}

    _write_history_dicts(items, config=config, history_file=history_file)
    return {
        "ok": True,
        "item": updated_item,
        "history_file": str(resolve_history_file(config=config, history_file=history_file)),
    }


def export_history(
    *,
    config: ConfigLike | Any | None = None,
    history_file: str | Path | None = None,
    export_path: str | Path | None = None,
    pretty: bool = True,
) -> dict[str, Any]:
    """
    Export current history to a standalone JSON file.

    If export_path is omitted, a timestamped export is written beside the main
    history file.
    """
    items = _read_history_dicts(config=config, history_file=history_file)
    history_path = resolve_history_file(config=config, history_file=history_file)

    if export_path is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        export_path = history_path.with_name(f"{DEFAULT_EXPORT_FILENAME_PREFIX}_{stamp}.json")
    else:
        export_path = Path(export_path).expanduser().resolve()

    export_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "exported_at": _utc_now(),
        "count": len(items),
        "items": items,
    }
    export_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2 if pretty else None),
        encoding="utf-8",
    )

    return {"ok": True, "count": len(items), "export_path": str(export_path)}


def import_history(
    source: str | Path,
    *,
    config: ConfigLike | Any | None = None,
    history_file: str | Path | None = None,
    merge: bool = True,
    replace: bool = False,
) -> dict[str, Any]:
    """
    Import history from a JSON file.

    Accepts either:
    - {"items": [...]} payload
    - raw list[dict]
    """
    source_path = Path(source).expanduser().resolve()
    if not source_path.exists():
        return {"ok": False, "error": f"Import file not found: {source_path}"}

    try:
        raw = json.loads(source_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"ok": False, "error": f"Could not parse import file: {exc}"}

    if isinstance(raw, Mapping):
        imported_items = raw.get("items", [])
    elif isinstance(raw, list):
        imported_items = raw
    else:
        return {"ok": False, "error": "Unsupported import payload shape."}

    if not isinstance(imported_items, list):
        return {"ok": False, "error": "Import payload did not contain a valid items list."}

    normalized_imports = [normalize_history_entry(item) for item in imported_items if isinstance(item, Mapping)]

    if replace:
        final_items = normalized_imports
    elif merge:
        current = _read_history_dicts(config=config, history_file=history_file)
        final_items = _deduplicate_items(normalized_imports + current)
    else:
        final_items = normalized_imports

    final_items = _apply_history_limit(final_items, config=config)
    _write_history_dicts(final_items, config=config, history_file=history_file)

    return {
        "ok": True,
        "imported": len(normalized_imports),
        "count": len(final_items),
        "history_file": str(resolve_history_file(config=config, history_file=history_file)),
    }


def search_history(
    query: str,
    *,
    config: ConfigLike | Any | None = None,
    history_file: str | Path | None = None,
    limit: int | None = None,
) -> dict[str, Any]:
    """Convenience wrapper for list_history(query=...)."""
    return list_history(
        config=config,
        history_file=history_file,
        query=query,
        limit=limit,
    )


def read_history_items(
    *,
    config: ConfigLike | Any | None = None,
    history_file: str | Path | None = None,
) -> list[HistoryEntry]:
    """Low-level typed reader."""
    return [HistoryEntry(**_strip_search_fields(item)) for item in _read_history_dicts(config=config, history_file=history_file)]


def write_history_items(
    items: Iterable[HistoryEntry | Mapping[str, Any]],
    *,
    config: ConfigLike | Any | None = None,
    history_file: str | Path | None = None,
) -> dict[str, Any]:
    """Low-level typed writer."""
    normalized = []
    for item in items:
        if isinstance(item, HistoryEntry):
            normalized.append(normalize_history_entry(asdict(item)))
        elif isinstance(item, Mapping):
            normalized.append(normalize_history_entry(item))
    normalized = _deduplicate_items(normalized)
    normalized = _apply_history_limit(normalized, config=config)
    _write_history_dicts(normalized, config=config, history_file=history_file)
    return {
        "ok": True,
        "count": len(normalized),
        "history_file": str(resolve_history_file(config=config, history_file=history_file)),
    }


def normalize_history_entry(entry: Mapping[str, Any] | MutableMapping[str, Any]) -> dict[str, Any]:
    """
    Normalize arbitrary UI/server payloads into the canonical history shape.
    """
    raw = dict(entry or {})
    now = _utc_now()

    entry_id = _clean_str(raw.get("id") or raw.get("entry_id") or uuid.uuid4().hex, 64) or uuid.uuid4().hex
    created_at = _normalize_timestamp(raw.get("created_at")) or now
    updated_at = _normalize_timestamp(raw.get("updated_at")) or now

    findings = _pull_text(raw, ("findings", "draft_findings", "input_findings"), MAX_REPORT_CHARS)
    impression = _pull_text(raw, ("impression", "draft_impression"), MAX_TEXT_CHARS)
    report = _pull_text(raw, ("report", "report_text", "draft_report", "generated_report"), MAX_REPORT_CHARS)
    differential = _pull_text(raw, ("differential", "differential_diagnosis"), MAX_TEXT_CHARS)
    guidelines = _pull_text(raw, ("guidelines", "guideline_text"), MAX_TEXT_CHARS)
    benchmark = _pull_text(raw, ("benchmark", "benchmark_notes", "benchmark_result"), MAX_TEXT_CHARS)
    transcript = _pull_text(raw, ("transcript", "dictation", "dictation_text"), MAX_REPORT_CHARS)
    notes = _pull_text(raw, ("notes", "user_notes"), MAX_TEXT_CHARS)

    title = _clean_str(raw.get("title"), MAX_TITLE_CHARS)
    if not title:
        title = _derive_title(
            impression=impression,
            report=report,
            findings=findings,
            modality=_clean_str(raw.get("modality"), MAX_MODALITY_CHARS),
            template_id=_clean_str(raw.get("template_id"), MAX_TEMPLATE_ID_CHARS),
        )

    normalized = HistoryEntry(
        id=entry_id,
        created_at=created_at,
        updated_at=updated_at,
        title=title,
        mode=_clean_str(raw.get("mode") or raw.get("view_mode") or "report", MAX_MODE_CHARS) or "report",
        template_id=_clean_str(raw.get("template_id") or raw.get("template"), MAX_TEMPLATE_ID_CHARS),
        modality=_clean_str(raw.get("modality") or raw.get("study_modality"), MAX_MODALITY_CHARS),
        model=_clean_str(raw.get("model") or raw.get("model_name"), MAX_MODEL_NAME_CHARS),
        findings=findings,
        impression=impression,
        report=report,
        differential=differential,
        guidelines=guidelines,
        benchmark=benchmark,
        transcript=transcript,
        notes=notes,
        source=_clean_str(raw.get("source") or "local", 40) or "local",
        starred=bool(raw.get("starred", False)),
        tags=_normalize_tags(raw.get("tags")),
        metadata=_normalize_metadata(raw.get("metadata")),
    ).to_dict()

    return normalized


def build_history_summary(entry: Mapping[str, Any]) -> str:
    """Build a short human-readable summary from a canonical history entry."""
    title = _clean_str(entry.get("title"), MAX_TITLE_CHARS)
    modality = _clean_str(entry.get("modality"), MAX_MODALITY_CHARS)
    impression = _first_nonempty(
        _clean_str(entry.get("impression"), 220),
        _clean_str(entry.get("report"), 220),
        _clean_str(entry.get("findings"), 220),
        "",
    )

    parts = [part for part in (title, modality) if part]
    prefix = " • ".join(parts)
    if prefix and impression:
        return f"{prefix}: {impression}"
    if prefix:
        return prefix
    return impression


def build_search_text(entry: Mapping[str, Any]) -> str:
    """Build a flattened lowercase search corpus for query filtering."""
    chunks = [
        entry.get("id", ""),
        entry.get("title", ""),
        entry.get("mode", ""),
        entry.get("template_id", ""),
        entry.get("modality", ""),
        entry.get("model", ""),
        entry.get("findings", ""),
        entry.get("impression", ""),
        entry.get("report", ""),
        entry.get("differential", ""),
        entry.get("guidelines", ""),
        entry.get("benchmark", ""),
        entry.get("transcript", ""),
        entry.get("notes", ""),
        " ".join(_normalize_tags(entry.get("tags"))),
        json.dumps(_normalize_metadata(entry.get("metadata")), ensure_ascii=False, sort_keys=True),
    ]
    return " ".join(_clean_str(chunk, MAX_REPORT_CHARS) for chunk in chunks if chunk).lower().strip()


def history_enabled(config: ConfigLike | Any | None = None) -> bool:
    """Check whether history saving is enabled."""
    ui = getattr(config, "ui", None)
    if ui is None:
        return True
    value = getattr(ui, "save_history", True)
    return bool(value)


def get_max_history_items(config: ConfigLike | Any | None = None) -> int:
    """Get max retained history items from config, with a safe fallback."""
    ui = getattr(config, "ui", None)
    if ui is None:
        return 20
    try:
        value = int(getattr(ui, "max_history_items", 20))
    except Exception:
        value = 20
    return max(1, min(value, 10_000))


def resolve_history_file(
    *,
    config: ConfigLike | Any | None = None,
    history_file: str | Path | None = None,
) -> Path:
    """
    Resolve the on-disk history file path.

    Resolution order:
    1. explicit history_file
    2. config.paths.data_dir/history/report_history.json
    3. config.data_dir/history/report_history.json
    4. ./radcopilot_datasets/history/report_history.json
    """
    if history_file is not None:
        path = Path(history_file).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    paths = getattr(config, "paths", None)
    if paths is not None:
        data_dir = getattr(paths, "data_dir", None)
        if data_dir:
            path = Path(data_dir).expanduser().resolve() / DEFAULT_HISTORY_SUBDIR / DEFAULT_HISTORY_FILENAME
            path.parent.mkdir(parents=True, exist_ok=True)
            return path

    data_dir = getattr(config, "data_dir", None) if config is not None else None
    if data_dir:
        path = Path(data_dir).expanduser().resolve() / DEFAULT_HISTORY_SUBDIR / DEFAULT_HISTORY_FILENAME
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    fallback = Path.cwd().resolve() / "radcopilot_datasets" / DEFAULT_HISTORY_SUBDIR / DEFAULT_HISTORY_FILENAME
    fallback.parent.mkdir(parents=True, exist_ok=True)
    return fallback


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------

def _read_history_dicts(
    *,
    config: ConfigLike | Any | None = None,
    history_file: str | Path | None = None,
) -> list[dict[str, Any]]:
    path = resolve_history_file(config=config, history_file=history_file)

    with _HISTORY_LOCK:
        if not path.exists():
            return []

        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            _quarantine_corrupt_history_file(path)
            return []

        if raw is None:
            return []
        if isinstance(raw, dict):
            raw_items = raw.get("items", [])
        elif isinstance(raw, list):
            raw_items = raw
        else:
            _quarantine_corrupt_history_file(path)
            return []

        normalized: list[dict[str, Any]] = []
        for item in raw_items:
            if not isinstance(item, Mapping):
                continue
            normalized.append(normalize_history_entry(item))

        normalized = _deduplicate_items(normalized)
        normalized.sort(key=lambda item: item.get("updated_at", ""), reverse=True)
        return normalized


def _write_history_dicts(
    items: Sequence[Mapping[str, Any]],
    *,
    config: ConfigLike | Any | None = None,
    history_file: str | Path | None = None,
) -> None:
    path = resolve_history_file(config=config, history_file=history_file)
    path.parent.mkdir(parents=True, exist_ok=True)

    normalized = [normalize_history_entry(item) for item in items]
    normalized = _deduplicate_items(normalized)
    normalized.sort(key=lambda item: item.get("updated_at", ""), reverse=True)

    payload = {
        "saved_at": _utc_now(),
        "count": len(normalized),
        "items": normalized,
    }

    with _HISTORY_LOCK:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            delete=False,
            dir=str(path.parent),
            prefix=f"{path.stem}_",
            suffix=".tmp",
        ) as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.flush()
            temp_name = handle.name
        Path(temp_name).replace(path)


def _merge_entry_input(
    entry: Mapping[str, Any] | None,
    kwargs: Mapping[str, Any],
) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    if entry:
        merged.update(dict(entry))
    if kwargs:
        merged.update(dict(kwargs))
    return merged


def _filter_history_items(
    items: Sequence[dict[str, Any]],
    *,
    query: str = "",
    starred_only: bool = False,
    tags: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    safe_query = _clean_str(query, MAX_SEARCH_QUERY_CHARS).lower()
    normalized_tags = {tag.lower() for tag in _normalize_tags(tags or [])}

    result: list[dict[str, Any]] = []
    for item in items:
        if starred_only and not bool(item.get("starred")):
            continue

        if normalized_tags:
            item_tags = {tag.lower() for tag in _normalize_tags(item.get("tags"))}
            if not normalized_tags.issubset(item_tags):
                continue

        if safe_query:
            haystack = item.get("search_text") or build_search_text(item)
            if safe_query not in haystack:
                continue

        result.append(item)

    return result


def _apply_history_limit(
    items: Sequence[dict[str, Any]],
    *,
    config: ConfigLike | Any | None = None,
) -> list[dict[str, Any]]:
    limit = get_max_history_items(config)
    return list(items[:limit])


def _deduplicate_items(items: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Deduplicate by ID first, then by content signature for entries missing IDs.
    Keep the newest version.
    """
    by_key: dict[str, dict[str, Any]] = {}

    for item in sorted(items, key=lambda x: x.get("updated_at", ""), reverse=True):
        item = normalize_history_entry(item)
        key = item.get("id") or _history_signature(item)
        if key not in by_key:
            by_key[key] = item

    deduped = list(by_key.values())
    deduped.sort(key=lambda item: item.get("updated_at", ""), reverse=True)
    return deduped


def _history_signature(item: Mapping[str, Any]) -> str:
    parts = [
        _clean_str(item.get("title"), MAX_TITLE_CHARS).lower(),
        _clean_str(item.get("modality"), MAX_MODALITY_CHARS).lower(),
        _clean_str(item.get("report"), 500).lower(),
        _clean_str(item.get("impression"), 500).lower(),
        _clean_str(item.get("findings"), 500).lower(),
    ]
    return "|".join(parts)


def _derive_title(
    *,
    impression: str,
    report: str,
    findings: str,
    modality: str,
    template_id: str,
) -> str:
    base = _first_nonempty(impression, report, findings, "").strip()
    if base:
        sentence = base.replace("\n", " ").strip()
        sentence = sentence.split(".")[0].strip()
        if sentence:
            return _clean_str(sentence, MAX_TITLE_CHARS)

    for fallback in (modality, template_id, "Untitled report"):
        value = _clean_str(fallback, MAX_TITLE_CHARS)
        if value:
            return value
    return "Untitled report"


def _normalize_tags(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        raw_tags = [part.strip() for part in value.split(",")]
    elif isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray, str)):
        raw_tags = [str(part).strip() for part in value]
    else:
        raw_tags = [str(value).strip()]

    cleaned: list[str] = []
    seen: set[str] = set()
    for tag in raw_tags:
        tag = _clean_str(tag, MAX_TAG_CHARS)
        if not tag:
            continue
        lowered = tag.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        cleaned.append(tag)
        if len(cleaned) >= MAX_TAGS:
            break
    return cleaned


def _normalize_metadata(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}

    normalized: dict[str, Any] = {}
    count = 0
    for key, item in value.items():
        if count >= MAX_METADATA_ITEMS:
            break

        clean_key = _clean_str(key, 80)
        if not clean_key:
            continue

        if item is None:
            continue
        if isinstance(item, (str, int, float, bool)):
            normalized[clean_key] = item if not isinstance(item, str) else _clean_str(item, MAX_TEXT_CHARS)
        elif isinstance(item, Sequence) and not isinstance(item, (bytes, bytearray, str)):
            normalized[clean_key] = [
                _clean_str(x, MAX_TEXT_CHARS) if isinstance(x, str) else x
                for x in list(item)[:50]
            ]
        elif isinstance(item, Mapping):
            try:
                normalized[clean_key] = json.loads(json.dumps(item, ensure_ascii=False))
            except Exception:
                normalized[clean_key] = str(item)
        else:
            normalized[clean_key] = str(item)

        count += 1

    return normalized


def _pull_text(data: Mapping[str, Any], keys: Sequence[str], max_chars: int) -> str:
    for key in keys:
        if key not in data:
            continue
        value = _clean_str(data.get(key), max_chars)
        if value:
            return value
    return ""


def _clean_str(value: Any, max_chars: int) -> str:
    text = str(value or "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    text = text.strip()
    if len(text) > max_chars:
        text = text[:max_chars].rstrip()
    return text


def _first_nonempty(*values: str) -> str:
    for value in values:
        if value:
            return value
    return ""


def _normalize_timestamp(value: Any) -> str:
    if not value:
        return ""
    text = str(value).strip()
    if not text:
        return ""
    # Accept already-usable ISO-like strings as-is.
    return text


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _coerce_limit(limit: int | None) -> int:
    if limit is None:
        return MAX_LIST_LIMIT
    try:
        value = int(limit)
    except Exception:
        value = MAX_LIST_LIMIT
    return max(1, min(value, MAX_LIST_LIMIT))


def _strip_search_fields(item: Mapping[str, Any]) -> dict[str, Any]:
    clean = dict(item)
    clean.pop("summary", None)
    clean.pop("search_text", None)
    return clean


def _quarantine_corrupt_history_file(path: Path) -> None:
    try:
        if not path.exists():
            return
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup = path.with_name(f"{path.stem}.corrupt_{stamp}{path.suffix}")
        shutil.move(str(path), str(backup))
    except Exception:
        # Best effort only. Avoid raising from recovery.
        return


# ---------------------------------------------------------------------------
# Compatibility aliases
# ---------------------------------------------------------------------------

save_history_entry = append_history_entry
record_report_history = append_history_entry
save_to_history = append_history_entry
get_recent_history = list_history
remove_history_entry = delete_history_entry
load_history = list_history
load_history_list = list_history


__all__ = [
    "DEFAULT_EXPORT_FILENAME_PREFIX",
    "DEFAULT_HISTORY_FILENAME",
    "DEFAULT_HISTORY_SUBDIR",
    "HistoryEntry",
    "append_history_entry",
    "build_history_summary",
    "build_search_text",
    "clear_history",
    "delete_history_entry",
    "export_history",
    "get_history_entry",
    "get_max_history_items",
    "get_recent_history",
    "history_enabled",
    "import_history",
    "list_history",
    "load_history",
    "load_history_list",
    "normalize_history_entry",
    "read_history_items",
    "record_report_history",
    "remove_history_entry",
    "resolve_history_file",
    "save_history_entry",
    "save_to_history",
    "search_history",
    "toggle_star",
    "upsert_history_entry",
    "write_history_items",
]
