from __future__ import annotations

"""
radcopilot.rag.rating

Feedback and rating helpers for the persistent RadCopilot retrieval library.

Purpose:
- store user feedback about generated impression lines
- normalize rating values into positive / negative feedback
- append ratings to a JSONL file
- optionally promote positively rated findings -> impression pairs into the
  persistent RAG library

Intended usage:
    from radcopilot.rag.rating import save_rating

    result = save_rating(
        payload={
            "rating": "good",
            "line": "1. No acute cardiopulmonary abnormality.",
            "findings": "Heart size is normal. Lungs are clear.",
            "template": "ct-chest",
            "modality": "xr-chest",
            "context": {"source": "ui"},
        },
        config=config,
    )
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Protocol

from .library import add_record


DEFAULT_RATING_FILE = "rag_ratings.jsonl"
POSITIVE_RATINGS = {
    "good",
    "up",
    "thumbs_up",
    "thumbsup",
    "positive",
    "like",
    "approved",
    "accept",
    "accepted",
}
NEGATIVE_RATINGS = {
    "bad",
    "down",
    "thumbs_down",
    "thumbsdown",
    "negative",
    "dislike",
    "reject",
    "rejected",
}


class ConfigLike(Protocol):
    """
    Minimal loose config contract.

    This module intentionally accepts multiple config shapes:
    - main.AppConfig with base_dir / data_dir
    - config.AppConfig with paths.rag_rating_file / data_dir / base_dir
    """

    base_dir: Path
    data_dir: Path


def get_rating_path(
    *,
    config: ConfigLike | Any | None = None,
    path: str | Path | None = None,
) -> Path:
    """
    Resolve the JSONL path used to store ratings.
    """
    if path is not None:
        return Path(path).expanduser().resolve()

    if config is not None:
        paths = getattr(config, "paths", None)
        if paths is not None:
            rag_rating_file = getattr(paths, "rag_rating_file", None)
            if rag_rating_file:
                return Path(rag_rating_file).expanduser().resolve()

        data_dir = getattr(config, "data_dir", None)
        if data_dir:
            return (Path(data_dir) / DEFAULT_RATING_FILE).resolve()

        base_dir = getattr(config, "base_dir", None)
        if base_dir:
            return (Path(base_dir) / DEFAULT_RATING_FILE).resolve()

    return Path(DEFAULT_RATING_FILE).resolve()


def normalize_rating_value(value: Any) -> str | None:
    """
    Normalize a user-supplied rating into a canonical positive / negative token.

    Returns:
        "good" for positive feedback
        "bad" for negative feedback
        None for unsupported values
    """
    raw = str(value or "").strip().lower()
    if not raw:
        return None
    if raw in POSITIVE_RATINGS:
        return "good"
    if raw in NEGATIVE_RATINGS:
        return "bad"
    return None


def save_rating(
    *,
    payload: dict[str, Any],
    config: ConfigLike | Any | None = None,
    path: str | Path | None = None,
    add_positive_to_library: bool = True,
) -> dict[str, Any]:
    """
    Save a rating entry and optionally promote positive feedback into the RAG library.

    Expected payload fields:
    - rating: required
    - line: impression line being rated
    - findings: source findings text
    - template: optional template id/name
    - modality: optional modality
    - context: optional dict or other serializable object
    - source_file: optional string
    """
    normalized_rating = normalize_rating_value(payload.get("rating"))
    if normalized_rating is None:
        return {
            "ok": False,
            "error": "rating must be a supported positive/negative value",
            "allowed_positive": sorted(POSITIVE_RATINGS),
            "allowed_negative": sorted(NEGATIVE_RATINGS),
        }

    entry = {
        "ts": _utc_now(),
        "rating": normalized_rating,
        "line": _clean_text(payload.get("line", ""), 4000),
        "findings": _clean_text(payload.get("findings", ""), 12000),
        "template": _clean_text(payload.get("template", ""), 200),
        "modality": _clean_text(payload.get("modality", "unknown"), 80) or "unknown",
        "source_file": _clean_text(payload.get("source_file", ""), 1000),
        "context": _safe_jsonable(payload.get("context", {})),
    }

    rating_file = get_rating_path(config=config, path=path)
    _append_jsonl(rating_file, entry)

    added_to_library = False
    if (
        add_positive_to_library
        and normalized_rating == "good"
        and entry["findings"]
        and entry["line"]
    ):
        added_to_library = add_record(
            {
                "findings": entry["findings"],
                "impression": entry["line"],
                "modality": entry["modality"],
                "source": "user_rating",
                "created_at": entry["ts"],
                "source_file": entry["source_file"],
            },
            config=config,
            deduplicate=True,
        )

    return {
        "ok": True,
        "saved": True,
        "added_to_library": added_to_library,
        "rating_file": str(rating_file),
        "rating": normalized_rating,
    }


def rate_line(
    *,
    rating: str,
    line: str,
    findings: str,
    template: str = "",
    modality: str = "unknown",
    context: Any = None,
    source_file: str = "",
    config: ConfigLike | Any | None = None,
    path: str | Path | None = None,
    add_positive_to_library: bool = True,
) -> dict[str, Any]:
    """
    Convenience wrapper around save_rating().
    """
    return save_rating(
        payload={
            "rating": rating,
            "line": line,
            "findings": findings,
            "template": template,
            "modality": modality,
            "context": context or {},
            "source_file": source_file,
        },
        config=config,
        path=path,
        add_positive_to_library=add_positive_to_library,
    )


def load_ratings(
    *,
    config: ConfigLike | Any | None = None,
    path: str | Path | None = None,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """
    Load ratings from disk.

    Returns newest-last in file order unless a limit is applied, in which case
    the last N records are returned.
    """
    rating_file = get_rating_path(config=config, path=path)
    if not rating_file.exists() or not rating_file.is_file():
        return []

    lines = rating_file.read_text(encoding="utf-8", errors="replace").splitlines()
    if limit is not None and limit > 0:
        lines = lines[-limit:]

    items: list[dict[str, Any]] = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            item = json.loads(line)
        except Exception:
            continue
        if isinstance(item, dict):
            items.append(item)
    return items


def count_ratings(
    *,
    config: ConfigLike | Any | None = None,
    path: str | Path | None = None,
) -> int:
    """
    Count rating records in the JSONL file.
    """
    rating_file = get_rating_path(config=config, path=path)
    if not rating_file.exists() or not rating_file.is_file():
        return 0

    with rating_file.open("r", encoding="utf-8", errors="replace") as fh:
        return sum(1 for line in fh if line.strip())


def get_rating_summary(
    *,
    config: ConfigLike | Any | None = None,
    path: str | Path | None = None,
) -> dict[str, Any]:
    """
    Return a compact summary of the rating store.
    """
    items = load_ratings(config=config, path=path)
    positive = 0
    negative = 0

    for item in items:
        rating = normalize_rating_value(item.get("rating"))
        if rating == "good":
            positive += 1
        elif rating == "bad":
            negative += 1

    rating_file = get_rating_path(config=config, path=path)
    return {
        "ok": True,
        "rating_file": str(rating_file),
        "count": len(items),
        "positive": positive,
        "negative": negative,
    }


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _safe_jsonable(value: Any) -> Any:
    """
    Best-effort conversion to a JSON-safe object.
    """
    if value is None:
        return {}
    if isinstance(value, (str, int, float, bool, list, dict)):
        try:
            json.dumps(value)
            return value
        except Exception:
            return str(value)
    return str(value)


def _clean_text(value: Any, max_len: int) -> str:
    text = str(value or "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = "\n".join(line.strip() for line in text.split("\n"))
    text = text.strip(" \n\t:")
    if len(text) > max_len:
        text = text[:max_len].rstrip()
    return text


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


__all__ = [
    "DEFAULT_RATING_FILE",
    "NEGATIVE_RATINGS",
    "POSITIVE_RATINGS",
    "count_ratings",
    "get_rating_path",
    "get_rating_summary",
    "load_ratings",
    "normalize_rating_value",
    "rate_line",
    "save_rating",
]
