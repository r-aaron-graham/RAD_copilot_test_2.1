from __future__ import annotations

"""
radcopilot.rag.library

Persistent retrieval library for RadCopilot Local.

This module replaces the temporary in-server fallback search in
`radcopilot.server.app` with a dedicated retrieval layer that can:
- load and save the persistent findings -> impression library
- normalize and deduplicate records
- add one or many records safely
- query similar records by findings text and optional modality
- provide status summaries for UI and API routes
- rebuild an in-memory index cache when the library changes

The implementation is standard-library first. If scikit-learn is available, it
will use TF-IDF + cosine similarity for stronger retrieval. If not, it falls
back to token overlap scoring.
"""

import json
from dataclasses import dataclass
from pathlib import Path
import threading
from typing import Any, Iterable, Protocol


DEFAULT_LIBRARY_FILE = "rag_library.json"
DEFAULT_MIN_SCORE = 0.05
DEFAULT_MAX_FINDINGS_CHARS = 5000
DEFAULT_MAX_IMPRESSION_CHARS = 3000


class ConfigLike(Protocol):
    """Minimal config contract expected from main.py/server.app."""

    base_dir: Path


@dataclass(slots=True)
class LibraryRecord:
    """Canonical persistent record used by the retrieval library."""

    findings: str
    impression: str
    modality: str = "unknown"
    source: str = "manual"
    created_at: str = ""
    source_file: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "findings": self.findings,
            "impression": self.impression,
            "modality": self.modality,
            "source": self.source,
            "created_at": self.created_at,
            "source_file": self.source_file,
        }


# -----------------------------
# Internal cache
# -----------------------------

_CACHE_LOCK = threading.RLock()
_CACHE: dict[str, dict[str, Any]] = {}


# -----------------------------
# Public path / load / save API
# -----------------------------

def get_library_path(config: ConfigLike | None = None, path: str | Path | None = None) -> Path:
    """Resolve the persistent JSON library path."""
    if path is not None:
        return Path(path).expanduser().resolve()
    if config is None:
        return Path(DEFAULT_LIBRARY_FILE).resolve()
    return (Path(config.base_dir) / DEFAULT_LIBRARY_FILE).resolve()



def load_records(
    *,
    config: ConfigLike | None = None,
    path: str | Path | None = None,
    force_reload: bool = False,
) -> list[dict[str, Any]]:
    """Load the library from disk and return normalized records."""
    library_path = get_library_path(config, path)

    with _CACHE_LOCK:
        cached = _CACHE.get(str(library_path))
        current_mtime = _safe_mtime(library_path)
        if cached and not force_reload and cached.get("mtime") == current_mtime:
            return [dict(item) for item in cached.get("records", [])]

    if not library_path.exists() or not library_path.is_file():
        records: list[dict[str, Any]] = []
    else:
        try:
            raw = json.loads(library_path.read_text(encoding="utf-8"))
        except Exception:
            raw = []
        records = []
        if isinstance(raw, list):
            for item in raw:
                normalized = normalize_record(item)
                if normalized is not None:
                    records.append(normalized)

    index_bundle = _build_index(records)
    with _CACHE_LOCK:
        _CACHE[str(library_path)] = {
            "mtime": _safe_mtime(library_path),
            "records": [dict(item) for item in records],
            **index_bundle,
        }
    return [dict(item) for item in records]



def save_records(
    records: Iterable[dict[str, Any] | LibraryRecord],
    *,
    config: ConfigLike | None = None,
    path: str | Path | None = None,
) -> Path:
    """Persist normalized records to disk and refresh the cache."""
    library_path = get_library_path(config, path)
    library_path.parent.mkdir(parents=True, exist_ok=True)

    normalized_records: list[dict[str, Any]] = []
    for item in records:
        normalized = normalize_record(item)
        if normalized is not None:
            normalized_records.append(normalized)

    library_path.write_text(
        json.dumps(normalized_records, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    with _CACHE_LOCK:
        _CACHE[str(library_path)] = {
            "mtime": _safe_mtime(library_path),
            "records": [dict(item) for item in normalized_records],
            **_build_index(normalized_records),
        }
    return library_path



def rebuild_index(*, config: ConfigLike | None = None, path: str | Path | None = None) -> dict[str, Any]:
    """Force a reload/rebuild of the in-memory retrieval index."""
    records = load_records(config=config, path=path, force_reload=True)
    return {
        "ok": True,
        "count": len(records),
        "library_file": str(get_library_path(config, path)),
        "modalities": count_by_modality(records),
    }


# -----------------------------
# Public mutation API
# -----------------------------

def add_record(
    record: dict[str, Any] | LibraryRecord,
    *,
    config: ConfigLike | None = None,
    path: str | Path | None = None,
    deduplicate: bool = True,
) -> bool:
    """Add a single record to the persistent library."""
    normalized = normalize_record(record)
    if normalized is None:
        return False

    records = load_records(config=config, path=path)
    if deduplicate:
        sig = record_signature(normalized)
        for existing in records:
            if record_signature(existing) == sig:
                return False

    records.append(normalized)
    save_records(records, config=config, path=path)
    return True



def add_records(
    records: Iterable[dict[str, Any] | LibraryRecord],
    *,
    config: ConfigLike | None = None,
    path: str | Path | None = None,
    deduplicate: bool = True,
) -> dict[str, int]:
    """Add many records and return summary counts."""
    current = load_records(config=config, path=path)
    existing_signatures = {record_signature(item) for item in current} if deduplicate else set()

    seen = 0
    added = 0
    skipped = 0

    for item in records:
        seen += 1
        normalized = normalize_record(item)
        if normalized is None:
            skipped += 1
            continue
        sig = record_signature(normalized)
        if deduplicate and sig in existing_signatures:
            skipped += 1
            continue
        current.append(normalized)
        existing_signatures.add(sig)
        added += 1

    save_records(current, config=config, path=path)
    return {"seen": seen, "added": added, "skipped": skipped}


# -----------------------------
# Public query / status API
# -----------------------------

def query_records(
    *,
    config: ConfigLike | None = None,
    findings: str,
    modality: str | None = None,
    k: int = 3,
    min_score: float = DEFAULT_MIN_SCORE,
    path: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Return the top-k similar retrieval records for the given findings."""
    findings = str(findings or "").strip()
    if not findings:
        return []

    library_path = get_library_path(config, path)
    records = load_records(config=config, path=library_path)
    if not records:
        return []

    with _CACHE_LOCK:
        bundle = _CACHE.get(str(library_path))
        if not bundle:
            bundle = {
                "records": records,
                **_build_index(records),
            }
            _CACHE[str(library_path)] = bundle

    filtered = _filter_by_modality(bundle.get("records", []), modality)
    if not filtered:
        return []

    # Prefer TF-IDF if sklearn is available and index is ready.
    tfidf_model = bundle.get("tfidf_model")
    tfidf_matrix = bundle.get("tfidf_matrix")
    tfidf_records = bundle.get("tfidf_records")
    if tfidf_model is not None and tfidf_matrix is not None and tfidf_records:
        try:
            from sklearn.metrics.pairwise import cosine_similarity  # type: ignore

            if modality:
                # When modality filtering is requested, rebuild a smaller view from filtered records.
                filtered_index = _build_tfidf_index(filtered)
                if filtered_index.get("tfidf_model") is not None:
                    tfidf_model = filtered_index["tfidf_model"]
                    tfidf_matrix = filtered_index["tfidf_matrix"]
                    tfidf_records = filtered_index["tfidf_records"]
                else:
                    tfidf_records = filtered
            vector = tfidf_model.transform([findings])
            scores = cosine_similarity(vector, tfidf_matrix).ravel().tolist()
            scored_items = []
            for record, score in zip(tfidf_records, scores, strict=False):
                if float(score) < min_score:
                    continue
                scored_items.append((float(score), record))
            scored_items.sort(key=lambda item: item[0], reverse=True)
            return [_scored_output(record, score) for score, record in scored_items[: max(1, k)]]
        except Exception:
            # Fall through to standard-library scoring.
            pass

    findings_tokens = _tokenize(findings)
    scored: list[tuple[float, dict[str, Any]]] = []
    for record in filtered:
        score = similarity_score(findings_tokens, _tokenize(record.get("findings", "")))
        if score < min_score:
            continue
        scored.append((score, record))

    scored.sort(key=lambda item: item[0], reverse=True)
    return [_scored_output(record, score) for score, record in scored[: max(1, k)]]



def get_status(*, config: ConfigLike | None = None, path: str | Path | None = None) -> dict[str, Any]:
    """Return a summary view of the persistent library."""
    library_path = get_library_path(config, path)
    records = load_records(config=config, path=library_path)
    return {
        "ok": True,
        "library_file": str(library_path),
        "count": len(records),
        "modalities": count_by_modality(records),
    }



def count_by_modality(records: Iterable[dict[str, Any]]) -> dict[str, int]:
    """Count records by modality."""
    counts: dict[str, int] = {}
    for item in records:
        modality = str(item.get("modality") or "unknown")
        counts[modality] = counts.get(modality, 0) + 1
    return counts


# -----------------------------
# Normalization / similarity
# -----------------------------

def normalize_record(record: dict[str, Any] | LibraryRecord | Any) -> dict[str, Any] | None:
    """Normalize user or parser input into the canonical record shape."""
    if isinstance(record, LibraryRecord):
        raw = record.to_dict()
    elif isinstance(record, dict):
        raw = dict(record)
    else:
        return None

    findings = _clean_text(raw.get("findings", ""), DEFAULT_MAX_FINDINGS_CHARS)
    impression = _clean_text(raw.get("impression", ""), DEFAULT_MAX_IMPRESSION_CHARS)
    if not findings or not impression:
        return None

    modality = _clean_text(raw.get("modality", "unknown"), 80) or "unknown"
    source = _clean_text(raw.get("source", "manual"), 120) or "manual"
    created_at = _clean_text(raw.get("created_at", ""), 80)
    source_file = _clean_text(raw.get("source_file", ""), 500)

    return {
        "findings": findings,
        "impression": impression,
        "modality": modality,
        "source": source,
        "created_at": created_at,
        "source_file": source_file,
    }



def record_signature(record: dict[str, Any]) -> str:
    """Stable deduplication signature for a retrieval record."""
    findings = _clean_text(record.get("findings", ""), 1000).lower()
    impression = _clean_text(record.get("impression", ""), 500).lower()
    modality = _clean_text(record.get("modality", "unknown"), 80).lower()
    return f"{modality}|{findings}|{impression}"



def similarity_score(a_tokens: set[str], b_tokens: set[str]) -> float:
    """Hybrid overlap score used when TF-IDF is unavailable."""
    if not a_tokens or not b_tokens:
        return 0.0
    inter = len(a_tokens & b_tokens)
    union = len(a_tokens | b_tokens)
    if union == 0:
        return 0.0
    jaccard = inter / union
    recall_like = inter / max(1, len(a_tokens))
    return (0.65 * jaccard) + (0.35 * recall_like)


# -----------------------------
# Internal indexing helpers
# -----------------------------

def _build_index(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Build the in-memory retrieval bundle."""
    bundle: dict[str, Any] = {
        "records": [dict(item) for item in records],
        "token_index": [_tokenize(item.get("findings", "")) for item in records],
    }
    bundle.update(_build_tfidf_index(records))
    return bundle



def _build_tfidf_index(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Best-effort TF-IDF index using scikit-learn if available."""
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore

        corpus = [str(item.get("findings", "")) for item in records]
        if not corpus:
            return {"tfidf_model": None, "tfidf_matrix": None, "tfidf_records": []}
        model = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=1)
        matrix = model.fit_transform(corpus)
        return {
            "tfidf_model": model,
            "tfidf_matrix": matrix,
            "tfidf_records": [dict(item) for item in records],
        }
    except Exception:
        return {"tfidf_model": None, "tfidf_matrix": None, "tfidf_records": []}



def _filter_by_modality(records: list[dict[str, Any]], modality: str | None) -> list[dict[str, Any]]:
    if not modality:
        return [dict(item) for item in records]
    wanted = str(modality).strip().lower()
    filtered = [
        dict(item)
        for item in records
        if str(item.get("modality") or "unknown").strip().lower() in {wanted, "unknown", ""}
    ]
    return filtered



def _scored_output(record: dict[str, Any], score: float) -> dict[str, Any]:
    return {
        "score": round(float(score), 4),
        "findings": record.get("findings", ""),
        "impression": record.get("impression", ""),
        "modality": record.get("modality", "unknown"),
        "source": record.get("source", "manual"),
        "created_at": record.get("created_at", ""),
        "source_file": record.get("source_file", ""),
    }



def _clean_text(value: Any, max_len: int) -> str:
    text = str(value or "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    text = text.strip()
    if len(text) > max_len:
        text = text[:max_len].rstrip()
    return text



def _tokenize(text: str) -> set[str]:
    cleaned = "".join(ch.lower() if ch.isalnum() else " " for ch in str(text))
    return {token for token in cleaned.split() if len(token) >= 2}



def _safe_mtime(path: Path) -> float | None:
    try:
        return path.stat().st_mtime
    except Exception:
        return None
