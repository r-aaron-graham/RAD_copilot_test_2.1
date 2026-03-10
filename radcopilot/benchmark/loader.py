from __future__ import annotations

"""
radcopilot.benchmark.loader

Benchmark dataset discovery and loading for RadCopilot Local.

Purpose
-------
This module gives the benchmark routes a real parsing layer so the UI can load
benchmark cases from local files or directories instead of only previewing file
metadata.

Design goals
------------
- Standard-library first.
- Compatible with the current modular refactor and its duck-typed config model.
- Accept common radiology benchmark inputs: .txt, .csv, .xml, .json, .jsonl,
  .tar, .tgz, .tar.gz, or directories containing those files.
- Normalize each case into a stable structure with findings, reference
  impression, modality, and source metadata.
- Return JSON-serializable payloads suitable for `/benchmark/load` and
  `/benchmark/load-path` responses.

The module is intentionally self-contained so it can be dropped into the
current repo even before every server integration point is finalized.
"""

import csv
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import io
import json
from pathlib import Path
import re
import tarfile
import tempfile
from typing import Any, Iterable, Iterator, Mapping, Protocol, Sequence
import xml.etree.ElementTree as ET


TEXT_EXTENSIONS = {".txt", ".md", ".text"}
CSV_EXTENSIONS = {".csv"}
XML_EXTENSIONS = {".xml"}
JSON_EXTENSIONS = {".json", ".jsonl"}
ARCHIVE_EXTENSIONS = {".tar", ".tgz", ".gz"}
SUPPORTED_FILE_EXTENSIONS = (
    TEXT_EXTENSIONS | CSV_EXTENSIONS | XML_EXTENSIONS | JSON_EXTENSIONS | ARCHIVE_EXTENSIONS
)
DISCOVERY_EXTENSIONS = SUPPORTED_FILE_EXTENSIONS | {".gz"}

MAX_TEXT_READ = 2_000_000
MAX_CASES_PER_LOAD = 25_000
MAX_ITEMS_PER_DATASET_LIST = 2_000
MAX_FINDINGS_CHARS = 5_000
MAX_IMPRESSION_CHARS = 3_000
MAX_REPORT_CHARS = 12_000
MAX_PREVIEW_CHARS = 4_000

_FINDINGS_KEYS = (
    "findings",
    "report_findings",
    "report_findings_text",
    "finding",
    "observations",
    "result",
    "results",
    "body",
    "text_findings",
    "input_findings",
)

_IMPRESSION_KEYS = (
    "impression",
    "reference_impression",
    "gold_impression",
    "target_impression",
    "expected_impression",
    "report_impression",
    "conclusion",
    "conclusions",
    "opinion",
    "assessment",
    "label",
    "answer",
    "target",
    "expected_output",
    "ground_truth",
    "reference",
)

_REPORT_KEYS = (
    "report",
    "report_text",
    "full_report",
    "reference_report",
    "full_text",
    "text",
    "content",
    "note",
)

_MODALITY_KEYS = (
    "modality",
    "study_modality",
    "exam_modality",
    "exam_type",
    "series_description",
)

_CASE_ID_KEYS = (
    "case_id",
    "caseid",
    "study_id",
    "studyid",
    "id",
    "uid",
)


class ConfigLike(Protocol):
    """Minimal duck-typed config contract used by the benchmark loader."""

    base_dir: Path


@dataclass(slots=True)
class BenchmarkCase:
    findings: str
    impression: str
    modality: str = "unknown"
    source: str = "benchmark"
    created_at: str = ""
    source_file: str = ""
    case_id: str = ""
    reference_report: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        if not payload["created_at"]:
            payload["created_at"] = _utc_now()
        if not payload["reference_report"]:
            payload.pop("reference_report", None)
        if not payload["case_id"]:
            payload.pop("case_id", None)
        if not payload["metadata"]:
            payload.pop("metadata", None)
        return payload


@dataclass(slots=True)
class BenchmarkLoadStats:
    path: str
    kind: str
    files_seen: int = 0
    records_seen: int = 0
    records_valid: int = 0
    records_skipped: int = 0
    duplicates_skipped: int = 0
    parser_errors: int = 0
    truncated: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def list_benchmark_datasets(
    *,
    config: ConfigLike | Any | None = None,
    roots: Sequence[str | Path] | None = None,
    max_items: int = MAX_ITEMS_PER_DATASET_LIST,
) -> list[dict[str, Any]]:
    """
    Discover benchmark-friendly datasets from the configured runtime paths.

    Returned items are JSON-serializable and safe to expose to the UI.
    """
    discovered: list[dict[str, Any]] = []
    seen_paths: set[str] = set()

    for root in _resolve_dataset_roots(config=config, roots=roots):
        if not root.exists():
            continue

        if root.is_file():
            item = _dataset_descriptor(root, root)
            if item and item["path"] not in seen_paths:
                discovered.append(item)
                seen_paths.add(item["path"])
            continue

        candidate_dirs_seen: set[str] = set()
        for child in sorted(root.rglob("*")):
            if len(discovered) >= max_items:
                return discovered

            if child.is_dir():
                # Only surface a directory when it contains at least one supported file.
                if str(child) in candidate_dirs_seen:
                    continue
                if _directory_contains_supported_files(child):
                    item = _dataset_descriptor(child, root)
                    if item and item["path"] not in seen_paths:
                        discovered.append(item)
                        seen_paths.add(item["path"])
                    candidate_dirs_seen.add(str(child))
                continue

            item = _dataset_descriptor(child, root)
            if item and item["path"] not in seen_paths:
                discovered.append(item)
                seen_paths.add(item["path"])

    return discovered


def load_path(
    *,
    config: ConfigLike | Any | None = None,
    path: str | Path,
    limit: int = MAX_CASES_PER_LOAD,
    deduplicate: bool = True,
) -> dict[str, Any]:
    """
    Load a benchmark dataset from a local file or directory.

    The returned payload is JSON-serializable and suitable for API responses.
    """
    source_path = Path(path).expanduser().resolve()
    if not source_path.exists():
        return {"ok": False, "error": f"Path does not exist: {source_path}"}

    stats = BenchmarkLoadStats(
        path=str(source_path),
        kind="directory" if source_path.is_dir() else "file",
    )

    cases = _collect_cases_from_path(
        source_path,
        stats=stats,
        limit=max(1, int(limit)),
        deduplicate=deduplicate,
    )
    payload: dict[str, Any] = {
        "ok": True,
        "source": "module",
        "path": str(source_path),
        "kind": stats.kind,
        "count": len(cases),
        "record_count": len(cases),
        "items": cases,
        "modalities": _count_by_modality(cases),
        "stats": stats.to_dict(),
    }

    if not cases:
        preview = _preview_path(source_path)
        if preview:
            payload["preview"] = preview
            payload["warning"] = "No benchmark cases were parsed from the supplied path."

    benchmark_dir = _resolve_benchmark_dir(config)
    if benchmark_dir is not None:
        payload["benchmark_dir"] = str(benchmark_dir)

    return payload


def load_bytes(
    *,
    filename: str,
    data: bytes,
    config: ConfigLike | Any | None = None,
    limit: int = MAX_CASES_PER_LOAD,
    deduplicate: bool = True,
) -> dict[str, Any]:
    """
    Load benchmark cases from uploaded bytes.

    This is intended for future `/benchmark/load` multipart support.
    """
    safe_name = Path(filename or "upload.txt").name or "upload.txt"
    suffix = "".join(Path(safe_name).suffixes) or Path(safe_name).suffix or ".txt"

    with tempfile.TemporaryDirectory(prefix="radcopilot-benchmark-") as tmpdir:
        temp_path = Path(tmpdir) / f"input{suffix}"
        temp_path.write_bytes(data)
        result = load_path(
            config=config,
            path=temp_path,
            limit=limit,
            deduplicate=deduplicate,
        )
        if result.get("ok"):
            result["filename"] = safe_name
            result["source"] = "module-upload"
        return result


# Backwards-compatible aliases for likely server integration names.
load_benchmark_path = load_path
load_benchmark_bytes = load_bytes
list_available_datasets = list_benchmark_datasets


# ---------------------------------------------------------------------------
# Path collection and parsing
# ---------------------------------------------------------------------------

def _collect_cases_from_path(
    source_path: Path,
    *,
    stats: BenchmarkLoadStats,
    limit: int,
    deduplicate: bool,
) -> list[dict[str, Any]]:
    seen_signatures: set[str] = set()
    cases: list[dict[str, Any]] = []

    for record in _iter_path_records(source_path, stats=stats):
        stats.records_seen += 1
        case = _normalize_case(record)
        if case is None:
            stats.records_skipped += 1
            continue

        signature = _case_signature(case)
        if deduplicate and signature in seen_signatures:
            stats.duplicates_skipped += 1
            continue

        seen_signatures.add(signature)
        cases.append(case.to_dict())
        stats.records_valid += 1

        if len(cases) >= limit:
            stats.truncated = True
            break

    return cases


def _iter_path_records(source_path: Path, *, stats: BenchmarkLoadStats) -> Iterator[dict[str, Any]]:
    if source_path.is_dir():
        for child in sorted(source_path.rglob("*")):
            if not child.is_file():
                continue
            stats.files_seen += 1
            yield from _iter_file_records(child, stats=stats)
        return

    stats.files_seen += 1
    yield from _iter_file_records(source_path, stats=stats)


def _iter_file_records(path: Path, *, stats: BenchmarkLoadStats) -> Iterator[dict[str, Any]]:
    suffix = path.suffix.lower()
    lower_name = path.name.lower()

    try:
        if suffix in TEXT_EXTENSIONS:
            record = parse_txt_case(_read_text(path), source_file=str(path))
            if record:
                yield record
            return

        if suffix in CSV_EXTENSIONS:
            yield from _iter_csv_records(path)
            return

        if suffix in XML_EXTENSIONS:
            yield from _iter_xml_records(path)
            return

        if suffix in JSON_EXTENSIONS:
            yield from _iter_json_records(path)
            return

        if suffix in {".tar", ".tgz"} or lower_name.endswith(".tar.gz"):
            yield from _iter_tar_records(path, stats=stats)
            return
    except Exception:
        stats.parser_errors += 1
        return


def parse_txt_case(text: str, *, source_file: str = "") -> dict[str, Any] | None:
    cleaned = _clean_text(text, MAX_TEXT_READ)
    if not cleaned:
        return None

    findings = _extract_section(
        cleaned,
        labels=("FINDINGS", "OBSERVATIONS", "RESULT", "RESULTS", "BODY"),
        stop_labels=("IMPRESSION", "CONCLUSION", "CONCLUSIONS", "OPINION", "ASSESSMENT"),
    )
    impression = _extract_section(
        cleaned,
        labels=("IMPRESSION", "CONCLUSION", "CONCLUSIONS", "OPINION", "ASSESSMENT"),
        stop_labels=("RECOMMENDATION", "RECOMMENDATIONS", "SIGNED", "END"),
    )

    if not findings or not impression:
        findings2, impression2 = _split_report_text(cleaned)
        findings = findings or findings2
        impression = impression or impression2

    if not findings or not impression:
        return None

    return {
        "findings": findings,
        "impression": impression,
        "reference_report": _clean_text(cleaned, MAX_REPORT_CHARS),
        "modality": _infer_modality(findings + "\n" + impression, source_file=source_file),
        "source": "txt_file",
        "created_at": _utc_now(),
        "source_file": source_file,
    }


def _iter_csv_records(path: Path) -> Iterator[dict[str, Any]]:
    text = _read_text(path)
    if not text.strip():
        return

    reader = csv.DictReader(io.StringIO(text))
    for row in reader:
        record = _mapping_to_case(row, source_file=str(path), source="csv_file")
        if record:
            yield record


def _iter_json_records(path: Path) -> Iterator[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except Exception:
                continue
            if isinstance(item, Mapping):
                record = _mapping_to_case(item, source_file=str(path), source="jsonl_file")
                if record:
                    yield record
        return

    try:
        payload = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except Exception:
        return

    yield from _iter_json_payload(payload, source_file=str(path), source="json_file")


def _iter_json_payload(payload: Any, *, source_file: str, source: str) -> Iterator[dict[str, Any]]:
    if isinstance(payload, Mapping):
        direct = _mapping_to_case(payload, source_file=source_file, source=source)
        if direct:
            yield direct

        for key in ("items", "records", "cases", "data", "reports", "examples"):
            nested = payload.get(key)
            if isinstance(nested, list):
                for item in nested:
                    if not isinstance(item, Mapping):
                        continue
                    record = _mapping_to_case(item, source_file=source_file, source=source)
                    if record:
                        yield record
        return

    if isinstance(payload, list):
        for item in payload:
            if not isinstance(item, Mapping):
                continue
            record = _mapping_to_case(item, source_file=source_file, source=source)
            if record:
                yield record


def _iter_xml_records(path: Path) -> Iterator[dict[str, Any]]:
    try:
        root = ET.fromstring(path.read_text(encoding="utf-8", errors="replace"))
    except Exception:
        return

    candidates: list[ET.Element] = []
    for tag in ("record", "report", "case", "document", "doc", "entry", "article"):
        candidates.extend(root.findall(f".//{tag}"))

    yielded = 0
    for node in candidates:
        record = _xml_node_to_case(node, source_file=str(path))
        if record:
            yielded += 1
            yield record

    if yielded:
        return

    record = _xml_node_to_case(root, source_file=str(path))
    if record:
        yield record


def _iter_tar_records(path: Path, *, stats: BenchmarkLoadStats) -> Iterator[dict[str, Any]]:
    try:
        with tarfile.open(path, mode="r:*") as tar:
            for member in tar.getmembers():
                if not member.isfile():
                    continue

                member_name = member.name
                member_suffix = Path(member_name).suffix.lower()
                extracted = tar.extractfile(member)
                if extracted is None:
                    continue
                data = extracted.read()
                stats.files_seen += 1

                if member_suffix in TEXT_EXTENSIONS:
                    record = parse_txt_case(
                        data.decode("utf-8", errors="replace"),
                        source_file=f"{path}:{member_name}",
                    )
                    if record:
                        record["source"] = "tar_txt"
                        yield record
                elif member_suffix in CSV_EXTENSIONS:
                    reader = csv.DictReader(io.StringIO(data.decode("utf-8", errors="replace")))
                    for row in reader:
                        record = _mapping_to_case(
                            row,
                            source_file=f"{path}:{member_name}",
                            source="tar_csv",
                        )
                        if record:
                            yield record
                elif member_suffix in JSON_EXTENSIONS:
                    text = data.decode("utf-8", errors="replace")
                    if member_suffix == ".jsonl":
                        for line in text.splitlines():
                            line = line.strip()
                            if not line:
                                continue
                            try:
                                item = json.loads(line)
                            except Exception:
                                continue
                            if isinstance(item, Mapping):
                                record = _mapping_to_case(
                                    item,
                                    source_file=f"{path}:{member_name}",
                                    source="tar_jsonl",
                                )
                                if record:
                                    yield record
                    else:
                        try:
                            payload = json.loads(text)
                        except Exception:
                            payload = None
                        if payload is not None:
                            yield from _iter_json_payload(
                                payload,
                                source_file=f"{path}:{member_name}",
                                source="tar_json",
                            )
                elif member_suffix in XML_EXTENSIONS:
                    try:
                        root = ET.fromstring(data.decode("utf-8", errors="replace"))
                    except Exception:
                        continue
                    record = _xml_node_to_case(root, source_file=f"{path}:{member_name}")
                    if record:
                        record["source"] = "tar_xml"
                        yield record
    except Exception:
        stats.parser_errors += 1
        return


# ---------------------------------------------------------------------------
# Normalization helpers
# ---------------------------------------------------------------------------

def _mapping_to_case(mapping: Mapping[str, Any], *, source_file: str, source: str) -> dict[str, Any] | None:
    lowered = {str(k).strip().lower(): v for k, v in mapping.items()}

    findings = _first_value(lowered, _FINDINGS_KEYS)
    impression = _first_value(lowered, _IMPRESSION_KEYS)
    report_text = _first_value(lowered, _REPORT_KEYS)
    modality = _first_value(lowered, _MODALITY_KEYS)
    case_id = _first_value(lowered, _CASE_ID_KEYS)

    if (not findings or not impression) and report_text:
        findings2, impression2 = _split_report_text(str(report_text))
        findings = findings or findings2
        impression = impression or impression2

    if not findings or not impression:
        return None

    metadata = _metadata_from_mapping(lowered)

    return {
        "findings": _clean_text(findings, MAX_FINDINGS_CHARS),
        "impression": _clean_text(impression, MAX_IMPRESSION_CHARS),
        "reference_report": _clean_text(report_text, MAX_REPORT_CHARS),
        "modality": _infer_modality(
            modality or (str(findings) + "\n" + str(impression)),
            source_file=source_file,
        ),
        "source": source,
        "created_at": _utc_now(),
        "source_file": source_file,
        "case_id": _clean_text(case_id, 120),
        "metadata": metadata,
    }


def _xml_node_to_case(node: ET.Element, *, source_file: str) -> dict[str, Any] | None:
    text_map: dict[str, str] = {}

    for elem in node.iter():
        tag = _strip_xml_ns(elem.tag).strip().lower()
        value = _clean_text("".join(elem.itertext()), 20_000)
        if value:
            text_map.setdefault(tag, value)

    findings = _first_value(text_map, _FINDINGS_KEYS)
    impression = _first_value(text_map, _IMPRESSION_KEYS)
    report_text = _first_value(text_map, _REPORT_KEYS)
    modality = _first_value(text_map, _MODALITY_KEYS)
    case_id = _first_value(text_map, _CASE_ID_KEYS)

    if not findings or not impression:
        labeled_sections = _extract_labeled_xml_sections(node)
        findings = findings or labeled_sections.get("findings", "")
        impression = impression or labeled_sections.get("impression", "")

    if (not findings or not impression) and report_text:
        findings2, impression2 = _split_report_text(report_text)
        findings = findings or findings2
        impression = impression or impression2

    if not findings or not impression:
        return None

    return {
        "findings": _clean_text(findings, MAX_FINDINGS_CHARS),
        "impression": _clean_text(impression, MAX_IMPRESSION_CHARS),
        "reference_report": _clean_text(report_text, MAX_REPORT_CHARS),
        "modality": _infer_modality(
            modality or (str(findings) + "\n" + str(impression)),
            source_file=source_file,
        ),
        "source": "xml_file",
        "created_at": _utc_now(),
        "source_file": source_file,
        "case_id": _clean_text(case_id, 120),
    }


def _normalize_case(record: Mapping[str, Any]) -> BenchmarkCase | None:
    findings = _clean_text(record.get("findings", ""), MAX_FINDINGS_CHARS)
    impression = _clean_text(record.get("impression", ""), MAX_IMPRESSION_CHARS)
    if not findings or not impression:
        return None

    modality = _clean_text(record.get("modality", "unknown"), 80) or "unknown"
    source = _clean_text(record.get("source", "benchmark"), 120) or "benchmark"
    created_at = _clean_text(record.get("created_at", _utc_now()), 80) or _utc_now()
    source_file = _clean_text(record.get("source_file", ""), 2000)
    case_id = _clean_text(record.get("case_id", ""), 120)
    reference_report = _clean_text(record.get("reference_report", ""), MAX_REPORT_CHARS)

    raw_metadata = record.get("metadata", {})
    metadata = raw_metadata if isinstance(raw_metadata, dict) else {}

    return BenchmarkCase(
        findings=findings,
        impression=impression,
        modality=modality,
        source=source,
        created_at=created_at,
        source_file=source_file,
        case_id=case_id,
        reference_report=reference_report,
        metadata=metadata,
    )


def _metadata_from_mapping(mapping: Mapping[str, Any]) -> dict[str, Any]:
    ignore = (
        set(_FINDINGS_KEYS)
        | set(_IMPRESSION_KEYS)
        | set(_REPORT_KEYS)
        | set(_MODALITY_KEYS)
        | set(_CASE_ID_KEYS)
    )
    keep: dict[str, Any] = {}
    for key, value in mapping.items():
        if key in ignore:
            continue
        if value in (None, "", [], {}, ()):
            continue
        if isinstance(value, (str, int, float, bool)):
            keep[key] = value
        else:
            try:
                keep[key] = json.loads(json.dumps(value, ensure_ascii=False))
            except Exception:
                keep[key] = str(value)
    return keep


# ---------------------------------------------------------------------------
# Discovery and config helpers
# ---------------------------------------------------------------------------

def _resolve_dataset_roots(
    *,
    config: ConfigLike | Any | None,
    roots: Sequence[str | Path] | None,
) -> list[Path]:
    candidates: list[Path] = []

    if roots:
        for item in roots:
            candidates.append(Path(item).expanduser().resolve())

    benchmark_dir = _resolve_benchmark_dir(config)
    if benchmark_dir is not None:
        candidates.append(benchmark_dir)

    # Modular config.py path model.
    if config is not None:
        paths = getattr(config, "paths", None)
        if paths is not None:
            for attr in ("benchmark_dir", "data_dir"):
                value = getattr(paths, attr, None)
                if value:
                    candidates.append(Path(value).expanduser().resolve())
            for item in getattr(paths, "dataset_search_roots", []) or []:
                try:
                    candidates.append(Path(item).expanduser().resolve())
                except Exception:
                    continue

    # Compatibility with simpler launcher/server config.
    for attr in ("benchmark_dir", "data_dir", "base_dir"):
        value = getattr(config, attr, None) if config is not None else None
        if not value:
            continue
        resolved = Path(value).expanduser().resolve()
        if attr == "base_dir":
            candidates.extend(
                [
                    resolved / "radcopilot_datasets",
                    resolved / "benchmarks",
                    resolved / "data" / "datasets",
                ]
            )
        else:
            candidates.append(resolved)

    deduped: list[Path] = []
    seen: set[str] = set()
    for path in candidates:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(path)

    return deduped


def _resolve_benchmark_dir(config: ConfigLike | Any | None) -> Path | None:
    if config is None:
        return None

    paths = getattr(config, "paths", None)
    if paths is not None:
        value = getattr(paths, "benchmark_dir", None)
        if value:
            return Path(value).expanduser().resolve()

    value = getattr(config, "benchmark_dir", None)
    if value:
        return Path(value).expanduser().resolve()

    data_dir = getattr(config, "data_dir", None)
    if data_dir:
        data_dir_path = Path(data_dir).expanduser().resolve()
        candidate = data_dir_path / "benchmarks"
        return candidate if candidate.exists() else data_dir_path

    base_dir = getattr(config, "base_dir", None)
    if base_dir:
        candidate = Path(base_dir).expanduser().resolve() / "radcopilot_datasets" / "benchmarks"
        return candidate

    return None


def _dataset_descriptor(path: Path, root: Path) -> dict[str, Any] | None:
    if path.is_dir():
        if not _directory_contains_supported_files(path):
            return None
        kind = "directory"
        suffix = ""
        size_bytes = 0
    else:
        suffix = path.suffix.lower()
        lower_name = path.name.lower()
        if suffix not in DISCOVERY_EXTENSIONS and not lower_name.endswith(".tar.gz"):
            return None
        kind = "file"
        size_bytes = path.stat().st_size

    try:
        relative_path = str(path.relative_to(root))
    except ValueError:
        relative_path = path.name

    return {
        "name": path.name,
        "relative_path": relative_path,
        "path": str(path),
        "kind": kind,
        "suffix": suffix,
        "size_bytes": size_bytes,
        "root": str(root),
    }


def _directory_contains_supported_files(path: Path) -> bool:
    try:
        for child in path.rglob("*"):
            if not child.is_file():
                continue
            suffix = child.suffix.lower()
            if suffix in DISCOVERY_EXTENSIONS or child.name.lower().endswith(".tar.gz"):
                return True
    except Exception:
        return False
    return False


# ---------------------------------------------------------------------------
# Generic text helpers
# ---------------------------------------------------------------------------

def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")[:MAX_TEXT_READ]


def _preview_path(path: Path) -> str:
    if path.is_dir():
        return ""
    if path.suffix.lower() not in {".txt", ".md", ".json", ".jsonl", ".csv", ".xml"}:
        return ""
    try:
        return path.read_text(encoding="utf-8", errors="replace")[:MAX_PREVIEW_CHARS]
    except Exception:
        return ""


def _extract_section(text: str, *, labels: Iterable[str], stop_labels: Iterable[str]) -> str:
    label_group = "|".join(re.escape(label) for label in labels)
    stop_group = "|".join(re.escape(label) for label in stop_labels)
    pattern = re.compile(
        rf"(?:^|\n)\s*(?:{label_group})\s*:?\s*(.*?)"
        rf"(?=(?:\n\s*(?:{stop_group})\s*:?)|\Z)",
        flags=re.IGNORECASE | re.DOTALL,
    )
    match = pattern.search(text)
    if not match:
        return ""
    return _clean_text(match.group(1), 12_000)


def _split_report_text(text: str) -> tuple[str, str]:
    cleaned = _clean_text(text, 20_000)
    if not cleaned:
        return "", ""

    findings = _extract_section(
        cleaned,
        labels=("FINDINGS", "OBSERVATIONS", "RESULT", "RESULTS"),
        stop_labels=("IMPRESSION", "CONCLUSION", "CONCLUSIONS", "OPINION", "ASSESSMENT"),
    )
    impression = _extract_section(
        cleaned,
        labels=("IMPRESSION", "CONCLUSION", "CONCLUSIONS", "OPINION", "ASSESSMENT"),
        stop_labels=("RECOMMENDATION", "RECOMMENDATIONS", "SIGNED", "END"),
    )
    if findings and impression:
        return findings, impression

    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", cleaned) if p.strip()]
    if len(paragraphs) >= 2:
        maybe_impression = paragraphs[-1]
        maybe_findings = "\n\n".join(paragraphs[:-1])
        if len(maybe_impression.split()) <= 80:
            return _clean_text(maybe_findings, 12_000), _clean_text(maybe_impression, 4_000)

    return "", ""


def _infer_modality(text: str, *, source_file: str = "") -> str:
    haystack = f"{source_file}\n{text}".lower()

    checks = [
        ("ct-chest", ("ct chest", "chest ct", "thorax ct", "ct thorax")),
        ("ct-abdomen-pelvis", ("abdomen pelvis", "a/p", "ct ap", "ct abdomen pelvis", "ct abd pelvis")),
        ("ct-head", ("head ct", "ct head", "brain ct")),
        ("mri-brain", ("mri brain", "brain mri", "mr brain")),
        ("mri-spine", ("mri cervical", "mri thoracic", "mri lumbar", "spine mri")),
        ("us-abdomen", ("ultrasound abdomen", "abdominal ultrasound", "us abdomen")),
        ("us-pelvis", ("pelvic ultrasound", "us pelvis")),
        ("xr-chest", ("chest x-ray", "chest radiograph", "portable chest", "pa and lateral chest", "cxr")),
        ("xr-kub", ("kub", "abdomen radiograph", "supine abdomen")),
        ("mammo", ("mammogram", "mammography", "screening mammo")),
    ]

    for modality, patterns in checks:
        if any(p in haystack for p in patterns):
            return modality

    if "ct" in haystack:
        return "ct"
    if "mri" in haystack or "mr " in haystack:
        return "mri"
    if "ultrasound" in haystack or " sonograph" in haystack or " us " in f" {haystack} ":
        return "ultrasound"
    if any(token in haystack for token in ("x-ray", "radiograph", "portable chest", "cxr")):
        return "xray"

    return "unknown"


def _extract_labeled_xml_sections(node: ET.Element) -> dict[str, str]:
    sections: dict[str, str] = {}

    for elem in node.iter():
        label = (
            elem.attrib.get("Label")
            or elem.attrib.get("label")
            or elem.attrib.get("Type")
            or elem.attrib.get("type")
            or ""
        )
        label_key = str(label).strip().lower()
        if not label_key:
            continue

        text = _clean_text("".join(elem.itertext()), 20_000)
        if not text:
            continue

        if "finding" in label_key and "findings" not in sections:
            sections["findings"] = text
        elif any(word in label_key for word in ("impression", "conclusion", "opinion")) and "impression" not in sections:
            sections["impression"] = text

    return sections


def _first_value(mapping: Mapping[str, Any], keys: Iterable[str]) -> str:
    for key in keys:
        value = mapping.get(key)
        if value is None:
            continue
        cleaned = _clean_text(value, 20_000)
        if cleaned:
            return cleaned
    return ""


def _clean_text(value: Any, max_len: int) -> str:
    text = str(value or "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[\t\f\v]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = "\n".join(line.strip() for line in text.split("\n"))
    text = text.strip(" \n:")
    if len(text) > max_len:
        text = text[:max_len].rstrip()
    return text


def _case_signature(case: BenchmarkCase) -> str:
    return "\x1f".join(
        [
            case.findings.strip().lower(),
            case.impression.strip().lower(),
            case.modality.strip().lower(),
            case.source_file.strip().lower(),
        ]
    )


def _count_by_modality(items: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in items:
        modality = str(item.get("modality", "unknown") or "unknown").strip().lower()
        modality = modality or "unknown"
        counts[modality] = counts.get(modality, 0) + 1
    return dict(sorted(counts.items(), key=lambda pair: (-pair[1], pair[0])))


def _strip_xml_ns(tag: str) -> str:
    return tag.split("}", 1)[-1] if "}" in tag else tag


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


__all__ = [
    "ARCHIVE_EXTENSIONS",
    "BenchmarkCase",
    "BenchmarkLoadStats",
    "CSV_EXTENSIONS",
    "DISCOVERY_EXTENSIONS",
    "JSON_EXTENSIONS",
    "MAX_CASES_PER_LOAD",
    "MAX_TEXT_READ",
    "SUPPORTED_FILE_EXTENSIONS",
    "TEXT_EXTENSIONS",
    "XML_EXTENSIONS",
    "list_available_datasets",
    "list_benchmark_datasets",
    "load_benchmark_bytes",
    "load_benchmark_path",
    "load_bytes",
    "load_path",
    "parse_txt_case",
]
