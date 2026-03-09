from __future__ import annotations

"""
radcopilot.rag.parser

Parsing helpers for RadCopilot retrieval training and dataset ingestion.

Purpose:
- parse local files and directories into findings -> impression records
- support common report formats used by the RAG trainer
- return normalized-ish dictionaries compatible with radcopilot.rag.library

Supported inputs:
- .txt / .md / .text
- .csv
- .xml
- .json
- .jsonl
- .tar / .tgz / .tar.gz
- directories containing supported files

Returned record shape:
{
    "findings": str,
    "impression": str,
    "modality": str,
    "source": str,
    "created_at": str,
    "source_file": str,
}
"""

import csv
import io
import json
from datetime import datetime, timezone
from pathlib import Path
import re
import tarfile
from typing import Any, Iterable, Iterator
import xml.etree.ElementTree as ET


TEXT_EXTENSIONS = {".txt", ".md", ".text"}
CSV_EXTENSIONS = {".csv"}
XML_EXTENSIONS = {".xml"}
JSON_EXTENSIONS = {".json", ".jsonl"}
ARCHIVE_EXTENSIONS = {".tar", ".tgz", ".gz"}

MAX_TEXT_READ = 2_000_000
MAX_RECORDS_PER_PARSE = 250_000

_FINDINGS_KEYS = (
    "findings",
    "report_findings",
    "report_findings_text",
    "finding",
    "body",
    "text_findings",
)

_IMPRESSION_KEYS = (
    "impression",
    "conclusion",
    "conclusions",
    "opinion",
    "assessment",
    "report_impression",
    "impr",
)

_TEXT_KEYS = (
    "report",
    "text",
    "report_text",
    "full_report",
    "note",
    "content",
)

_MODALITY_KEYS = (
    "modality",
    "study_modality",
    "exam_modality",
    "exam_type",
    "series_description",
)


def parse_path(path: str | Path) -> list[dict[str, Any]]:
    """
    Parse a file or directory into a list of candidate RAG records.

    This function is safe and best-effort. Unsupported files are skipped.
    """
    source_path = Path(path).expanduser().resolve()
    if not source_path.exists():
        return []

    records: list[dict[str, Any]] = []
    for record in iter_path_records(source_path):
        if len(records) >= MAX_RECORDS_PER_PARSE:
            break
        records.append(record)
    return records


def parse_file(path: str | Path) -> list[dict[str, Any]]:
    """
    Parse a single file path into zero or more records.
    """
    source_path = Path(path).expanduser().resolve()
    if not source_path.exists() or not source_path.is_file():
        return []
    return list(_iter_file_records(source_path))


def iter_path_records(source_path: Path) -> Iterator[dict[str, Any]]:
    """
    Yield candidate records from a file or directory.
    """
    if source_path.is_dir():
        yielded = 0
        for child in sorted(source_path.rglob("*")):
            if not child.is_file():
                continue
            for record in _iter_file_records(child):
                yield record
                yielded += 1
                if yielded >= MAX_RECORDS_PER_PARSE:
                    return
        return

    yield from _iter_file_records(source_path)


def parse_txt_record(text: str, *, source_file: str = "") -> dict[str, Any] | None:
    """
    Parse a report-like text block into findings/impression.
    """
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

    modality = _infer_modality(findings + "\n" + impression, source_file=source_file)
    return {
        "findings": findings,
        "impression": impression,
        "modality": modality,
        "source": "txt_file",
        "created_at": _utc_now(),
        "source_file": source_file,
    }


def _iter_file_records(path: Path) -> Iterator[dict[str, Any]]:
    suffix = path.suffix.lower()
    lower_name = path.name.lower()

    try:
        if suffix in TEXT_EXTENSIONS:
            text = _read_text(path)
            record = parse_txt_record(text, source_file=str(path))
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
            yield from _iter_tar_records(path)
            return
    except Exception:
        return


def _iter_csv_records(path: Path) -> Iterator[dict[str, Any]]:
    text = _read_text(path)
    if not text.strip():
        return

    reader = csv.DictReader(io.StringIO(text))
    for row in reader:
        record = _row_to_record(row, source_file=str(path), source="csv_file")
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
            if isinstance(item, dict):
                record = _row_to_record(item, source_file=str(path), source="jsonl_file")
                if record:
                    yield record
        return

    try:
        payload = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except Exception:
        return

    if isinstance(payload, list):
        for item in payload:
            if not isinstance(item, dict):
                continue
            record = _row_to_record(item, source_file=str(path), source="json_file")
            if record:
                yield record

    elif isinstance(payload, dict):
        direct = _row_to_record(payload, source_file=str(path), source="json_file")
        if direct:
            yield direct

        for key in ("items", "records", "cases", "data", "reports"):
            nested = payload.get(key)
            if isinstance(nested, list):
                for item in nested:
                    if not isinstance(item, dict):
                        continue
                    record = _row_to_record(item, source_file=str(path), source="json_file")
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
        record = _xml_node_to_record(node, source_file=str(path))
        if record:
            yielded += 1
            yield record

    if yielded:
        return

    record = _xml_node_to_record(root, source_file=str(path))
    if record:
        yield record


def _iter_tar_records(path: Path) -> Iterator[dict[str, Any]]:
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

                if member_suffix in TEXT_EXTENSIONS:
                    text = data.decode("utf-8", errors="replace")
                    record = parse_txt_record(text, source_file=f"{path}:{member_name}")
                    if record:
                        record["source"] = "tar_txt"
                        yield record

                elif member_suffix in CSV_EXTENSIONS:
                    text = data.decode("utf-8", errors="replace")
                    reader = csv.DictReader(io.StringIO(text))
                    for row in reader:
                        record = _row_to_record(row, source_file=f"{path}:{member_name}", source="tar_csv")
                        if record:
                            yield record

                elif member_suffix in XML_EXTENSIONS:
                    try:
                        root = ET.fromstring(data.decode("utf-8", errors="replace"))
                    except Exception:
                        continue
                    record = _xml_node_to_record(root, source_file=f"{path}:{member_name}")
                    if record:
                        record["source"] = "tar_xml"
                        yield record
    except Exception:
        return


def _row_to_record(row: dict[str, Any], *, source_file: str, source: str) -> dict[str, Any] | None:
    lowered = {str(k).strip().lower(): v for k, v in row.items()}

    findings = _first_value(lowered, _FINDINGS_KEYS)
    impression = _first_value(lowered, _IMPRESSION_KEYS)
    text = _first_value(lowered, _TEXT_KEYS)
    modality = _first_value(lowered, _MODALITY_KEYS)

    if (not findings or not impression) and text:
        findings2, impression2 = _split_report_text(str(text))
        findings = findings or findings2
        impression = impression or impression2

    if not findings or not impression:
        return None

    return {
        "findings": _clean_text(findings, 5000),
        "impression": _clean_text(impression, 3000),
        "modality": _infer_modality(modality or (str(findings) + "\n" + str(impression)), source_file=source_file),
        "source": source,
        "created_at": _utc_now(),
        "source_file": source_file,
    }


def _xml_node_to_record(node: ET.Element, *, source_file: str) -> dict[str, Any] | None:
    text_map: dict[str, str] = {}

    for elem in node.iter():
        tag = _strip_xml_ns(elem.tag).strip().lower()
        value = _clean_text("".join(elem.itertext()), 20000)
        if value:
            text_map.setdefault(tag, value)

    findings = _first_value(text_map, _FINDINGS_KEYS)
    impression = _first_value(text_map, _IMPRESSION_KEYS)
    full_text = _first_value(text_map, _TEXT_KEYS)
    modality = _first_value(text_map, _MODALITY_KEYS)

    if not findings or not impression:
        labeled_sections = _extract_labeled_xml_sections(node)
        findings = findings or labeled_sections.get("findings", "")
        impression = impression or labeled_sections.get("impression", "")

    if (not findings or not impression) and full_text:
        findings2, impression2 = _split_report_text(full_text)
        findings = findings or findings2
        impression = impression or impression2

    if not findings or not impression:
        return None

    return {
        "findings": _clean_text(findings, 5000),
        "impression": _clean_text(impression, 3000),
        "modality": _infer_modality(modality or (str(findings) + "\n" + str(impression)), source_file=source_file),
        "source": "xml_file",
        "created_at": _utc_now(),
        "source_file": source_file,
    }


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

        text = _clean_text("".join(elem.itertext()), 20000)
        if not text:
            continue

        if "finding" in label_key and "findings" not in sections:
            sections["findings"] = text
        elif any(word in label_key for word in ("impression", "conclusion", "opinion")) and "impression" not in sections:
            sections["impression"] = text

    return sections


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")[:MAX_TEXT_READ]


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
    return _clean_text(match.group(1), 12000)


def _split_report_text(text: str) -> tuple[str, str]:
    cleaned = _clean_text(text, 20000)
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
            return _clean_text(maybe_findings, 12000), _clean_text(maybe_impression, 4000)

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


def _first_value(mapping: dict[str, Any], keys: Iterable[str]) -> str:
    for key in keys:
        value = mapping.get(key)
        if value is None:
            continue
        cleaned = _clean_text(value, 20000)
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


def _strip_xml_ns(tag: str) -> str:
    return tag.split("}", 1)[-1] if "}" in tag else tag


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


__all__ = [
    "ARCHIVE_EXTENSIONS",
    "CSV_EXTENSIONS",
    "JSON_EXTENSIONS",
    "MAX_RECORDS_PER_PARSE",
    "MAX_TEXT_READ",
    "TEXT_EXTENSIONS",
    "XML_EXTENSIONS",
    "iter_path_records",
    "parse_file",
    "parse_path",
    "parse_txt_record",
]
