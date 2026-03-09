"""Validation utilities for RadCopilot report output.

This module focuses on validating radiology impressions generated from a
reporting pipeline. The goal is not to perform clinical reasoning, but to apply
structured checks that catch common generation defects before a report is
accepted or handed to a repair step.

Design goals:
- standard-library only
- deterministic and explainable checks
- compatible with future repair / scoring pipeline modules
- safe fallback behavior when little context is available
"""
from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


_NEGATION_PATTERNS = (
    r"\bno\b",
    r"\bwithout\b",
    r"\bnegative for\b",
    r"\babsent\b",
    r"\bfree of\b",
    r"\bnot seen\b",
    r"\bno evidence of\b",
    r"\bno acute\b",
)

_PROMPT_LEAK_PATTERNS = (
    r"\byou are\b",
    r"\bdo not\b",
    r"\brespond with\b",
    r"\bsection\s*\|\s*",
    r"\bimpression rules\b",
    r"\bfindings to sections\b",
    r"\bverbatim\b",
    r"\btemplate\b",
    r"\boutput format\b",
)

_ORGAN_KEYWORDS: Dict[str, Tuple[str, ...]] = {
    "lung": ("lung", "pulmonary", "nodule", "pleural", "pneumothorax", "airspace"),
    "adrenal": ("adrenal", "adrenals"),
    "adnexa": ("adnexa", "adnexal", "ovary", "ovarian", "fallopian"),
    "liver": ("liver", "hepatic"),
    "kidney": ("kidney", "renal", "hydronephrosis"),
    "bowel": ("bowel", "colon", "colonic", "ileum", "jejunum", "small bowel"),
    "brain": ("brain", "intracranial", "hemorrhage", "stroke"),
    "heart": ("heart", "cardiac", "pericardial", "coronary"),
}

_GUIDELINE_PATTERNS: Dict[str, Tuple[str, ...]] = {
    "fleischner": ("fleischner",),
    "birads": ("bi-rads", "birads"),
    "li-rads": ("li-rads", "lirads"),
    "ti-rads": ("ti-rads", "tirads"),
    "bosniak": ("bosniak",),
}

_NUMBER_RE = re.compile(r"\b(\d+(?:\.\d+)?)\s*(mm|cm)\b", re.IGNORECASE)
_SENTENCE_SPLIT_RE = re.compile(r"\n+|(?<=[.!?])\s+")
_LINE_NUMBER_RE = re.compile(r"^\s*(\d+)[\.)]\s+")
_BULLET_PREFIX_RE = re.compile(r"^\s*(?:[-*•]|\d+[\.)])\s*")


@dataclass(slots=True)
class ValidationIssue:
    """A deterministic validation finding."""

    code: str
    message: str
    severity: str = "warning"
    line_index: Optional[int] = None
    line_text: Optional[str] = None
    metadata: Dict[str, str] = field(default_factory=dict)


@dataclass(slots=True)
class ValidationResult:
    """Structured result returned by the validator."""

    valid: bool
    issues: List[ValidationIssue] = field(default_factory=list)
    normalized_impression: str = ""
    lines: List[str] = field(default_factory=list)
    score: float = 1.0

    def error_count(self) -> int:
        return sum(1 for issue in self.issues if issue.severity == "error")

    def warning_count(self) -> int:
        return sum(1 for issue in self.issues if issue.severity == "warning")

    def to_dict(self) -> Dict[str, object]:
        return {
            "valid": self.valid,
            "score": round(self.score, 4),
            "normalized_impression": self.normalized_impression,
            "lines": self.lines,
            "issues": [
                {
                    "code": issue.code,
                    "message": issue.message,
                    "severity": issue.severity,
                    "line_index": issue.line_index,
                    "line_text": issue.line_text,
                    "metadata": issue.metadata,
                }
                for issue in self.issues
            ],
        }


def validate_impression(
    impression: str,
    *,
    findings_text: str = "",
    abnormals: Optional[Dict[str, Sequence[str]]] = None,
    template_id: Optional[str] = None,
    allow_negatives: bool = False,
) -> ValidationResult:
    """Validate an impression for common report-generation defects.

    Args:
        impression: Generated impression text.
        findings_text: Original findings input, used for rough consistency checks.
        abnormals: Optional map of abnormal section -> list of finding strings.
        template_id: Optional template identifier used for exam-specific heuristics.
        allow_negatives: Whether negative statements are acceptable in impression.

    Returns:
        ValidationResult with issue list, normalized lines, and a confidence score.
    """
    cleaned = normalize_impression(impression)
    lines = split_impression_lines(cleaned)
    issues: List[ValidationIssue] = []

    if not cleaned:
        issues.append(ValidationIssue("empty_impression", "Impression is empty.", "error"))
        return _build_result(cleaned, lines, issues)

    issues.extend(_check_prompt_leak(cleaned, lines))
    issues.extend(_check_numbering(lines))
    issues.extend(_check_duplicate_lines(lines))
    issues.extend(_check_negatives(lines, allow_negatives=allow_negatives))
    issues.extend(_check_guidelines(lines, findings_text=findings_text, template_id=template_id))
    issues.extend(_check_measurement_mismatch(lines, findings_text=findings_text))
    issues.extend(_check_anatomy_mismatch(lines, findings_text=findings_text, abnormals=abnormals))
    issues.extend(_check_hallucinated_lines(lines, findings_text=findings_text, abnormals=abnormals))
    issues.extend(_check_length(cleaned, lines))

    return _build_result(cleaned, lines, issues)


def normalize_impression(impression: str) -> str:
    """Normalize impression text for validation.

    Removes leading headers such as "Impression:" and trims whitespace while
    preserving line structure when practical.
    """
    text = (impression or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    text = re.sub(r"^\s*impression\s*:\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_impression_lines(impression: str) -> List[str]:
    """Split impression into normalized logical lines.

    Preference order:
    1. explicit line breaks / bullets
    2. numbered lines
    3. sentence boundaries
    """
    if not impression:
        return []

    raw_parts = [part.strip() for part in impression.split("\n") if part.strip()]
    if len(raw_parts) > 1:
        return [_strip_prefix(part) for part in raw_parts if _strip_prefix(part)]

    numbered = re.split(r"\s+(?=\d+[\.)]\s+)", impression)
    if len(numbered) > 1:
        return [_strip_prefix(part.strip()) for part in numbered if _strip_prefix(part.strip())]

    parts = [part.strip() for part in _SENTENCE_SPLIT_RE.split(impression) if part.strip()]
    return [_strip_prefix(part) for part in parts if _strip_prefix(part)]


def summarize_issues(result: ValidationResult) -> str:
    """Return a concise, human-readable issue summary."""
    if not result.issues:
        return "No validation issues detected."
    chunks = []
    for issue in result.issues:
        prefix = f"[{issue.severity.upper()}] {issue.code}"
        chunks.append(f"{prefix}: {issue.message}")
    return " | ".join(chunks)


def should_attempt_repair(result: ValidationResult) -> bool:
    """Determine whether a repair pass is warranted."""
    if not result.issues:
        return False
    return any(issue.severity == "error" for issue in result.issues) or result.score < 0.82


def get_repair_targets(result: ValidationResult) -> List[str]:
    """Return unique issue messages suitable for a repair prompt."""
    seen = set()
    targets: List[str] = []
    for issue in result.issues:
        message = issue.message.strip()
        if message and message not in seen:
            seen.add(message)
            targets.append(message)
    return targets


def _build_result(cleaned: str, lines: List[str], issues: List[ValidationIssue]) -> ValidationResult:
    score = _score_issues(issues)
    valid = not any(issue.severity == "error" for issue in issues)
    return ValidationResult(
        valid=valid,
        issues=issues,
        normalized_impression=cleaned,
        lines=lines,
        score=score,
    )


def _score_issues(issues: Sequence[ValidationIssue]) -> float:
    score = 1.0
    for issue in issues:
        if issue.severity == "error":
            score -= 0.15
        elif issue.severity == "warning":
            score -= 0.05
        else:
            score -= 0.02
    return max(0.0, min(1.0, score))


def _strip_prefix(text: str) -> str:
    return _BULLET_PREFIX_RE.sub("", text).strip()


def _check_prompt_leak(cleaned: str, lines: Sequence[str]) -> List[ValidationIssue]:
    issues: List[ValidationIssue] = []
    for pat in _PROMPT_LEAK_PATTERNS:
        if re.search(pat, cleaned, flags=re.IGNORECASE):
            issues.append(
                ValidationIssue(
                    "prompt_leak",
                    "Impression appears to contain prompt or instruction text.",
                    "error",
                )
            )
            break
    for idx, line in enumerate(lines):
        if "|" in line and re.search(r"\bsection\b", line, flags=re.IGNORECASE):
            issues.append(
                ValidationIssue(
                    "mapping_leak",
                    "Line appears to include findings-to-section mapping output.",
                    "error",
                    line_index=idx,
                    line_text=line,
                )
            )
    return issues


def _check_numbering(lines: Sequence[str]) -> List[ValidationIssue]:
    issues: List[ValidationIssue] = []
    raw_numbers: List[int] = []
    for idx, line in enumerate(lines):
        match = _LINE_NUMBER_RE.match(line)
        if match:
            raw_numbers.append(int(match.group(1)))
            issues.append(
                ValidationIssue(
                    "line_prefix_present",
                    "Line still contains numbering prefix after normalization.",
                    "warning",
                    line_index=idx,
                    line_text=line,
                )
            )

    if raw_numbers and raw_numbers != list(range(1, len(raw_numbers) + 1)):
        issues.append(
            ValidationIssue(
                "non_sequential_numbering",
                "Impression numbering is non-sequential or malformed.",
                "warning",
            )
        )

    if not lines:
        return issues

    if len(lines) == 1 and len(lines[0].split()) < 3:
        issues.append(
            ValidationIssue(
                "too_short",
                "Impression is too short to be clinically useful.",
                "error",
                line_index=0,
                line_text=lines[0],
            )
        )
    return issues


def _check_duplicate_lines(lines: Sequence[str]) -> List[ValidationIssue]:
    issues: List[ValidationIssue] = []
    seen: Dict[str, int] = {}
    for idx, line in enumerate(lines):
        key = _normalize_for_compare(line)
        if key in seen:
            issues.append(
                ValidationIssue(
                    "duplicate_line",
                    "Duplicate or near-duplicate impression line detected.",
                    "warning",
                    line_index=idx,
                    line_text=line,
                    metadata={"first_index": str(seen[key])},
                )
            )
        else:
            seen[key] = idx
    return issues


def _check_negatives(lines: Sequence[str], *, allow_negatives: bool) -> List[ValidationIssue]:
    if allow_negatives:
        return []

    issues: List[ValidationIssue] = []
    for idx, line in enumerate(lines):
        lower = line.lower()
        if any(re.search(pat, lower) for pat in _NEGATION_PATTERNS):
            issues.append(
                ValidationIssue(
                    "negative_in_impression",
                    "Negative or exclusionary statement detected in impression.",
                    "warning",
                    line_index=idx,
                    line_text=line,
                )
            )
    return issues


def _check_guidelines(
    lines: Sequence[str], *, findings_text: str, template_id: Optional[str]
) -> List[ValidationIssue]:
    issues: List[ValidationIssue] = []
    combined = " ".join(lines).lower()
    findings_lower = findings_text.lower()
    template_lower = (template_id or "").lower()

    if any(token in combined for token in _GUIDELINE_PATTERNS["fleischner"]):
        pulmonary_context = any(word in findings_lower for word in _ORGAN_KEYWORDS["lung"]) or "chest" in template_lower
        if not pulmonary_context:
            issues.append(
                ValidationIssue(
                    "guideline_mismatch",
                    "Fleischner guidance appears outside a pulmonary nodule context.",
                    "error",
                )
            )

    if any(token in combined for token in _GUIDELINE_PATTERNS["birads"]):
        breast_context = "breast" in findings_lower or "breast" in template_lower or "mamm" in template_lower
        if not breast_context:
            issues.append(
                ValidationIssue(
                    "guideline_mismatch",
                    "BI-RADS guidance appears outside a breast imaging context.",
                    "error",
                )
            )

    if any(token in combined for token in _GUIDELINE_PATTERNS["bosniak"]):
        renal_context = any(word in findings_lower for word in _ORGAN_KEYWORDS["kidney"])
        if not renal_context:
            issues.append(
                ValidationIssue(
                    "guideline_mismatch",
                    "Bosniak classification appears outside a renal cyst context.",
                    "warning",
                )
            )
    return issues


def _check_measurement_mismatch(lines: Sequence[str], *, findings_text: str) -> List[ValidationIssue]:
    issues: List[ValidationIssue] = []
    if not findings_text.strip():
        return issues

    source_measurements = set(_NUMBER_RE.findall(findings_text))
    if not source_measurements:
        return issues

    source_strings = {f"{num}{unit.lower()}" for num, unit in source_measurements}
    impression_measurements = {
        f"{num}{unit.lower()}" for line in lines for num, unit in _NUMBER_RE.findall(line)
    }
    for item in sorted(impression_measurements - source_strings):
        issues.append(
            ValidationIssue(
                "measurement_not_in_findings",
                f"Measurement {item} appears in impression but not in findings.",
                "warning",
                metadata={"measurement": item},
            )
        )
    return issues


def _check_anatomy_mismatch(
    lines: Sequence[str], *, findings_text: str, abnormals: Optional[Dict[str, Sequence[str]]]
) -> List[ValidationIssue]:
    issues: List[ValidationIssue] = []
    source_text = " ".join(_flatten_abnormals(abnormals)) or findings_text
    source_lower = source_text.lower()
    if not source_lower.strip():
        return issues

    allowed_organs = {
        organ for organ, keywords in _ORGAN_KEYWORDS.items() if any(word in source_lower for word in keywords)
    }
    if not allowed_organs:
        return issues

    for idx, line in enumerate(lines):
        line_lower = line.lower()
        for organ, keywords in _ORGAN_KEYWORDS.items():
            if organ in allowed_organs:
                continue
            if any(word in line_lower for word in keywords):
                issues.append(
                    ValidationIssue(
                        "anatomy_mismatch",
                        f"Line may introduce unsupported anatomy: {organ}.",
                        "warning",
                        line_index=idx,
                        line_text=line,
                        metadata={"organ": organ},
                    )
                )
                break

    if "adrenal" in source_lower:
        for idx, line in enumerate(lines):
            if any(word in line.lower() for word in _ORGAN_KEYWORDS["adnexa"]):
                issues.append(
                    ValidationIssue(
                        "adrenal_adnexa_confusion",
                        "Possible adrenal/adnexal confusion detected.",
                        "error",
                        line_index=idx,
                        line_text=line,
                    )
                )
    if any(word in source_lower for word in _ORGAN_KEYWORDS["adnexa"]):
        for idx, line in enumerate(lines):
            if any(word in line.lower() for word in _ORGAN_KEYWORDS["adrenal"]):
                issues.append(
                    ValidationIssue(
                        "adrenal_adnexa_confusion",
                        "Possible adnexal/adrenal confusion detected.",
                        "error",
                        line_index=idx,
                        line_text=line,
                    )
                )
    return issues


def _check_hallucinated_lines(
    lines: Sequence[str], *, findings_text: str, abnormals: Optional[Dict[str, Sequence[str]]]
) -> List[ValidationIssue]:
    issues: List[ValidationIssue] = []
    source = " ".join(_flatten_abnormals(abnormals)) or findings_text
    if not source.strip():
        return issues

    source_tokens = _significant_tokens(source)
    for idx, line in enumerate(lines):
        line_tokens = _significant_tokens(line)
        if not line_tokens:
            issues.append(
                ValidationIssue(
                    "empty_line_content",
                    "Impression line has insufficient content.",
                    "warning",
                    line_index=idx,
                    line_text=line,
                )
            )
            continue

        overlap = line_tokens & source_tokens
        if not overlap and len(line_tokens) >= 3:
            issues.append(
                ValidationIssue(
                    "possible_hallucination",
                    "Impression line has little lexical overlap with source findings.",
                    "warning",
                    line_index=idx,
                    line_text=line,
                )
            )
    return issues


def _check_length(cleaned: str, lines: Sequence[str]) -> List[ValidationIssue]:
    issues: List[ValidationIssue] = []
    if len(cleaned) > 1600:
        issues.append(
            ValidationIssue(
                "impression_too_long",
                "Impression is unusually long for a summary section.",
                "warning",
            )
        )
    if len(lines) > 8:
        issues.append(
            ValidationIssue(
                "too_many_lines",
                "Impression contains many lines and may be overly verbose.",
                "warning",
            )
        )
    return issues


def _flatten_abnormals(abnormals: Optional[Dict[str, Sequence[str]]]) -> List[str]:
    if not abnormals:
        return []
    flat: List[str] = []
    for values in abnormals.values():
        flat.extend(str(v) for v in values if str(v).strip())
    return flat


def _normalize_for_compare(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _significant_tokens(text: str) -> set[str]:
    stopwords = {
        "the", "a", "an", "and", "or", "of", "to", "for", "with", "on", "in",
        "is", "are", "was", "were", "this", "that", "there", "it", "as", "at",
        "by", "from", "be", "has", "have", "had", "likely", "may", "can",
        "mild", "moderate", "severe", "consistent", "compatible", "suggesting",
    }
    tokens = set(re.findall(r"[a-z0-9]+", text.lower()))
    return {tok for tok in tokens if tok not in stopwords and len(tok) > 2}


__all__ = [
    "ValidationIssue",
    "ValidationResult",
    "validate_impression",
    "normalize_impression",
    "split_impression_lines",
    "summarize_issues",
    "should_attempt_repair",
    "get_repair_targets",
]
