"""Deterministic impression repair utilities for RadCopilot.

This module sits between validation and any optional LLM repair pass. Its job
is to make safe, explainable, standard-library-only corrections to generated
impressions before the pipeline decides whether a model retry is still needed.

The implementation is intentionally conservative:
- remove obvious prompt / mapping leakage
- remove duplicate or low-value lines
- remove lines flagged as unsupported or risky when the validator provides a
  specific line target
- normalize formatting and numbering
- keep the best clinically plausible subset rather than inventing new content

It does *not* attempt medical reasoning or synthesis. If the remaining output is
still weak after deterministic cleanup, the caller should use the validator's
repair targets to drive an LLM correction pass.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from .validator import (
    ValidationIssue,
    ValidationResult,
    normalize_impression,
    split_impression_lines,
    summarize_issues,
    validate_impression,
)


_NEGATIVE_PATTERNS = (
    r"\bno\b",
    r"\bwithout\b",
    r"\bnegative for\b",
    r"\babsent\b",
    r"\bnot seen\b",
    r"\bno evidence of\b",
    r"\bno acute\b",
)

_PROMPT_LEAK_PATTERNS = (
    r"\byou are\b",
    r"\bdo not\b",
    r"\brespond with\b",
    r"\bsection\s*\|\s*",
    r"\btemplate\b",
    r"\boutput format\b",
    r"\bfindings to sections\b",
    r"\bimpression rules\b",
)

_HEADER_PATTERNS = (
    r"^\s*impression\s*:\s*",
    r"^\s*conclusion\s*:\s*",
    r"^\s*summary\s*:\s*",
)

_BULLET_RE = re.compile(r"^\s*(?:[-*•]|\d+[\.)])\s*")
_NUMBER_RE = re.compile(r"\b(\d+(?:\.\d+)?)\s*(mm|cm)\b", re.IGNORECASE)
_MULTISPACE_RE = re.compile(r"\s+")

# Issue codes where removing the targeted line is generally safer than trying to
# rewrite it deterministically.
_REMOVE_LINE_CODES: Set[str] = {
    "mapping_leak",
    "duplicate_line",
    "empty_line_content",
    "possible_hallucination",
    "anatomy_mismatch",
    "adrenal_adnexa_confusion",
    "guideline_mismatch",
    "negative_in_impression",
}

# Issue codes that affect the entire impression or require global cleanup.
_GLOBAL_CODES: Set[str] = {
    "prompt_leak",
    "non_sequential_numbering",
    "impression_too_long",
    "too_many_lines",
    "line_prefix_present",
}


@dataclass(slots=True)
class FixAction:
    """Single deterministic change applied during impression cleanup."""

    action: str
    detail: str
    line_index: Optional[int] = None
    before: Optional[str] = None
    after: Optional[str] = None

    def to_dict(self) -> Dict[str, object]:
        return {
            "action": self.action,
            "detail": self.detail,
            "line_index": self.line_index,
            "before": self.before,
            "after": self.after,
        }


@dataclass(slots=True)
class FixResult:
    """Structured output from deterministic repair."""

    original_impression: str
    fixed_impression: str
    original_validation: ValidationResult
    final_validation: ValidationResult
    changed: bool
    actions: List[FixAction] = field(default_factory=list)
    removed_lines: List[str] = field(default_factory=list)
    kept_lines: List[str] = field(default_factory=list)
    should_escalate_to_llm: bool = False

    def summary(self) -> str:
        if not self.actions:
            return "No deterministic fix actions applied."
        return " | ".join(f"{a.action}: {a.detail}" for a in self.actions)

    def to_dict(self) -> Dict[str, object]:
        return {
            "changed": self.changed,
            "original_impression": self.original_impression,
            "fixed_impression": self.fixed_impression,
            "removed_lines": self.removed_lines,
            "kept_lines": self.kept_lines,
            "should_escalate_to_llm": self.should_escalate_to_llm,
            "actions": [a.to_dict() for a in self.actions],
            "original_validation": self.original_validation.to_dict(),
            "final_validation": self.final_validation.to_dict(),
        }


def auto_fix_impression(
    impression: str,
    *,
    findings_text: str = "",
    abnormals: Optional[Dict[str, Sequence[str]]] = None,
    template_id: Optional[str] = None,
    allow_negatives: bool = False,
    max_lines: int = 5,
    prefer_numbered_output: bool = True,
) -> FixResult:
    """Apply conservative deterministic cleanup to an impression.

    Typical usage:
    1. validate the impression
    2. call auto_fix_impression(...)
    3. inspect final_validation
    4. if still invalid, escalate to an LLM repair pass
    """
    original = normalize_impression(impression)
    original_validation = validate_impression(
        original,
        findings_text=findings_text,
        abnormals=abnormals,
        template_id=template_id,
        allow_negatives=allow_negatives,
    )

    actions: List[FixAction] = []
    removed_lines: List[str] = []

    lines = [clean_line(line, actions=actions, line_index=idx) for idx, line in enumerate(original_validation.lines)]
    lines = [line for line in lines if line]

    if original_validation.lines and not lines and original:
        # Entire impression may have been a single leak/header-like string.
        fallback_lines = split_impression_lines(original)
        lines = [clean_line(line, actions=actions) for line in fallback_lines if clean_line(line)]
        lines = [line for line in lines if line]

    remove_indexes = _collect_line_indexes_to_remove(original_validation)

    filtered_lines: List[str] = []
    for idx, line in enumerate(lines):
        if idx in remove_indexes:
            removed_lines.append(line)
            actions.append(
                FixAction(
                    action="remove_line",
                    detail="Removed line flagged by validator.",
                    line_index=idx,
                    before=line,
                    after=None,
                )
            )
            continue
        filtered_lines.append(line)

    deduped = dedupe_lines(filtered_lines, actions=actions, removed_lines=removed_lines)

    if not allow_negatives:
        deduped = remove_negative_lines(deduped, actions=actions, removed_lines=removed_lines)

    deduped = remove_prompt_leak_lines(deduped, actions=actions, removed_lines=removed_lines)
    deduped = prune_measurement_mismatch(deduped, findings_text=findings_text, actions=actions, removed_lines=removed_lines)
    deduped = prune_verbosity(deduped, max_lines=max_lines, findings_text=findings_text, actions=actions, removed_lines=removed_lines)

    fixed_impression = format_impression(deduped, numbered=prefer_numbered_output)

    final_validation = validate_impression(
        fixed_impression,
        findings_text=findings_text,
        abnormals=abnormals,
        template_id=template_id,
        allow_negatives=allow_negatives,
    )

    changed = _normalize_compare(original) != _normalize_compare(fixed_impression)
    should_escalate = bool(final_validation.issues) and (
        final_validation.error_count() > 0 or final_validation.score < 0.9
    )

    return FixResult(
        original_impression=original,
        fixed_impression=fixed_impression,
        original_validation=original_validation,
        final_validation=final_validation,
        changed=changed,
        actions=actions,
        removed_lines=removed_lines,
        kept_lines=deduped,
        should_escalate_to_llm=should_escalate,
    )


def clean_line(
    line: str,
    *,
    actions: Optional[List[FixAction]] = None,
    line_index: Optional[int] = None,
) -> str:
    """Normalize a single impression line without changing its meaning."""
    original = str(line or "")
    text = original.strip()
    for pat in _HEADER_PATTERNS:
        updated = re.sub(pat, "", text, flags=re.IGNORECASE)
        if updated != text and actions is not None:
            actions.append(
                FixAction(
                    action="strip_header",
                    detail="Removed leading impression-style header.",
                    line_index=line_index,
                    before=text,
                    after=updated.strip(),
                )
            )
        text = updated

    updated = _BULLET_RE.sub("", text).strip()
    if updated != text and actions is not None:
        actions.append(
            FixAction(
                action="strip_bullet",
                detail="Removed bullet or numbering prefix.",
                line_index=line_index,
                before=text,
                after=updated,
            )
        )
    text = updated

    # Remove obvious mapping prefix like "Lung | nodule..."
    pipe_parts = [part.strip() for part in text.split("|", 1)]
    if len(pipe_parts) == 2 and pipe_parts[0] and len(pipe_parts[0].split()) <= 4:
        candidate = pipe_parts[1]
        if candidate:
            if actions is not None:
                actions.append(
                    FixAction(
                        action="strip_mapping_prefix",
                        detail="Removed section-to-findings mapping prefix.",
                        line_index=line_index,
                        before=text,
                        after=candidate,
                    )
                )
            text = candidate

    text = text.replace("\t", " ")
    text = _MULTISPACE_RE.sub(" ", text).strip(" ;,\n\t")
    text = re.sub(r"\s+([,.;:])", r"\1", text)

    # Ensure terminal punctuation for readability.
    if text and text[-1] not in ".;":
        text = f"{text}."

    return text.strip()


def dedupe_lines(
    lines: Sequence[str],
    *,
    actions: Optional[List[FixAction]] = None,
    removed_lines: Optional[List[str]] = None,
) -> List[str]:
    """Remove exact and near-duplicate lines while preserving order."""
    kept: List[str] = []
    seen: Set[str] = set()
    for idx, line in enumerate(lines):
        key = _normalize_compare(line)
        if not key:
            continue
        if key in seen:
            if removed_lines is not None:
                removed_lines.append(line)
            if actions is not None:
                actions.append(
                    FixAction(
                        action="dedupe_line",
                        detail="Removed duplicate or near-duplicate line.",
                        line_index=idx,
                        before=line,
                        after=None,
                    )
                )
            continue
        seen.add(key)
        kept.append(line)
    return kept


def remove_negative_lines(
    lines: Sequence[str],
    *,
    actions: Optional[List[FixAction]] = None,
    removed_lines: Optional[List[str]] = None,
) -> List[str]:
    """Drop negative/exclusionary lines from impression when negatives are disallowed."""
    kept: List[str] = []
    for idx, line in enumerate(lines):
        lower = line.lower()
        if any(re.search(pat, lower) for pat in _NEGATIVE_PATTERNS):
            if removed_lines is not None:
                removed_lines.append(line)
            if actions is not None:
                actions.append(
                    FixAction(
                        action="remove_negative_line",
                        detail="Removed negative or exclusionary statement from impression.",
                        line_index=idx,
                        before=line,
                        after=None,
                    )
                )
            continue
        kept.append(line)
    return kept


def remove_prompt_leak_lines(
    lines: Sequence[str],
    *,
    actions: Optional[List[FixAction]] = None,
    removed_lines: Optional[List[str]] = None,
) -> List[str]:
    """Remove lines that still look like leaked prompt or control text."""
    kept: List[str] = []
    for idx, line in enumerate(lines):
        if any(re.search(pat, line, flags=re.IGNORECASE) for pat in _PROMPT_LEAK_PATTERNS):
            if removed_lines is not None:
                removed_lines.append(line)
            if actions is not None:
                actions.append(
                    FixAction(
                        action="remove_prompt_leak",
                        detail="Removed line containing prompt or instruction text.",
                        line_index=idx,
                        before=line,
                        after=None,
                    )
                )
            continue
        kept.append(line)
    return kept


def prune_measurement_mismatch(
    lines: Sequence[str],
    *,
    findings_text: str,
    actions: Optional[List[FixAction]] = None,
    removed_lines: Optional[List[str]] = None,
) -> List[str]:
    """Remove lines whose measurements are unsupported by findings.

    This is intentionally conservative. If a line introduces measurements not
    present in source findings, the safest deterministic action is to drop the
    whole line rather than attempt partial surgery.
    """
    findings_measurements = {f"{n}{u.lower()}" for n, u in _NUMBER_RE.findall(findings_text or "")}
    if not findings_measurements:
        return list(lines)

    kept: List[str] = []
    for idx, line in enumerate(lines):
        line_measurements = {f"{n}{u.lower()}" for n, u in _NUMBER_RE.findall(line)}
        if line_measurements and not line_measurements.issubset(findings_measurements):
            if removed_lines is not None:
                removed_lines.append(line)
            if actions is not None:
                actions.append(
                    FixAction(
                        action="remove_measurement_mismatch",
                        detail="Removed line containing measurement(s) absent from findings.",
                        line_index=idx,
                        before=line,
                        after=None,
                    )
                )
            continue
        kept.append(line)
    return kept


def prune_verbosity(
    lines: Sequence[str],
    *,
    max_lines: int = 5,
    findings_text: str = "",
    actions: Optional[List[FixAction]] = None,
    removed_lines: Optional[List[str]] = None,
) -> List[str]:
    """Trim low-value lines when the impression becomes too long."""
    working = [line for line in lines if line.strip()]
    if len(working) <= max_lines:
        return working

    ranked = sorted(
        ((idx, line, _line_priority(line, findings_text=findings_text)) for idx, line in enumerate(working)),
        key=lambda item: item[2],
        reverse=True,
    )
    keep_indexes = {idx for idx, _, _ in ranked[:max_lines]}

    kept: List[str] = []
    for idx, line in enumerate(working):
        if idx in keep_indexes:
            kept.append(line)
            continue
        if removed_lines is not None:
            removed_lines.append(line)
        if actions is not None:
            actions.append(
                FixAction(
                    action="trim_verbose_line",
                    detail=f"Trimmed line to keep impression within {max_lines} lines.",
                    line_index=idx,
                    before=line,
                    after=None,
                )
            )
    return kept


def format_impression(lines: Sequence[str], *, numbered: bool = True) -> str:
    """Render normalized lines into final impression text."""
    cleaned = [clean_line(line) for line in lines if clean_line(line)]
    if not cleaned:
        return ""
    if numbered:
        return "\n".join(f"{idx}. {line}" for idx, line in enumerate(cleaned, start=1))
    return "\n".join(cleaned)


def build_repair_context(result: FixResult) -> Dict[str, object]:
    """Return a compact, prompt-friendly summary for optional LLM repair."""
    return {
        "changed": result.changed,
        "fixed_impression": result.fixed_impression,
        "validation_summary": summarize_issues(result.final_validation),
        "remaining_targets": [issue.message for issue in result.final_validation.issues],
        "actions": [action.to_dict() for action in result.actions],
    }


def _collect_line_indexes_to_remove(validation: ValidationResult) -> Set[int]:
    remove_indexes: Set[int] = set()
    for issue in validation.issues:
        if issue.line_index is None:
            continue
        if issue.code in _REMOVE_LINE_CODES:
            remove_indexes.add(issue.line_index)
    return remove_indexes


def _line_priority(line: str, *, findings_text: str = "") -> Tuple[int, int, int, int]:
    """Higher tuple sorts earlier when trimming verbose impressions.

    Priority order:
    1. line contains measurement
    2. lexical overlap with findings
    3. line length (moderate information density)
    4. penalty for recommendation-only lines
    """
    lower = line.lower()
    has_measurement = 1 if _NUMBER_RE.search(line) else 0
    overlap = len(_significant_tokens(lower) & _significant_tokens(findings_text.lower()))
    length_score = min(len(line.split()), 20)
    recommendation_penalty = -1 if any(tok in lower for tok in ("follow-up", "recommend", "correlate")) else 0
    return (has_measurement, overlap, length_score, recommendation_penalty)


def _normalize_compare(text: str) -> str:
    text = (text or "").lower().strip()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _significant_tokens(text: str) -> Set[str]:
    stopwords = {
        "the", "a", "an", "and", "or", "of", "to", "for", "with", "on", "in",
        "is", "are", "was", "were", "this", "that", "there", "it", "as", "at",
        "by", "from", "be", "has", "have", "had", "likely", "may", "can",
        "mild", "moderate", "severe", "consistent", "compatible", "suggesting",
        "findings", "impression", "line",
    }
    tokens = set(re.findall(r"[a-z0-9]+", (text or "").lower()))
    return {tok for tok in tokens if tok not in stopwords and len(tok) > 2}


__all__ = [
    "FixAction",
    "FixResult",
    "auto_fix_impression",
    "clean_line",
    "dedupe_lines",
    "remove_negative_lines",
    "remove_prompt_leak_lines",
    "prune_measurement_mismatch",
    "prune_verbosity",
    "format_impression",
    "build_repair_context",
]
