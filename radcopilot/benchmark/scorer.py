from __future__ import annotations

"""
radcopilot.benchmark.scorer

Deterministic benchmark scoring utilities for RadCopilot.

Purpose
-------
This module scores generated radiology-style benchmark outputs against
reference answers using explainable, standard-library-only metrics.

Design goals
------------
- standard-library only
- JSON-serializable outputs for route/UI use
- robust to partial or messy benchmark payloads
- compatible with the current modular refactor
- able to score either single cases or whole datasets
- validator-aware when `radcopilot.report.validator` is available

What it scores
--------------
Primary target:
- findings -> impression generation quality

Supported scoring shapes:
- a single prediction vs a single reference
- a benchmark case dict containing both generated and reference text
- a list of cases plus aligned predictions
- a list of cases plus a predictor callback

Returned metrics
----------------
- exact match
- token precision / recall / F1
- content-token precision / recall / F1
- Jaccard overlap
- ROUGE-1 recall
- ROUGE-2 recall
- ROUGE-L F1
- sentence coverage
- negation consistency
- number consistency
- length similarity
- optional validator score / issue counts
- weighted composite score

Notes
-----
This scorer does NOT perform clinical reasoning. It only measures textual /
structural alignment and local consistency.
"""

from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import math
import re
from statistics import mean
from typing import Any, Callable, Iterable, Iterator, Mapping, MutableMapping, Protocol, Sequence

# Optional validator integration.
try:  # pragma: no cover - import path depends on package execution mode
    from radcopilot.report.validator import validate_impression
except Exception:  # pragma: no cover - fallback for partial environments
    try:
        from ..report.validator import validate_impression  # type: ignore
    except Exception:  # pragma: no cover - validator truly unavailable
        validate_impression = None  # type: ignore[assignment]


MAX_TEXT_CHARS = 50_000
MAX_FINDINGS_CHARS = 50_000
MAX_NOTES_CHARS = 5_000
MAX_CASES_PER_RUN = 50_000

# Conservative stopwords to reduce noise in content-token metrics.
_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "have",
    "in", "is", "it", "its", "of", "on", "or", "that", "the", "their", "there",
    "this", "to", "was", "were", "with", "within", "without", "into", "than",
    "then", "these", "those", "which", "also", "may", "can", "could", "would",
    "should", "noted", "not", "no", "none", "mild", "moderate", "severe",
}

_NEGATION_PATTERNS = (
    r"\bno\b",
    r"\bwithout\b",
    r"\bnegative for\b",
    r"\babsent\b",
    r"\bfree of\b",
    r"\bnot seen\b",
    r"\bno evidence of\b",
    r"\bno acute\b",
    r"\bdenies\b",
)

_NUMBER_RE = re.compile(r"\b(\d+(?:\.\d+)?)\s*(mm|cm|m|kg|g|cc|ml|%)?\b", re.IGNORECASE)
_SENTENCE_SPLIT_RE = re.compile(r"\n+|(?<=[.!?])\s+")
_NON_ALNUM_RE = re.compile(r"[^a-z0-9\s]+", re.IGNORECASE)
_MULTI_SPACE_RE = re.compile(r"\s+")
_HEADER_RE = re.compile(
    r"^\s*(impression|conclusion|conclusions|findings|report|answer|output)\s*:\s*",
    re.IGNORECASE,
)
_BULLET_RE = re.compile(r"^\s*(?:[-*•]|\d+[\.)])\s*")


@dataclass(slots=True)
class ScoreComponents:
    exact_match: float = 0.0
    token_precision: float = 0.0
    token_recall: float = 0.0
    token_f1: float = 0.0
    content_precision: float = 0.0
    content_recall: float = 0.0
    content_f1: float = 0.0
    jaccard: float = 0.0
    rouge1_recall: float = 0.0
    rouge2_recall: float = 0.0
    rouge_l_f1: float = 0.0
    sentence_coverage: float = 0.0
    negation_consistency: float = 0.0
    number_consistency: float = 0.0
    length_similarity: float = 0.0
    validator_score: float = 1.0
    composite: float = 0.0

    def to_dict(self) -> dict[str, float]:
        payload = asdict(self)
        return {k: round(float(v), 4) for k, v in payload.items()}


@dataclass(slots=True)
class CaseScore:
    case_index: int
    case_id: str
    modality: str
    template_id: str
    source: str
    findings: str
    reference_text: str
    predicted_text: str
    metrics: ScoreComponents
    validator: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    passed: bool = False
    created_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_index": self.case_index,
            "case_id": self.case_id,
            "modality": self.modality,
            "template_id": self.template_id,
            "source": self.source,
            "findings": self.findings,
            "reference_text": self.reference_text,
            "predicted_text": self.predicted_text,
            "metrics": self.metrics.to_dict(),
            "validator": self.validator,
            "notes": self.notes,
            "passed": self.passed,
            "created_at": self.created_at or _utc_now(),
        }


@dataclass(slots=True)
class BenchmarkAggregate:
    count: int = 0
    passed: int = 0
    failed: int = 0
    average_exact_match: float = 0.0
    average_token_f1: float = 0.0
    average_content_f1: float = 0.0
    average_jaccard: float = 0.0
    average_rouge_l_f1: float = 0.0
    average_sentence_coverage: float = 0.0
    average_negation_consistency: float = 0.0
    average_number_consistency: float = 0.0
    average_length_similarity: float = 0.0
    average_validator_score: float = 0.0
    average_composite: float = 0.0
    pass_rate: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        return {
            k: round(float(v), 4) if isinstance(v, float) else v
            for k, v in payload.items()
        }


# ---------------------------------------------------------------------------
# Public single-case scoring
# ---------------------------------------------------------------------------

def score_text_pair(
    prediction: str,
    reference: str,
    *,
    findings_text: str = "",
    modality: str = "",
    template_id: str = "",
    source: str = "benchmark",
    case_id: str = "",
    case_index: int = 0,
    pass_threshold: float = 0.72,
) -> dict[str, Any]:
    """
    Score one predicted text against one reference text.
    """
    predicted_text = _clean_text(prediction, MAX_TEXT_CHARS)
    reference_text = _clean_text(reference, MAX_TEXT_CHARS)
    findings = _clean_text(findings_text, MAX_FINDINGS_CHARS)

    if not reference_text:
        return {
            "ok": False,
            "error": "Reference text is empty.",
            "case_index": case_index,
            "case_id": str(case_id or ""),
        }

    if not predicted_text:
        return {
            "ok": False,
            "error": "Predicted text is empty.",
            "case_index": case_index,
            "case_id": str(case_id or ""),
        }

    metrics, validator_payload, notes = _score_components(
        predicted_text=predicted_text,
        reference_text=reference_text,
        findings_text=findings,
        template_id=template_id,
    )

    passed = metrics.composite >= float(pass_threshold)

    item = CaseScore(
        case_index=int(case_index),
        case_id=str(case_id or ""),
        modality=str(modality or "unknown"),
        template_id=str(template_id or ""),
        source=str(source or "benchmark"),
        findings=findings,
        reference_text=reference_text,
        predicted_text=predicted_text,
        metrics=metrics,
        validator=validator_payload,
        notes=notes,
        passed=passed,
        created_at=_utc_now(),
    )

    return {"ok": True, "item": item.to_dict()}


def score_case(
    case: Mapping[str, Any],
    *,
    prediction: str | None = None,
    case_index: int = 0,
    pass_threshold: float = 0.72,
) -> dict[str, Any]:
    """
    Score a benchmark case mapping.

    Supported case shapes include fields such as:
    - findings / input_findings
    - impression / reference_impression / gold_impression / target_impression
    - predicted_text / prediction / generated_impression / generated_text
    - modality, template_id, case_id, source
    """
    if not isinstance(case, Mapping):
        return {"ok": False, "error": "Case must be a mapping."}

    findings = _first_text(
        case,
        ("findings", "input_findings", "report_findings", "report_findings_text"),
    )
    reference = _first_text(
        case,
        (
            "reference_impression",
            "gold_impression",
            "target_impression",
            "expected_impression",
            "impression",
            "label",
            "answer",
            "target",
            "reference",
        ),
    )
    predicted = _clean_text(
        prediction if prediction is not None else _first_text(
            case,
            (
                "predicted_text",
                "prediction",
                "generated_impression",
                "generated_text",
                "output",
                "candidate",
                "response",
                "result",
            ),
        ),
        MAX_TEXT_CHARS,
    )

    if not reference:
        return {
            "ok": False,
            "error": "Case does not contain a usable reference impression.",
            "case_index": int(case_index),
            "case_id": _clean_text(case.get("case_id", ""), 120),
        }

    if not predicted:
        return {
            "ok": False,
            "error": "Case does not contain a usable prediction.",
            "case_index": int(case_index),
            "case_id": _clean_text(case.get("case_id", ""), 120),
        }

    return score_text_pair(
        predicted,
        reference,
        findings_text=findings,
        modality=_clean_text(case.get("modality", "unknown"), 80) or "unknown",
        template_id=_clean_text(case.get("template_id", ""), 80),
        source=_clean_text(case.get("source", "benchmark"), 120) or "benchmark",
        case_id=_clean_text(case.get("case_id", ""), 120),
        case_index=int(case_index),
        pass_threshold=pass_threshold,
    )


# ---------------------------------------------------------------------------
# Public dataset scoring
# ---------------------------------------------------------------------------

PredictorFn = Callable[[Mapping[str, Any]], str | Mapping[str, Any]]


def score_cases(
    cases: Sequence[Mapping[str, Any]],
    *,
    predictions: Sequence[str | Mapping[str, Any]] | Mapping[str, str | Mapping[str, Any]] | None = None,
    predictor: PredictorFn | None = None,
    pass_threshold: float = 0.72,
    max_cases: int = MAX_CASES_PER_RUN,
    strict_count: bool = False,
) -> dict[str, Any]:
    """
    Score many cases.

    Supported modes:
    1. cases already contain their own prediction field(s)
    2. aligned `predictions` sequence
    3. `predictions` mapping keyed by case_id
    4. `predictor(case)` callback returning a string or mapping

    Returns:
    {
      ok: true,
      count: ...,
      items: [...],
      aggregate: {...},
      errors: [...]
    }
    """
    if not isinstance(cases, Sequence):
        return {"ok": False, "error": "Cases must be a sequence."}

    safe_cases = list(cases[: max(1, int(max_cases))])
    results: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []

    if strict_count and predictions is not None and isinstance(predictions, Sequence) and not isinstance(predictions, (str, bytes, bytearray)):
        if len(predictions) != len(safe_cases):
            return {
                "ok": False,
                "error": "Prediction count does not match case count.",
                "case_count": len(safe_cases),
                "prediction_count": len(predictions),
            }

    for idx, case in enumerate(safe_cases):
        resolved_prediction = _resolve_prediction(
            case=case,
            case_index=idx,
            predictions=predictions,
            predictor=predictor,
        )

        if isinstance(resolved_prediction, Mapping):
            candidate_text = _first_text(
                resolved_prediction,
                ("predicted_text", "prediction", "generated_impression", "generated_text", "output", "result", "response"),
            )
        else:
            candidate_text = _clean_text(resolved_prediction, MAX_TEXT_CHARS)

        scored = score_case(
            case,
            prediction=candidate_text if candidate_text else None,
            case_index=idx,
            pass_threshold=pass_threshold,
        )

        if scored.get("ok"):
            results.append(scored["item"])
        else:
            errors.append(
                {
                    "case_index": idx,
                    "case_id": _clean_text(case.get("case_id", ""), 120),
                    "error": str(scored.get("error", "Unknown scoring error")),
                }
            )

    aggregate = summarize_case_scores(results, pass_threshold=pass_threshold)
    return {
        "ok": True,
        "count": len(results),
        "items": results,
        "aggregate": aggregate,
        "errors": errors,
        "pass_threshold": round(float(pass_threshold), 4),
        "created_at": _utc_now(),
    }


def summarize_case_scores(
    items: Sequence[Mapping[str, Any]],
    *,
    pass_threshold: float = 0.72,
) -> dict[str, Any]:
    """
    Aggregate a list of per-case score payloads.
    """
    if not items:
        return BenchmarkAggregate().to_dict()

    def _metric(name: str) -> float:
        values = []
        for item in items:
            metrics = item.get("metrics", {})
            try:
                values.append(float(metrics.get(name, 0.0)))
            except Exception:
                values.append(0.0)
        return mean(values) if values else 0.0

    passed = 0
    for item in items:
        metrics = item.get("metrics", {})
        try:
            composite = float(metrics.get("composite", 0.0))
        except Exception:
            composite = 0.0
        if composite >= pass_threshold:
            passed += 1

    aggregate = BenchmarkAggregate(
        count=len(items),
        passed=passed,
        failed=max(0, len(items) - passed),
        average_exact_match=_metric("exact_match"),
        average_token_f1=_metric("token_f1"),
        average_content_f1=_metric("content_f1"),
        average_jaccard=_metric("jaccard"),
        average_rouge_l_f1=_metric("rouge_l_f1"),
        average_sentence_coverage=_metric("sentence_coverage"),
        average_negation_consistency=_metric("negation_consistency"),
        average_number_consistency=_metric("number_consistency"),
        average_length_similarity=_metric("length_similarity"),
        average_validator_score=_metric("validator_score"),
        average_composite=_metric("composite"),
        pass_rate=(passed / len(items)) if items else 0.0,
    )
    return aggregate.to_dict()


# ---------------------------------------------------------------------------
# Core metric computation
# ---------------------------------------------------------------------------

def _score_components(
    *,
    predicted_text: str,
    reference_text: str,
    findings_text: str,
    template_id: str,
) -> tuple[ScoreComponents, dict[str, Any], list[str]]:
    pred_norm = normalize_text(predicted_text)
    ref_norm = normalize_text(reference_text)

    pred_tokens = tokenize(pred_norm)
    ref_tokens = tokenize(ref_norm)

    pred_content = content_tokens(pred_norm)
    ref_content = content_tokens(ref_norm)

    token_precision, token_recall, token_f1 = _counter_f1(pred_tokens, ref_tokens)
    content_precision, content_recall, content_f1 = _counter_f1(pred_content, ref_content)

    jaccard = _jaccard(set(pred_tokens), set(ref_tokens))
    rouge1_recall = _ngram_recall(pred_tokens, ref_tokens, n=1)
    rouge2_recall = _ngram_recall(pred_tokens, ref_tokens, n=2)
    rouge_l_f1 = _rouge_l_f1(pred_tokens, ref_tokens)
    sentence_coverage = _sentence_coverage(predicted_text, reference_text)
    negation_consistency = _negation_consistency(predicted_text, reference_text)
    number_consistency = _number_consistency(predicted_text, reference_text)
    length_similarity = _length_similarity(predicted_text, reference_text)
    exact_match = 1.0 if pred_norm == ref_norm else 0.0

    validator_payload = _run_validator(
        predicted_text=predicted_text,
        findings_text=findings_text,
        template_id=template_id,
    )
    validator_score = float(validator_payload.get("score", 1.0))
    validator_error_count = int(validator_payload.get("error_count", 0))
    validator_warning_count = int(validator_payload.get("warning_count", 0))

    # Weighted composite tuned for local benchmarking:
    # lexical/core overlap matters most, but structural correctness matters too.
    composite = (
        0.22 * token_f1
        + 0.16 * content_f1
        + 0.16 * rouge_l_f1
        + 0.10 * rouge1_recall
        + 0.08 * sentence_coverage
        + 0.08 * number_consistency
        + 0.07 * negation_consistency
        + 0.05 * length_similarity
        + 0.08 * validator_score
    )
    if exact_match >= 1.0:
        composite = min(1.0, composite + 0.05)

    # Penalize validator hard errors.
    if validator_error_count > 0:
        composite = max(0.0, composite - min(0.25, 0.08 * validator_error_count))
    elif validator_warning_count > 0:
        composite = max(0.0, composite - min(0.08, 0.02 * validator_warning_count))

    metrics = ScoreComponents(
        exact_match=exact_match,
        token_precision=token_precision,
        token_recall=token_recall,
        token_f1=token_f1,
        content_precision=content_precision,
        content_recall=content_recall,
        content_f1=content_f1,
        jaccard=jaccard,
        rouge1_recall=rouge1_recall,
        rouge2_recall=rouge2_recall,
        rouge_l_f1=rouge_l_f1,
        sentence_coverage=sentence_coverage,
        negation_consistency=negation_consistency,
        number_consistency=number_consistency,
        length_similarity=length_similarity,
        validator_score=validator_score,
        composite=max(0.0, min(1.0, composite)),
    )

    notes: list[str] = []
    if exact_match >= 1.0:
        notes.append("Exact normalized text match.")
    if number_consistency < 0.8:
        notes.append("Numeric findings differ from the reference.")
    if negation_consistency < 0.8:
        notes.append("Negation pattern differs from the reference.")
    if validator_error_count > 0:
        notes.append(f"Validator reported {validator_error_count} error(s).")
    elif validator_warning_count > 0:
        notes.append(f"Validator reported {validator_warning_count} warning(s).")

    return metrics, validator_payload, notes


# ---------------------------------------------------------------------------
# Normalization / tokenization
# ---------------------------------------------------------------------------

def normalize_text(text: str) -> str:
    """
    Normalize text for fair comparison.
    """
    text = _clean_text(text, MAX_TEXT_CHARS)
    if not text:
        return ""

    lines = []
    for raw_line in text.splitlines():
        line = _HEADER_RE.sub("", raw_line.strip())
        line = _BULLET_RE.sub("", line)
        if line:
            lines.append(line)

    text = "\n".join(lines).strip().lower()
    text = _NON_ALNUM_RE.sub(" ", text)
    text = _MULTI_SPACE_RE.sub(" ", text)
    return text.strip()


def tokenize(text: str) -> list[str]:
    normalized = normalize_text(text)
    if not normalized:
        return []
    return [tok for tok in normalized.split(" ") if tok]


def content_tokens(text: str) -> list[str]:
    return [tok for tok in tokenize(text) if tok not in _STOPWORDS and len(tok) > 1]


def split_sentences(text: str) -> list[str]:
    cleaned = _clean_text(text, MAX_TEXT_CHARS)
    if not cleaned:
        return []
    parts = [_BULLET_RE.sub("", part.strip()) for part in _SENTENCE_SPLIT_RE.split(cleaned)]
    return [part for part in parts if part]


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def _counter_f1(a_tokens: Sequence[str], b_tokens: Sequence[str]) -> tuple[float, float, float]:
    if not a_tokens and not b_tokens:
        return 1.0, 1.0, 1.0
    if not a_tokens or not b_tokens:
        return 0.0, 0.0, 0.0

    a_counts = Counter(a_tokens)
    b_counts = Counter(b_tokens)
    overlap = sum(min(a_counts[t], b_counts[t]) for t in (a_counts.keys() | b_counts.keys()))

    precision = overlap / max(1, sum(a_counts.values()))
    recall = overlap / max(1, sum(b_counts.values()))
    f1 = _harmonic_mean(precision, recall)
    return precision, recall, f1


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    union = a | b
    if not union:
        return 1.0
    return len(a & b) / len(union)


def _ngrams(tokens: Sequence[str], n: int) -> list[tuple[str, ...]]:
    if n <= 0 or len(tokens) < n:
        return []
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def _ngram_recall(pred_tokens: Sequence[str], ref_tokens: Sequence[str], *, n: int) -> float:
    ref_ngrams = _ngrams(ref_tokens, n)
    pred_ngrams = _ngrams(pred_tokens, n)
    if not ref_ngrams and not pred_ngrams:
        return 1.0
    if not ref_ngrams or not pred_ngrams:
        return 0.0

    ref_counts = Counter(ref_ngrams)
    pred_counts = Counter(pred_ngrams)
    overlap = sum(min(ref_counts[g], pred_counts[g]) for g in ref_counts.keys())
    return overlap / max(1, sum(ref_counts.values()))


def _rouge_l_f1(pred_tokens: Sequence[str], ref_tokens: Sequence[str]) -> float:
    if not pred_tokens and not ref_tokens:
        return 1.0
    if not pred_tokens or not ref_tokens:
        return 0.0

    lcs = _lcs_length(pred_tokens, ref_tokens)
    precision = lcs / max(1, len(pred_tokens))
    recall = lcs / max(1, len(ref_tokens))
    return _harmonic_mean(precision, recall)


def _lcs_length(a: Sequence[str], b: Sequence[str]) -> int:
    if not a or not b:
        return 0
    # Memory-efficient dynamic programming.
    prev = [0] * (len(b) + 1)
    for token_a in a:
        curr = [0]
        for j, token_b in enumerate(b, start=1):
            if token_a == token_b:
                curr.append(prev[j - 1] + 1)
            else:
                curr.append(max(curr[-1], prev[j]))
        prev = curr
    return prev[-1]


def _sentence_coverage(predicted_text: str, reference_text: str) -> float:
    ref_sentences = split_sentences(reference_text)
    pred_sentences = split_sentences(predicted_text)

    if not ref_sentences and not pred_sentences:
        return 1.0
    if not ref_sentences or not pred_sentences:
        return 0.0

    pred_token_sets = [set(content_tokens(s)) or set(tokenize(s)) for s in pred_sentences]
    scores: list[float] = []

    for ref_sentence in ref_sentences:
        ref_set = set(content_tokens(ref_sentence)) or set(tokenize(ref_sentence))
        if not ref_set:
            continue
        best = 0.0
        for pred_set in pred_token_sets:
            best = max(best, _jaccard(ref_set, pred_set))
        scores.append(best)

    return mean(scores) if scores else 0.0


def _negation_consistency(predicted_text: str, reference_text: str) -> float:
    pred_flags = _extract_negation_flags(predicted_text)
    ref_flags = _extract_negation_flags(reference_text)

    if not pred_flags and not ref_flags:
        return 1.0
    if not pred_flags or not ref_flags:
        return 0.0

    keys = sorted(set(pred_flags) | set(ref_flags))
    matches = 0
    for key in keys:
        if pred_flags.get(key, False) == ref_flags.get(key, False):
            matches += 1
    return matches / max(1, len(keys))


def _extract_negation_flags(text: str) -> dict[str, bool]:
    lines = split_sentences(text)
    flags: dict[str, bool] = {}
    for idx, line in enumerate(lines):
        line_key = f"line_{idx}"
        lowered = line.lower()
        flags[line_key] = any(re.search(pattern, lowered, flags=re.IGNORECASE) for pattern in _NEGATION_PATTERNS)
    return flags


def _number_consistency(predicted_text: str, reference_text: str) -> float:
    pred_numbers = _extract_number_units(predicted_text)
    ref_numbers = _extract_number_units(reference_text)

    if not pred_numbers and not ref_numbers:
        return 1.0
    if not ref_numbers:
        return 1.0
    if not pred_numbers:
        return 0.0

    ref_counts = Counter(ref_numbers)
    pred_counts = Counter(pred_numbers)
    overlap = sum(min(ref_counts[k], pred_counts[k]) for k in ref_counts.keys())
    recall = overlap / max(1, sum(ref_counts.values()))

    # Also reward approximate numeric closeness when exact unit tuples differ.
    if recall >= 1.0:
        return 1.0

    fuzzy = _fuzzy_number_overlap(pred_numbers, ref_numbers)
    return max(recall, fuzzy)


def _extract_number_units(text: str) -> list[tuple[float, str]]:
    items: list[tuple[float, str]] = []
    for value, unit in _NUMBER_RE.findall(_clean_text(text, MAX_TEXT_CHARS)):
        try:
            number = float(value)
        except Exception:
            continue
        normalized_unit = (unit or "").strip().lower()
        items.append((number, normalized_unit))
    return items


def _fuzzy_number_overlap(
    pred_numbers: Sequence[tuple[float, str]],
    ref_numbers: Sequence[tuple[float, str]],
) -> float:
    if not pred_numbers or not ref_numbers:
        return 0.0

    used: set[int] = set()
    matches = 0
    for ref_value, ref_unit in ref_numbers:
        best_idx = None
        best_delta = None
        for idx, (pred_value, pred_unit) in enumerate(pred_numbers):
            if idx in used:
                continue
            if ref_unit and pred_unit and ref_unit != pred_unit:
                continue
            tolerance = max(0.1, 0.05 * max(abs(ref_value), 1.0))
            delta = abs(pred_value - ref_value)
            if delta <= tolerance and (best_delta is None or delta < best_delta):
                best_delta = delta
                best_idx = idx
        if best_idx is not None:
            used.add(best_idx)
            matches += 1

    return matches / max(1, len(ref_numbers))


def _length_similarity(predicted_text: str, reference_text: str) -> float:
    pred_len = max(1, len(tokenize(predicted_text)))
    ref_len = max(1, len(tokenize(reference_text)))
    ratio = min(pred_len, ref_len) / max(pred_len, ref_len)
    # Gentle penalty for massive length mismatch.
    return max(0.0, min(1.0, ratio))


def _harmonic_mean(a: float, b: float) -> float:
    if a <= 0.0 or b <= 0.0:
        return 0.0
    return 2.0 * a * b / (a + b)


def _run_validator(
    *,
    predicted_text: str,
    findings_text: str,
    template_id: str,
) -> dict[str, Any]:
    if validate_impression is None:
        return {
            "available": False,
            "score": 1.0,
            "valid": True,
            "error_count": 0,
            "warning_count": 0,
            "issues": [],
        }

    try:
        result = validate_impression(
            predicted_text,
            findings_text=findings_text,
            template_id=template_id or None,
        )
    except Exception as exc:
        return {
            "available": True,
            "score": 0.75,
            "valid": False,
            "error_count": 1,
            "warning_count": 0,
            "issues": [
                {
                    "code": "validator_exception",
                    "message": f"{type(exc).__name__}: {exc}",
                    "severity": "error",
                }
            ],
        }

    # Compatible with either a dataclass result or plain dict.
    if hasattr(result, "to_dict"):
        payload = result.to_dict()
    elif isinstance(result, Mapping):
        payload = dict(result)
    else:
        payload = {"valid": True, "score": 1.0, "issues": []}

    issues = payload.get("issues", [])
    if not isinstance(issues, list):
        issues = []

    error_count = 0
    warning_count = 0
    for issue in issues:
        if not isinstance(issue, Mapping):
            continue
        severity = str(issue.get("severity", "")).lower()
        if severity == "error":
            error_count += 1
        elif severity == "warning":
            warning_count += 1

    return {
        "available": True,
        "valid": bool(payload.get("valid", True)),
        "score": float(payload.get("score", 1.0)),
        "error_count": error_count,
        "warning_count": warning_count,
        "issues": issues,
        "normalized_impression": payload.get("normalized_impression", ""),
        "lines": payload.get("lines", []),
    }


# ---------------------------------------------------------------------------
# Prediction resolution
# ---------------------------------------------------------------------------

def _resolve_prediction(
    *,
    case: Mapping[str, Any],
    case_index: int,
    predictions: Sequence[str | Mapping[str, Any]] | Mapping[str, str | Mapping[str, Any]] | None,
    predictor: PredictorFn | None,
) -> str | Mapping[str, Any]:
    if predictor is not None:
        try:
            result = predictor(case)
            if isinstance(result, Mapping):
                return result
            return _clean_text(result, MAX_TEXT_CHARS)
        except Exception as exc:
            return {"prediction": "", "error": f"{type(exc).__name__}: {exc}"}

    if predictions is None:
        return _first_text(
            case,
            ("predicted_text", "prediction", "generated_impression", "generated_text", "output", "result", "response"),
        )

    if isinstance(predictions, Mapping):
        case_id = _clean_text(case.get("case_id", ""), 120)
        if case_id and case_id in predictions:
            return predictions[case_id]
        return ""

    if isinstance(predictions, Sequence) and not isinstance(predictions, (str, bytes, bytearray)):
        if 0 <= case_index < len(predictions):
            return predictions[case_index]
        return ""

    return ""


# ---------------------------------------------------------------------------
# Generic data extraction helpers
# ---------------------------------------------------------------------------

def _first_text(data: Mapping[str, Any], keys: Sequence[str]) -> str:
    for key in keys:
        if key not in data:
            continue
        value = _clean_text(data.get(key, ""), MAX_TEXT_CHARS)
        if value:
            return value
    return ""


def _clean_text(value: Any, max_chars: int) -> str:
    text = str(value or "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    text = text.strip()
    if len(text) > max_chars:
        text = text[:max_chars].rstrip()
    return text


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


# ---------------------------------------------------------------------------
# Compatibility aliases
# ---------------------------------------------------------------------------

score_prediction = score_text_pair
score_benchmark_case = score_case
score_benchmark_cases = score_cases
aggregate_scores = summarize_case_scores


__all__ = [
    "BenchmarkAggregate",
    "CaseScore",
    "MAX_CASES_PER_RUN",
    "ScoreComponents",
    "aggregate_scores",
    "content_tokens",
    "normalize_text",
    "score_benchmark_case",
    "score_benchmark_cases",
    "score_case",
    "score_cases",
    "score_prediction",
    "score_text_pair",
    "split_sentences",
    "summarize_case_scores",
    "tokenize",
]
