"""Guideline recommendation layer for RadCopilot.

This module provides a dedicated home for follow-up and recommendation logic so
that report generation does not need to keep guideline concerns mixed into the
main impression pipeline.

Design goals:
- provide deterministic, explainable baseline recommendations
- optionally use the Ollama client for narrative recommendation text
- avoid unsupported society / threshold claims when case context is weak
- remain usable even if only part of the refactor exists

Notes:
- This module does not attempt to be a comprehensive clinical guideline engine.
- Deterministic rules are intentionally conservative and should be reviewed by a
  clinician before clinical use.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any, Iterable, Mapping, Optional, Protocol, Sequence

from radcopilot.services.ollama_client import OllamaClient, OllamaClientError


class ConfigLike(Protocol):
    """Minimal config contract used by this module."""

    ollama_url: str


_DEFAULT_MODEL = "llama3.1:8b"
_MAX_FINDINGS_CHARS = 12000
_MAX_INDICATION_CHARS = 1000

_SIZE_RE = re.compile(r"\b(?P<num>\d+(?:\.\d+)?)\s*(?P<unit>mm|cm)\b", re.IGNORECASE)
_WORD_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)


@dataclass(slots=True)
class GuidelineContext:
    """Normalized case context for recommendation generation."""

    findings: str
    indication: str = ""
    template_id: str = ""
    template_label: str = ""
    guideline_hint: str = ""
    age: str = ""
    sex: str = ""
    model: str = _DEFAULT_MODEL
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "GuidelineContext":
        return cls(
            findings=_clean_text(payload.get("findings", ""))[:_MAX_FINDINGS_CHARS],
            indication=_clean_text(payload.get("indication", ""))[:_MAX_INDICATION_CHARS],
            template_id=str(payload.get("template_id", "") or "").strip().lower(),
            template_label=str(payload.get("template_label", "") or "").strip(),
            guideline_hint=_clean_text(payload.get("guideline_hint", ""))[:1000],
            age=str(payload.get("age", "") or "").strip(),
            sex=str(payload.get("sex", "") or "").strip(),
            model=str(payload.get("model", _DEFAULT_MODEL) or _DEFAULT_MODEL).strip(),
            metadata=dict(payload.get("metadata", {}) or {}),
        )


@dataclass(slots=True)
class GuidelineRecommendation:
    """Single recommendation line / block."""

    category: str
    title: str
    recommendation: str
    rationale: str = ""
    confidence: str = "moderate"
    source_hint: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "category": self.category,
            "title": self.title,
            "recommendation": self.recommendation,
            "rationale": self.rationale,
            "confidence": self.confidence,
            "source_hint": self.source_hint,
        }


@dataclass(slots=True)
class GuidelineResult:
    """Structured output from the guideline engine."""

    ok: bool
    context_summary: str
    recommendations: list[GuidelineRecommendation] = field(default_factory=list)
    narrative: str = ""
    warnings: list[str] = field(default_factory=list)
    trace: list[str] = field(default_factory=list)
    model_used: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "context_summary": self.context_summary,
            "recommendations": [item.to_dict() for item in self.recommendations],
            "narrative": self.narrative,
            "warnings": self.warnings,
            "trace": self.trace,
            "model_used": self.model_used,
        }


def generate_guideline_text(
    *,
    config: ConfigLike | None,
    context: GuidelineContext | Mapping[str, Any],
    use_model: bool = True,
) -> GuidelineResult:
    """Generate guideline-oriented recommendation text.

    Behavior:
    1. Normalize case context.
    2. Produce deterministic recommendation candidates.
    3. Optionally ask Ollama to rewrite them into concise reviewable text.
    4. Fall back to deterministic narrative if model generation fails.
    """
    ctx = context if isinstance(context, GuidelineContext) else GuidelineContext.from_mapping(context)
    trace: list[str] = []
    warnings: list[str] = []

    recommendations = recommend_guidance(ctx, trace=trace, warnings=warnings)
    context_summary = summarize_context(ctx)
    narrative = render_recommendations(recommendations)
    model_used = ""

    if use_model:
        client = _build_client(config=config, model=ctx.model)
        if client is None:
            trace.append("guidelines:model_unavailable")
        else:
            prompt = build_guideline_prompt(ctx, recommendations)
            try:
                response_text = client.chat_text(prompt, model=client.model)
                cleaned = _clean_guideline_output(response_text)
                if cleaned:
                    narrative = cleaned
                    model_used = client.model
                    trace.append("guidelines:model_success")
                else:
                    trace.append("guidelines:model_empty")
                    warnings.append("Guideline model response was empty; using deterministic recommendation text.")
            except OllamaClientError as exc:
                trace.append("guidelines:model_error")
                warnings.append(f"Guideline model call failed: {exc}. Using deterministic recommendation text.")

    if not recommendations:
        warnings.append("No strong guideline-specific trigger was detected. Returned generic review language.")
        recommendations = [
            GuidelineRecommendation(
                category="general",
                title="Clinical review",
                recommendation=(
                    "No confident rule-specific follow-up trigger was identified from the supplied findings. "
                    "Consider correlating with the full clinical context and the finalized report."
                ),
                rationale="The findings did not clearly match a narrow guideline pattern.",
                confidence="low",
                source_hint="Case-specific radiologist review",
            )
        ]
        if not narrative.strip():
            narrative = render_recommendations(recommendations)

    return GuidelineResult(
        ok=True,
        context_summary=context_summary,
        recommendations=recommendations,
        narrative=narrative,
        warnings=warnings,
        trace=trace,
        model_used=model_used,
    )


def recommend_guidance(
    context: GuidelineContext | Mapping[str, Any],
    *,
    trace: Optional[list[str]] = None,
    warnings: Optional[list[str]] = None,
) -> list[GuidelineRecommendation]:
    """Return deterministic guideline candidates from the case context."""
    ctx = context if isinstance(context, GuidelineContext) else GuidelineContext.from_mapping(context)
    trace = trace if trace is not None else []
    warnings = warnings if warnings is not None else []

    findings_lower = ctx.findings.lower()
    template_lower = (ctx.template_id or ctx.template_label).lower()
    recs: list[GuidelineRecommendation] = []

    lung_nodule_size_mm = _largest_size_mm(ctx.findings, organ="lung")
    adrenal_size_mm = _largest_size_mm(ctx.findings, organ="adrenal")
    thyroid_size_mm = _largest_size_mm(ctx.findings, organ="thyroid")
    renal_context = _contains_any(findings_lower, ["renal cyst", "kidney cyst", "bosniak", "complex cyst"])

    pulmonary_context = _contains_any(findings_lower, ["lung", "pulmonary", "subpleural", "nodule", "nodular"]) or "chest" in template_lower
    adrenal_context = _contains_any(findings_lower, ["adrenal", "adrenals"])
    thyroid_context = _contains_any(findings_lower, ["thyroid"]) or "thyroid" in template_lower

    if pulmonary_context and _contains_any(findings_lower, ["nodule", "nodular"]):
        trace.append("rule:pulmonary_nodule")
        recs.append(_pulmonary_nodule_recommendation(ctx, lung_nodule_size_mm))

    if adrenal_context and _contains_any(findings_lower, ["mass", "nodule", "lesion", "adenoma", "incidentaloma"]):
        trace.append("rule:adrenal_incidentaloma")
        recs.append(_adrenal_recommendation(ctx, adrenal_size_mm))

    if renal_context:
        trace.append("rule:renal_cyst")
        recs.append(_renal_cyst_recommendation(ctx))

    if thyroid_context and _contains_any(findings_lower, ["nodule", "lesion"]):
        trace.append("rule:thyroid_nodule")
        recs.append(_thyroid_recommendation(ctx, thyroid_size_mm))

    if _contains_any(findings_lower, ["aneurysm", "ectasia", "aortic dilation", "aortic dilatation"]):
        trace.append("rule:vascular_followup")
        recs.append(_vascular_recommendation(ctx))

    if _contains_any(findings_lower, ["lymph node", "adenopathy", "lymphadenopathy"]):
        trace.append("rule:nodal_followup")
        recs.append(_nodal_recommendation(ctx))

    if not recs and ctx.guideline_hint:
        trace.append("rule:hint_only")
        recs.append(
            GuidelineRecommendation(
                category="template-hint",
                title="Template guidance",
                recommendation=ctx.guideline_hint,
                rationale="No stronger deterministic rule matched; returning template-specific guidance hint.",
                confidence="low",
                source_hint="Template guidance",
            )
        )

    return _dedupe_recommendations(recs)


def build_guideline_prompt(
    context: GuidelineContext | Mapping[str, Any],
    recommendations: Sequence[GuidelineRecommendation] | None = None,
) -> str:
    """Build a tightly scoped prompt for LLM recommendation polishing."""
    ctx = context if isinstance(context, GuidelineContext) else GuidelineContext.from_mapping(context)
    recs = list(recommendations or recommend_guidance(ctx))
    candidate_block = render_recommendations(recs, bullets=True)
    patient_bits = []
    if ctx.age:
        patient_bits.append(f"Age: {ctx.age}")
    if ctx.sex:
        patient_bits.append(f"Sex: {ctx.sex}")
    patient_block = " | ".join(patient_bits) if patient_bits else "Not provided"

    return (
        "You are summarizing guideline-oriented follow-up or recommendation logic for a radiology case.\n"
        "Requirements:\n"
        "- Be concise and clinically reviewable.\n"
        "- Do not invent society names, thresholds, or management pathways that are unsupported by the findings.\n"
        "- If the case context is weak or incomplete, say so plainly.\n"
        "- Prefer short paragraph or bullet-style recommendation language suitable for radiologist review.\n"
        "- Do not restate the entire report.\n\n"
        f"Template: {ctx.template_label or ctx.template_id or 'Not provided'}\n"
        f"Patient: {patient_block}\n"
        f"Indication: {ctx.indication or 'Not provided'}\n"
        f"Template guidance: {ctx.guideline_hint or 'None provided'}\n\n"
        f"Findings:\n{ctx.findings or 'Not provided'}\n\n"
        f"Deterministic recommendation candidates:\n{candidate_block or '- None'}\n\n"
        "Return concise recommendation language only."
    )


def render_recommendations(
    recommendations: Sequence[GuidelineRecommendation],
    *,
    bullets: bool = False,
) -> str:
    """Render recommendations into plain text."""
    items = [item for item in recommendations if item.recommendation.strip()]
    if not items:
        return ""

    lines: list[str] = []
    for idx, item in enumerate(items, start=1):
        prefix = "-" if bullets else f"{idx}."
        line = f"{prefix} {item.recommendation.strip()}"
        if item.source_hint:
            line += f" [{item.source_hint}]"
        lines.append(line)
        if item.rationale and bullets:
            lines.append(f"  rationale: {item.rationale.strip()}")
    return "\n".join(lines).strip()


def summarize_context(context: GuidelineContext | Mapping[str, Any]) -> str:
    """Return a compact summary of the case context."""
    ctx = context if isinstance(context, GuidelineContext) else GuidelineContext.from_mapping(context)
    bits = []
    if ctx.template_label:
        bits.append(ctx.template_label)
    elif ctx.template_id:
        bits.append(ctx.template_id)
    if ctx.age:
        bits.append(f"Age {ctx.age}")
    if ctx.sex:
        bits.append(ctx.sex)
    if ctx.indication:
        bits.append(f"Indication: {ctx.indication}")
    if not bits:
        bits.append("Radiology case context")
    return " | ".join(bits)


# ---------------------------------------------------------------------------
# Deterministic recommendation rules
# ---------------------------------------------------------------------------

def _pulmonary_nodule_recommendation(ctx: GuidelineContext, size_mm: Optional[float]) -> GuidelineRecommendation:
    if size_mm is None:
        return GuidelineRecommendation(
            category="pulmonary-nodule",
            title="Pulmonary nodule follow-up",
            recommendation=(
                "Pulmonary nodule follow-up may be relevant if the described nodule is truly incidental and not already characterized. "
                "Confirm the exact nodule size, number, and patient risk context before applying a specific interval."
            ),
            rationale="Pulmonary nodule language is present, but a reliable size threshold could not be extracted.",
            confidence="moderate",
            source_hint="Pulmonary nodule follow-up guidance",
        )

    if size_mm < 6:
        rec = (
            f"Small pulmonary nodule measuring approximately {size_mm:g} mm. "
            "If this is an incidental solid pulmonary nodule, follow-up may be limited or not required depending on risk context; correlate with patient risk and prior imaging."
        )
    elif size_mm < 8:
        rec = (
            f"Pulmonary nodule measuring approximately {size_mm:g} mm. "
            "Consider interval chest CT follow-up if this represents an incidental solid pulmonary nodule, with timing based on risk context and prior studies."
        )
    else:
        rec = (
            f"Pulmonary nodule measuring approximately {size_mm:g} mm. "
            "More intensive evaluation or shorter-interval follow-up may be warranted depending on morphology, prior imaging, and risk context."
        )

    return GuidelineRecommendation(
        category="pulmonary-nodule",
        title="Pulmonary nodule follow-up",
        recommendation=rec,
        rationale="Lung nodule terminology and a measurable lesion were identified in the findings.",
        confidence="moderate",
        source_hint="Pulmonary nodule follow-up guidance",
    )


def _adrenal_recommendation(ctx: GuidelineContext, size_mm: Optional[float]) -> GuidelineRecommendation:
    if size_mm is None:
        rec = (
            "Adrenal lesion follow-up may depend on lesion size and characterization. "
            "Correlate with prior imaging and consider dedicated adrenal imaging only if the lesion is indeterminate."
        )
    elif size_mm < 10:
        rec = (
            f"Small adrenal lesion measuring approximately {size_mm:g} mm. "
            "If benign imaging features are present, additional follow-up may be unnecessary; otherwise correlate with prior studies and characterization."
        )
    else:
        rec = (
            f"Adrenal lesion measuring approximately {size_mm:g} mm. "
            "If indeterminate on this exam, consider dedicated adrenal protocol characterization or interval imaging based on clinical context and prior studies."
        )

    return GuidelineRecommendation(
        category="adrenal",
        title="Adrenal incidental finding",
        recommendation=rec,
        rationale="Adrenal lesion terminology is present in the findings.",
        confidence="moderate",
        source_hint="Adrenal incidental lesion review",
    )


def _renal_cyst_recommendation(ctx: GuidelineContext) -> GuidelineRecommendation:
    has_bosniak = "bosniak" in ctx.findings.lower()
    if has_bosniak:
        rec = (
            "A Bosniak descriptor is referenced in the findings. "
            "Management should follow the assigned Bosniak category and the finalized radiologist interpretation."
        )
        rationale = "The findings explicitly mention Bosniak terminology."
    else:
        rec = (
            "If the renal cyst is complex or indeterminate, additional characterization or interval imaging may be appropriate. "
            "Simple cysts usually do not require dedicated follow-up."
        )
        rationale = "Renal cyst context is present, but no explicit Bosniak class was extracted."

    return GuidelineRecommendation(
        category="renal-cyst",
        title="Renal cyst / Bosniak context",
        recommendation=rec,
        rationale=rationale,
        confidence="moderate",
        source_hint="Renal cyst characterization guidance",
    )


def _thyroid_recommendation(ctx: GuidelineContext, size_mm: Optional[float]) -> GuidelineRecommendation:
    if size_mm is None:
        rec = (
            "Thyroid nodule follow-up depends on sonographic features and size thresholds. "
            "If this was an incidental non-thyroid exam finding, consider dedicated thyroid ultrasound only when clinically appropriate."
        )
    else:
        rec = (
            f"Thyroid lesion measuring approximately {size_mm:g} mm. "
            "Further thyroid ultrasound evaluation may be appropriate depending on the exam context and the lesion description."
        )
    return GuidelineRecommendation(
        category="thyroid",
        title="Thyroid nodule follow-up",
        recommendation=rec,
        rationale="Thyroid lesion terminology is present in the findings.",
        confidence="low",
        source_hint="Incidental thyroid lesion review",
    )


def _vascular_recommendation(ctx: GuidelineContext) -> GuidelineRecommendation:
    return GuidelineRecommendation(
        category="vascular",
        title="Vascular follow-up",
        recommendation=(
            "Vascular follow-up should be based on the exact vessel involved, the measured diameter, symptoms, and comparison with prior imaging. "
            "If aneurysmal dilation is new or enlarging, closer surveillance may be warranted."
        ),
        rationale="Aneurysm or vascular dilation terminology is present in the findings.",
        confidence="low",
        source_hint="Vascular surveillance review",
    )


def _nodal_recommendation(ctx: GuidelineContext) -> GuidelineRecommendation:
    return GuidelineRecommendation(
        category="nodes",
        title="Nodal follow-up",
        recommendation=(
            "Lymph node follow-up depends on nodal station, short-axis size, morphology, clinical history, and prior imaging. "
            "Correlate with oncologic or infectious context before recommending interval imaging."
        ),
        rationale="Nodal terminology is present, but no single narrow management rule applies from the extracted text alone.",
        confidence="low",
        source_hint="Context-dependent nodal assessment",
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_client(*, config: ConfigLike | None, model: str) -> OllamaClient | None:
    base_url = getattr(config, "ollama_url", None) if config is not None else None
    try:
        return OllamaClient(base_url=base_url or "http://localhost:11434", model=model or _DEFAULT_MODEL)
    except Exception:
        return None


def _largest_size_mm(text: str, *, organ: str = "") -> Optional[float]:
    if not text:
        return None
    text_lower = text.lower()
    sizes: list[float] = []
    for match in _SIZE_RE.finditer(text):
        value = float(match.group("num"))
        unit = match.group("unit").lower()
        mm = value * 10.0 if unit == "cm" else value
        if organ:
            window_start = max(0, match.start() - 60)
            window_end = min(len(text_lower), match.end() + 60)
            window = text_lower[window_start:window_end]
            if organ == "lung" and not _contains_any(window, ["lung", "pulmonary", "nodule", "subpleural", "fissural"]):
                continue
            if organ == "adrenal" and "adrenal" not in window:
                continue
            if organ == "thyroid" and "thyroid" not in window:
                continue
        sizes.append(mm)
    return max(sizes) if sizes else None


def _contains_any(text: str, terms: Iterable[str]) -> bool:
    base = text.lower()
    return any(term.lower() in base for term in terms)


def _clean_guideline_output(text: str) -> str:
    cleaned = _clean_text(text)
    cleaned = re.sub(r"^recommendation\s*:\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"^guideline\s*:\s*", "", cleaned, flags=re.IGNORECASE)
    return cleaned.strip()


def _dedupe_recommendations(items: Sequence[GuidelineRecommendation]) -> list[GuidelineRecommendation]:
    seen: set[tuple[str, str]] = set()
    out: list[GuidelineRecommendation] = []
    for item in items:
        key = (item.category.strip().lower(), item.recommendation.strip().lower())
        if key in seen or not item.recommendation.strip():
            continue
        seen.add(key)
        out.append(item)
    return out


def _clean_text(value: Any) -> str:
    text = str(value or "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\t+", " ", text)
    text = re.sub(r"[ ]{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


__all__ = [
    "GuidelineContext",
    "GuidelineRecommendation",
    "GuidelineResult",
    "build_guideline_prompt",
    "generate_guideline_text",
    "recommend_guidance",
    "render_recommendations",
    "summarize_context",
]
