"""Report generation pipeline for RadCopilot.

This module is the modular replacement for the large single-file report path in
RadCopilot. It orchestrates:
- request normalization
- template lookup
- findings-to-section mapping
- optional RAG retrieval
- impression generation through Ollama
- deterministic validation / repair
- optional LLM correction passes
- final text rendering

Design goals:
- integrate cleanly with the existing modular refactor
- remain usable even when only part of the package exists
- provide deterministic fallbacks when the model or RAG is unavailable
- avoid clinical invention during fallback behavior
"""
from __future__ import annotations

from dataclasses import dataclass, field
import json
import re
from typing import Any, Iterable, Mapping, Optional, Protocol, Sequence

from radcopilot.rag.library import query_records
from radcopilot.report.fixer import FixResult, auto_fix_impression
from radcopilot.report.validator import ValidationResult, should_attempt_repair, validate_impression
from radcopilot.services.ollama_client import OllamaClient, OllamaClientError


class ConfigLike(Protocol):
    """Minimal config contract used by this module."""

    base_dir: Any
    ollama_url: str


_DEFAULT_TEMPLATE_ID = "ct-chest"
_DEFAULT_MODEL = "llama3.1:8b"
_MAX_FINDINGS_CHARS = 12000
_MAX_INDICATION_CHARS = 1000

_MEASUREMENT_RE = re.compile(r"\b\d+(?:\.\d+)?\s*(?:mm|cm)\b", re.IGNORECASE)
_SECTION_LINE_RE = re.compile(r"^\s*([A-Za-z][A-Za-z /_-]{1,50})\s*:\s*(.+?)\s*$")
_LINE_SPLIT_RE = re.compile(r"\n+|(?<=[.!?])\s+")

_SECTION_ALIASES: dict[str, tuple[str, ...]] = {
    "lungs": ("lungs", "lung", "pulmonary", "airspace", "parenchyma"),
    "pleura": ("pleura", "pleural", "effusion", "pneumothorax"),
    "mediastinum": ("mediastinum", "mediastinal", "hila", "hilum", "nodes", "lymph node"),
    "heart": ("heart", "cardiac", "pericardial", "coronary"),
    "chest_wall": ("chest wall", "osseous", "rib", "spine", "bony", "soft tissues"),
    "liver": ("liver", "hepatic"),
    "gallbladder": ("gallbladder", "biliary", "duct", "cholelithiasis", "cholang"),
    "spleen": ("spleen", "splenic"),
    "pancreas": ("pancreas", "pancreatic"),
    "adrenals": ("adrenal", "adrenals"),
    "kidneys": ("kidney", "kidneys", "renal", "hydronephrosis", "ureter"),
    "bowel": ("bowel", "colonic", "colon", "ileum", "jejunum", "appendix", "stomach"),
    "peritoneum": ("peritoneum", "ascites", "mesentery", "omental"),
    "pelvis": ("pelvis", "bladder", "uterus", "prostate", "adnexa", "ovary", "ovarian"),
    "brain": ("brain", "intracranial", "hemorrhage", "ventricle", "midline"),
    "sinuses": ("sinus", "sinuses", "mastoid"),
    "vasculature": ("aorta", "vascular", "vessel", "artery", "vein", "ivc"),
    "general": ("general", "other", "misc", "findings"),
}


@dataclass(slots=True)
class ReportTemplate:
    """Definition of a report template."""

    id: str
    label: str
    modality: str
    section_order: list[str]
    section_labels: dict[str, str]
    section_defaults: dict[str, str] = field(default_factory=dict)
    guideline_hint: str = ""
    allow_negatives_in_impression: bool = False


@dataclass(slots=True)
class ReportRequest:
    """Input request for the report pipeline."""

    findings: str
    indication: str = ""
    age: str = ""
    sex: str = ""
    template_id: str = _DEFAULT_TEMPLATE_ID
    model: str = _DEFAULT_MODEL
    use_rag: bool = True
    max_rag_examples: int = 3
    max_repair_passes: int = 2
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "ReportRequest":
        return cls(
            findings=str(payload.get("findings", "") or ""),
            indication=str(payload.get("indication", "") or ""),
            age=str(payload.get("age", "") or ""),
            sex=str(payload.get("sex", "") or ""),
            template_id=str(payload.get("template_id", _DEFAULT_TEMPLATE_ID) or _DEFAULT_TEMPLATE_ID),
            model=str(payload.get("model", _DEFAULT_MODEL) or _DEFAULT_MODEL),
            use_rag=bool(payload.get("use_rag", True)),
            max_rag_examples=int(payload.get("max_rag_examples", 3) or 3),
            max_repair_passes=int(payload.get("max_repair_passes", 2) or 2),
            metadata=dict(payload.get("metadata", {}) or {}),
        )

    def normalized(self) -> "ReportRequest":
        return ReportRequest(
            findings=_clean_text(self.findings)[:_MAX_FINDINGS_CHARS],
            indication=_clean_text(self.indication)[:_MAX_INDICATION_CHARS],
            age=self.age.strip(),
            sex=self.sex.strip(),
            template_id=(self.template_id or _DEFAULT_TEMPLATE_ID).strip().lower(),
            model=(self.model or _DEFAULT_MODEL).strip(),
            use_rag=bool(self.use_rag),
            max_rag_examples=max(0, min(int(self.max_rag_examples), 8)),
            max_repair_passes=max(0, min(int(self.max_repair_passes), 4)),
            metadata=dict(self.metadata),
        )


@dataclass(slots=True)
class ReportResult:
    """Structured output from the report pipeline."""

    ok: bool
    template_id: str
    template_label: str
    findings: str
    indication: str
    section_map: dict[str, list[str]]
    rendered_sections: dict[str, str]
    rag_examples: list[dict[str, Any]] = field(default_factory=list)
    raw_impression: str = ""
    final_impression: str = ""
    validation: Optional[ValidationResult] = None
    fix_result: Optional[FixResult] = None
    final_report: str = ""
    warnings: list[str] = field(default_factory=list)
    trace: list[str] = field(default_factory=list)
    model_used: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "template_id": self.template_id,
            "template_label": self.template_label,
            "findings": self.findings,
            "indication": self.indication,
            "section_map": self.section_map,
            "rendered_sections": self.rendered_sections,
            "rag_examples": self.rag_examples,
            "raw_impression": self.raw_impression,
            "final_impression": self.final_impression,
            "validation": self.validation.to_dict() if self.validation else None,
            "fix_result": self.fix_result.to_dict() if self.fix_result else None,
            "final_report": self.final_report,
            "warnings": self.warnings,
            "trace": self.trace,
            "model_used": self.model_used,
        }


TEMPLATES: dict[str, ReportTemplate] = {
    "ct-chest": ReportTemplate(
        id="ct-chest",
        label="CT Chest",
        modality="ct-chest",
        section_order=["lungs", "pleura", "mediastinum", "heart", "vasculature", "chest_wall"],
        section_labels={
            "lungs": "Lungs",
            "pleura": "Pleura",
            "mediastinum": "Mediastinum / Hila",
            "heart": "Heart / Pericardium",
            "vasculature": "Vasculature",
            "chest_wall": "Chest Wall / Osseous Structures",
        },
        section_defaults={
            "lungs": "No focal air-space consolidation or suspicious acute pulmonary abnormality.",
            "pleura": "No pleural effusion or pneumothorax.",
            "mediastinum": "No pathologically enlarged mediastinal lymph nodes are identified.",
            "heart": "Heart size is within normal limits.",
            "vasculature": "No acute vascular abnormality is evident on this exam.",
            "chest_wall": "No acute osseous abnormality is evident on this exam.",
        },
        guideline_hint="Use pulmonary nodule follow-up language only when the findings clearly describe a lung nodule.",
    ),
    "ct-abdomen-pelvis": ReportTemplate(
        id="ct-abdomen-pelvis",
        label="CT Abdomen / Pelvis",
        modality="ct-abdomen-pelvis",
        section_order=["liver", "gallbladder", "spleen", "pancreas", "adrenals", "kidneys", "bowel", "peritoneum", "pelvis", "vasculature"],
        section_labels={
            "liver": "Liver",
            "gallbladder": "Gallbladder / Biliary",
            "spleen": "Spleen",
            "pancreas": "Pancreas",
            "adrenals": "Adrenal Glands",
            "kidneys": "Kidneys / Urinary Tract",
            "bowel": "Bowel",
            "peritoneum": "Peritoneum / Mesentery",
            "pelvis": "Pelvic Organs",
            "vasculature": "Vasculature",
        },
        section_defaults={
            "liver": "No acute hepatic abnormality is identified.",
            "gallbladder": "No acute biliary abnormality is identified.",
            "spleen": "No acute splenic abnormality is identified.",
            "pancreas": "No acute pancreatic abnormality is identified.",
            "adrenals": "No suspicious adrenal abnormality is identified.",
            "kidneys": "No hydronephrosis or other acute urinary tract abnormality is identified.",
            "bowel": "No bowel obstruction or focal acute inflammatory bowel process is identified.",
            "peritoneum": "No ascites or free intraperitoneal air.",
            "pelvis": "No acute pelvic abnormality is identified on this exam.",
            "vasculature": "No acute vascular abnormality is evident on this exam.",
        },
        guideline_hint="Do not apply pulmonary nodule guidance to adrenal or adnexal findings.",
    ),
    "xr-chest": ReportTemplate(
        id="xr-chest",
        label="Chest X-Ray",
        modality="xr-chest",
        section_order=["lungs", "pleura", "heart", "mediastinum", "chest_wall"],
        section_labels={
            "lungs": "Lungs",
            "pleura": "Pleura",
            "heart": "Cardiomediastinal Silhouette",
            "mediastinum": "Mediastinum",
            "chest_wall": "Osseous Structures",
        },
        section_defaults={
            "lungs": "No focal air-space opacity is identified.",
            "pleura": "No pleural effusion or pneumothorax.",
            "heart": "Cardiomediastinal silhouette is within normal size limits.",
            "mediastinum": "No acute mediastinal abnormality is evident on this exam.",
            "chest_wall": "No acute osseous abnormality is identified on this exam.",
        },
    ),
    "mri-brain": ReportTemplate(
        id="mri-brain",
        label="MRI Brain",
        modality="mri-brain",
        section_order=["brain", "sinuses", "vasculature", "general"],
        section_labels={
            "brain": "Brain",
            "sinuses": "Paranasal Sinuses / Mastoids",
            "vasculature": "Vascular Flow Voids",
            "general": "Other",
        },
        section_defaults={
            "brain": "No acute intracranial abnormality is identified on this exam.",
            "sinuses": "No acute paranasal sinus or mastoid abnormality is evident on this exam.",
            "vasculature": "Major intracranial vascular flow voids are preserved.",
            "general": "No additional acute finding is identified.",
        },
    ),
    "generic": ReportTemplate(
        id="generic",
        label="Generic Report",
        modality="generic",
        section_order=["general"],
        section_labels={"general": "Findings"},
        section_defaults={"general": "No acute abnormality is identified on this exam."},
    ),
}


def get_template(template_id: str | None) -> ReportTemplate:
    """Return a template or a safe generic fallback."""
    key = (template_id or _DEFAULT_TEMPLATE_ID).strip().lower()
    return TEMPLATES.get(key, TEMPLATES["generic"])


def list_templates() -> list[dict[str, Any]]:
    """Return serializable template metadata for UI or API use."""
    return [
        {
            "id": template.id,
            "label": template.label,
            "modality": template.modality,
            "section_order": list(template.section_order),
            "section_labels": dict(template.section_labels),
        }
        for template in TEMPLATES.values()
    ]


def generate_report(
    request: ReportRequest | Mapping[str, Any],
    *,
    config: ConfigLike | None = None,
    client: OllamaClient | None = None,
) -> ReportResult:
    """Execute the report generation pipeline.

    The function remains usable even when Ollama is unavailable. In that case it
    falls back to deterministic section mapping and a conservative impression.
    """
    req = request if isinstance(request, ReportRequest) else ReportRequest.from_mapping(request)
    req = req.normalized()
    template = get_template(req.template_id)

    if not req.findings:
        result = ReportResult(
            ok=False,
            template_id=template.id,
            template_label=template.label,
            findings=req.findings,
            indication=req.indication,
            section_map={section: [] for section in template.section_order},
            rendered_sections=_build_default_rendered_sections(template),
            final_report="",
            warnings=["Findings are required to generate a report."],
            trace=["request:empty_findings"],
            model_used=req.model,
        )
        return result

    trace: list[str] = ["request:normalized"]
    warnings: list[str] = []
    client = client or _build_client(config=config, model=req.model)

    section_map = _map_findings(
        findings_text=req.findings,
        template=template,
        client=client,
        trace=trace,
        warnings=warnings,
    )
    rendered_sections = _merge_with_defaults(template, section_map)

    rag_examples: list[dict[str, Any]] = []
    if req.use_rag and req.max_rag_examples > 0:
        rag_examples = _get_rag_examples(
            config=config,
            findings_text=req.findings,
            template=template,
            max_examples=req.max_rag_examples,
            trace=trace,
            warnings=warnings,
        )

    raw_impression = _generate_impression(
        req=req,
        template=template,
        section_map=section_map,
        rag_examples=rag_examples,
        client=client,
        trace=trace,
        warnings=warnings,
    )

    validation = validate_impression(
        raw_impression,
        findings_text=req.findings,
        abnormals=section_map,
        template_id=template.id,
        allow_negatives=template.allow_negatives_in_impression,
    )
    trace.append(f"validation:initial score={validation.score:.3f} errors={validation.error_count()} warnings={validation.warning_count()}")

    fix_result = auto_fix_impression(
        raw_impression,
        findings_text=req.findings,
        abnormals=section_map,
        template_id=template.id,
        allow_negatives=template.allow_negatives_in_impression,
        max_lines=5,
    )
    final_impression = fix_result.fixed_impression or raw_impression
    final_validation = fix_result.final_validation
    trace.append(f"fixer:changed={fix_result.changed} escalate={fix_result.should_escalate_to_llm}")

    if client and fix_result.should_escalate_to_llm and req.max_repair_passes > 0:
        repaired = _attempt_llm_repairs(
            req=req,
            template=template,
            findings_text=req.findings,
            section_map=section_map,
            current_impression=final_impression,
            client=client,
            trace=trace,
            warnings=warnings,
        )
        if repaired:
            repaired_validation = validate_impression(
                repaired,
                findings_text=req.findings,
                abnormals=section_map,
                template_id=template.id,
                allow_negatives=template.allow_negatives_in_impression,
            )
            trace.append(
                f"validation:post_repair score={repaired_validation.score:.3f} errors={repaired_validation.error_count()} warnings={repaired_validation.warning_count()}"
            )
            if repaired_validation.score >= final_validation.score:
                final_impression = repaired
                final_validation = repaired_validation

    if not final_impression.strip():
        final_impression = _fallback_impression(section_map, template=template)
        final_validation = validate_impression(
            final_impression,
            findings_text=req.findings,
            abnormals=section_map,
            template_id=template.id,
            allow_negatives=template.allow_negatives_in_impression,
        )
        warnings.append("Used deterministic fallback impression.")
        trace.append("impression:fallback")

    final_report = render_report(
        template=template,
        indication=req.indication,
        section_text=rendered_sections,
        impression=final_impression,
        age=req.age,
        sex=req.sex,
    )

    return ReportResult(
        ok=bool(final_report.strip()),
        template_id=template.id,
        template_label=template.label,
        findings=req.findings,
        indication=req.indication,
        section_map=section_map,
        rendered_sections=rendered_sections,
        rag_examples=rag_examples,
        raw_impression=raw_impression,
        final_impression=final_impression,
        validation=final_validation,
        fix_result=fix_result,
        final_report=final_report,
        warnings=warnings,
        trace=trace,
        model_used=req.model,
    )


def render_report(
    *,
    template: ReportTemplate,
    indication: str,
    section_text: Mapping[str, str],
    impression: str,
    age: str = "",
    sex: str = "",
) -> str:
    """Render a final plain-text radiology report."""
    parts: list[str] = []
    if indication:
        parts.append(f"INDICATION: {indication}")
    if age or sex:
        demographic_bits = []
        if age:
            demographic_bits.append(f"Age: {age}")
        if sex:
            demographic_bits.append(f"Sex: {sex}")
        parts.append("PATIENT: " + " | ".join(demographic_bits))

    parts.append("FINDINGS:")
    for section in template.section_order:
        label = template.section_labels.get(section, section.title())
        text = section_text.get(section, "").strip()
        if text:
            parts.append(f"{label}: {text}")

    parts.append("")
    parts.append("IMPRESSION:")
    impression_text = impression.strip() or "1. No acute abnormality identified on this exam."
    parts.extend([line for line in impression_text.splitlines() if line.strip()])
    return "\n".join(parts).strip() + "\n"


def build_correction_prompt(
    *,
    template: ReportTemplate,
    findings_text: str,
    section_map: Mapping[str, Sequence[str]],
    current_impression: str,
    validation: ValidationResult,
) -> str:
    """Build a tightly scoped correction prompt for the LLM."""
    issues = []
    for issue in validation.issues:
        issues.append(f"- {issue.message}")
    abnormal_block = _abnormal_block(section_map, template)
    return (
        "You are correcting a radiology impression.\n"
        "Rewrite only the impression.\n"
        "Requirements:\n"
        "- Keep only findings supported by the source findings and abnormal sections.\n"
        "- Do not include prompt text, section mappings, or instructional language.\n"
        "- Use concise numbered impression lines.\n"
        "- Do not add new anatomy, diagnoses, or measurements.\n"
        "- Avoid negative filler unless required by the source finding.\n\n"
        f"Template: {template.label}\n"
        f"Guidance: {template.guideline_hint or 'No special guidance.'}\n\n"
        f"Source findings:\n{findings_text}\n\n"
        f"Abnormal sections:\n{abnormal_block}\n\n"
        f"Current impression:\n{current_impression}\n\n"
        f"Problems to fix:\n{'\n'.join(issues) if issues else '- Make the impression concise and supported.'}\n\n"
        "Return only the corrected impression text."
    )


def build_impression_prompt(
    *,
    req: ReportRequest,
    template: ReportTemplate,
    section_map: Mapping[str, Sequence[str]],
    rag_examples: Sequence[Mapping[str, Any]],
) -> str:
    """Build the main impression-generation prompt."""
    abnormal_block = _abnormal_block(section_map, template)
    example_block = _rag_block(rag_examples)
    patient_bits = []
    if req.age:
        patient_bits.append(f"Age: {req.age}")
    if req.sex:
        patient_bits.append(f"Sex: {req.sex}")
    patient_context = " | ".join(patient_bits) if patient_bits else "Not provided"
    return (
        "You are generating the IMPRESSION section of a radiology report.\n"
        "Requirements:\n"
        "- Output only the impression.\n"
        "- Use concise numbered lines.\n"
        "- Include only positive or clinically meaningful abnormalities supported by the findings.\n"
        "- Do not mention prompt instructions or section names unless clinically necessary.\n"
        "- Do not invent new diagnoses, anatomy, or measurements.\n"
        "- Keep the impression short and clinically focused.\n\n"
        f"Template: {template.label}\n"
        f"Patient: {patient_context}\n"
        f"Indication: {req.indication or 'Not provided'}\n"
        f"Guidance: {template.guideline_hint or 'No special guidance.'}\n\n"
        f"Source findings:\n{req.findings}\n\n"
        f"Abnormal sections:\n{abnormal_block}\n\n"
        f"Reference examples (style only, not facts):\n{example_block}\n\n"
        "Return only the impression text."
    )


def build_mapping_prompt(*, template: ReportTemplate, findings_text: str) -> str:
    """Build a strict findings-to-sections prompt."""
    allowed = ", ".join(template.section_order)
    return (
        "Map each abnormal finding into the most appropriate report section.\n"
        "Rules:\n"
        "- Use only the allowed section names exactly as provided.\n"
        "- Output one mapping per line in the format: Section | finding\n"
        "- Keep each finding concise and faithful to the source text.\n"
        "- Do not invent content.\n"
        "- Skip clearly normal filler statements.\n\n"
        f"Allowed sections: {allowed}\n\n"
        f"Findings:\n{findings_text}\n"
    )


def _build_client(*, config: ConfigLike | None, model: str) -> OllamaClient | None:
    base_url = getattr(config, "ollama_url", None) if config is not None else None
    try:
        client = OllamaClient(base_url=base_url or "http://localhost:11434", model=model)
    except Exception:
        return None
    return client


def _map_findings(
    *,
    findings_text: str,
    template: ReportTemplate,
    client: OllamaClient | None,
    trace: list[str],
    warnings: list[str],
) -> dict[str, list[str]]:
    direct = _extract_direct_section_lines(findings_text, template=template)
    consumed = {text for values in direct.values() for text in values}
    residual_lines = [line for line in _split_source_lines(findings_text) if line and line not in consumed]

    section_map: dict[str, list[str]] = {section: list(values) for section, values in direct.items()}
    for section in template.section_order:
        section_map.setdefault(section, [])

    if residual_lines and client is not None:
        prompt = build_mapping_prompt(template=template, findings_text="\n".join(residual_lines))
        try:
            mapped_text = client.chat_text(prompt, model=client.model)
            parsed = _parse_mapped_lines(mapped_text, template=template)
            if parsed:
                trace.append("mapping:llm")
                _merge_section_map(section_map, parsed)
            else:
                trace.append("mapping:llm_empty")
                warnings.append("LLM mapping response was empty or could not be parsed; using heuristic mapping for remaining findings.")
                _merge_section_map(section_map, _heuristic_map_lines(residual_lines, template=template))
        except OllamaClientError as exc:
            trace.append("mapping:llm_error")
            warnings.append(f"LLM mapping failed: {exc}. Using heuristic mapping.")
            _merge_section_map(section_map, _heuristic_map_lines(residual_lines, template=template))
    else:
        if residual_lines:
            trace.append("mapping:heuristic")
            _merge_section_map(section_map, _heuristic_map_lines(residual_lines, template=template))
        else:
            trace.append("mapping:direct_only")

    for section, values in list(section_map.items()):
        section_map[section] = _unique_preserve_order([_clean_text(v) for v in values if _clean_text(v)])
    return section_map


def _get_rag_examples(
    *,
    config: ConfigLike | None,
    findings_text: str,
    template: ReportTemplate,
    max_examples: int,
    trace: list[str],
    warnings: list[str],
) -> list[dict[str, Any]]:
    if config is None:
        trace.append("rag:skipped_no_config")
        return []
    try:
        items = query_records(
            config=config,
            findings=findings_text,
            modality=template.modality,
            k=max_examples,
        )
        trace.append(f"rag:query count={len(items)}")
        return items
    except Exception as exc:
        trace.append("rag:error")
        warnings.append(f"RAG query failed: {type(exc).__name__}: {exc}")
        return []


def _generate_impression(
    *,
    req: ReportRequest,
    template: ReportTemplate,
    section_map: Mapping[str, Sequence[str]],
    rag_examples: Sequence[Mapping[str, Any]],
    client: OllamaClient | None,
    trace: list[str],
    warnings: list[str],
) -> str:
    fallback = _fallback_impression(section_map, template=template)
    if client is None:
        trace.append("impression:fallback_no_client")
        warnings.append("Ollama client unavailable; using deterministic fallback impression.")
        return fallback

    prompt = build_impression_prompt(req=req, template=template, section_map=section_map, rag_examples=rag_examples)
    try:
        text = client.chat_text(prompt, model=req.model)
        text = _strip_headers_from_impression(text)
        if text.strip():
            trace.append("impression:llm")
            return text.strip()
        trace.append("impression:llm_empty")
    except OllamaClientError as exc:
        trace.append("impression:llm_error")
        warnings.append(f"Impression generation failed: {exc}. Using deterministic fallback impression.")

    return fallback


def _attempt_llm_repairs(
    *,
    req: ReportRequest,
    template: ReportTemplate,
    findings_text: str,
    section_map: Mapping[str, Sequence[str]],
    current_impression: str,
    client: OllamaClient,
    trace: list[str],
    warnings: list[str],
) -> str:
    candidate = current_impression
    for attempt in range(1, req.max_repair_passes + 1):
        validation = validate_impression(
            candidate,
            findings_text=findings_text,
            abnormals=section_map,
            template_id=template.id,
            allow_negatives=template.allow_negatives_in_impression,
        )
        if not should_attempt_repair(validation):
            break
        prompt = build_correction_prompt(
            template=template,
            findings_text=findings_text,
            section_map=section_map,
            current_impression=candidate,
            validation=validation,
        )
        try:
            repaired = client.chat_text(prompt, model=req.model)
        except OllamaClientError as exc:
            warnings.append(f"Repair pass {attempt} failed: {exc}")
            trace.append(f"repair:error attempt={attempt}")
            break
        repaired = _strip_headers_from_impression(repaired)
        if not repaired.strip():
            trace.append(f"repair:empty attempt={attempt}")
            break
        candidate = repaired.strip()
        trace.append(f"repair:llm attempt={attempt}")
    return candidate


def _fallback_impression(section_map: Mapping[str, Sequence[str]], *, template: ReportTemplate) -> str:
    lines: list[str] = []
    for section in template.section_order:
        values = [v.strip().rstrip(".") for v in section_map.get(section, []) if v.strip()]
        if not values:
            continue
        label = template.section_labels.get(section, section.title())
        summary = "; ".join(values[:2])
        lines.append(f"{label}: {summary}.")
        if len(lines) >= 4:
            break
    if not lines:
        return "1. No acute abnormality identified on this exam."
    return "\n".join(f"{idx}. {line}" for idx, line in enumerate(lines, start=1))


def _build_default_rendered_sections(template: ReportTemplate) -> dict[str, str]:
    return {section: template.section_defaults.get(section, "") for section in template.section_order}


def _merge_with_defaults(template: ReportTemplate, section_map: Mapping[str, Sequence[str]]) -> dict[str, str]:
    rendered: dict[str, str] = {}
    for section in template.section_order:
        values = [v.strip().rstrip(".") for v in section_map.get(section, []) if v and v.strip()]
        if values:
            rendered[section] = "; ".join(values) + "."
        else:
            rendered[section] = template.section_defaults.get(section, "")
    return rendered


def _extract_direct_section_lines(findings_text: str, *, template: ReportTemplate) -> dict[str, list[str]]:
    output: dict[str, list[str]] = {section: [] for section in template.section_order}
    for raw_line in findings_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        match = _SECTION_LINE_RE.match(line)
        if not match:
            continue
        section_name = _match_section_name(match.group(1), template=template)
        if not section_name:
            continue
        output.setdefault(section_name, []).append(_clean_text(match.group(2)))
    return output


def _parse_mapped_lines(mapped_text: str, *, template: ReportTemplate) -> dict[str, list[str]]:
    parsed: dict[str, list[str]] = {section: [] for section in template.section_order}
    for raw_line in str(mapped_text or "").splitlines():
        line = raw_line.strip().lstrip("-*• ")
        if not line or "|" not in line:
            continue
        left, right = line.split("|", 1)
        section = _match_section_name(left.strip(), template=template)
        value = _clean_text(right)
        if section and value:
            parsed.setdefault(section, []).append(value)
    return {k: v for k, v in parsed.items() if v}


def _heuristic_map_lines(lines: Sequence[str], *, template: ReportTemplate) -> dict[str, list[str]]:
    mapped: dict[str, list[str]] = {section: [] for section in template.section_order}
    for line in lines:
        normalized = _clean_text(line)
        if not normalized:
            continue
        section = _infer_section_from_text(normalized, template=template)
        mapped.setdefault(section, []).append(normalized)
    return mapped


def _infer_section_from_text(text: str, *, template: ReportTemplate) -> str:
    lower = text.lower()
    best_section = template.section_order[0] if template.section_order else "general"
    best_score = -1
    for section in template.section_order:
        score = 0
        for token in _SECTION_ALIASES.get(section, (section,)):
            if token in lower:
                score += len(token)
        if score > best_score:
            best_score = score
            best_section = section
    return best_section


def _merge_section_map(target: dict[str, list[str]], source: Mapping[str, Sequence[str]]) -> None:
    for section, values in source.items():
        target.setdefault(section, [])
        for value in values:
            cleaned = _clean_text(value)
            if cleaned:
                target[section].append(cleaned)


def _abnormal_block(section_map: Mapping[str, Sequence[str]], template: ReportTemplate) -> str:
    lines: list[str] = []
    for section in template.section_order:
        values = [v for v in section_map.get(section, []) if v and str(v).strip()]
        if not values:
            continue
        label = template.section_labels.get(section, section.title())
        lines.append(f"{label}: {'; '.join(values)}")
    return "\n".join(lines) if lines else "No clear abnormal section content provided."


def _rag_block(examples: Sequence[Mapping[str, Any]]) -> str:
    if not examples:
        return "No retrieval examples available."
    chunks: list[str] = []
    for idx, item in enumerate(examples[:5], start=1):
        findings = _clean_text(str(item.get("findings", "")))
        impression = _clean_text(str(item.get("impression", "")))
        chunks.append(f"Example {idx}\nFindings: {findings}\nImpression: {impression}")
    return "\n\n".join(chunks)


def _split_source_lines(text: str) -> list[str]:
    parts = []
    for chunk in _LINE_SPLIT_RE.split(text or ""):
        line = _clean_text(chunk)
        if line:
            parts.append(line)
    return _unique_preserve_order(parts)


def _strip_headers_from_impression(text: str) -> str:
    cleaned = str(text or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    cleaned = re.sub(r"^\s*(?:impression|conclusion|summary)\s*:\s*", "", cleaned, flags=re.IGNORECASE)
    return cleaned.strip()


def _match_section_name(name: str, *, template: ReportTemplate) -> Optional[str]:
    normalized = _clean_text(name).lower()
    if not normalized:
        return None
    for section in template.section_order:
        if normalized == section:
            return section
        label = template.section_labels.get(section, "").lower()
        if normalized == label or normalized in label:
            return section
        aliases = _SECTION_ALIASES.get(section, ())
        if normalized in aliases:
            return section
    return None


def _clean_text(value: str) -> str:
    text = str(value or "").replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _unique_preserve_order(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for item in values:
        key = item.lower().strip().rstrip(".")
        if not key or key in seen:
            continue
        seen.add(key)
        output.append(item.strip())
    return output


__all__ = [
    "ReportRequest",
    "ReportResult",
    "ReportTemplate",
    "TEMPLATES",
    "build_correction_prompt",
    "build_impression_prompt",
    "build_mapping_prompt",
    "generate_report",
    "get_template",
    "list_templates",
    "render_report",
]
