from __future__ import annotations

"""
radcopilot.server.report_routes

Dedicated backend routes for the report pipeline.

Why this file exists:
- the modular refactor now has a reusable report generator, validator, and
  guideline layer
- the UI currently orchestrates much of the workflow client-side; this module
  exposes a clean backend contract for report-related operations
- it keeps report concerns out of the generic server/app and route-registry code

What this module provides:
- `/report/templates` metadata endpoints
- `/report/generate` for end-to-end report generation
- `/report/guidelines` for follow-up recommendation generation
- `/report/validate` for deterministic impression validation
- helpers to register these routes on a `RouteRegistry`
"""

from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Optional, Protocol, Sequence

from radcopilot.report.generator import (
    ReportRequest,
    ReportTemplate,
    generate_report,
    get_template,
    list_templates,
)
from radcopilot.report.guidelines import (
    GuidelineContext,
    generate_guideline_text,
)
from radcopilot.report.validator import summarize_issues, validate_impression
from radcopilot.server.routes import RouteContext, RouteRegistry, send_json

try:  # optional shared logging dependency
    from radcopilot.services.logging_service import log_event, log_exception
except Exception:  # pragma: no cover - logging remains optional during refactor
    log_event = None
    log_exception = None


class ConfigLike(Protocol):
    """Minimal runtime config expected from the launcher/server."""

    ollama_url: str
    log_file: Path


def build_report_registry() -> RouteRegistry:
    """Build a registry containing only report-focused routes."""
    registry = RouteRegistry()
    register_report_routes(registry)
    return registry


def register_report_routes(registry: RouteRegistry) -> RouteRegistry:
    """Attach report routes to an existing registry and return it."""
    registry.get(
        "/report/templates",
        handle_report_templates,
        name="report_templates",
        description="List available report templates.",
        tags=("report", "templates"),
    )
    registry.get(
        "/report/templates/{template_id}",
        handle_report_template_detail,
        name="report_template_detail",
        description="Get one report template definition.",
        tags=("report", "templates"),
    )
    registry.post(
        "/report/generate",
        handle_report_generate,
        name="report_generate",
        description="Generate a report from findings and template context.",
        tags=("report", "generation"),
    )
    registry.post(
        "/report/guidelines",
        handle_report_guidelines,
        name="report_guidelines",
        description="Generate guideline-oriented follow-up recommendations.",
        tags=("report", "guidelines"),
    )
    registry.post(
        "/report/validate",
        handle_report_validate,
        name="report_validate",
        description="Validate an impression against deterministic rules.",
        tags=("report", "validation"),
    )
    return registry


def describe_report_routes() -> list[dict[str, Any]]:
    """Return serializable metadata describing the report route surface."""
    registry = build_report_registry()
    items: list[dict[str, Any]] = []
    for route in registry.iter_routes():
        items.append(
            {
                "method": route.method,
                "pattern": route.pattern,
                "name": route.name,
                "description": route.description,
                "tags": list(route.tags),
            }
        )
    return items


def handle_report_templates(ctx: RouteContext) -> None:
    """Return high-level template metadata for UI pickers and docs."""
    payload = {
        "ok": True,
        "default_template_id": "ct-chest",
        "items": list_templates(),
    }
    _log_event(
        "REPORT_TEMPLATES",
        detail="Listed report templates.",
        ctx=ctx,
        level="INFO",
    )
    send_json(ctx.request_handler, payload)


def handle_report_template_detail(ctx: RouteContext) -> None:
    """Return one template definition including defaults and hints."""
    template_id = (ctx.params.get("template_id") or "").strip().lower()
    template = get_template(template_id)
    ok = template.id == template_id or (not template_id and template.id == "ct-chest")
    payload = {
        "ok": True,
        "requested_template_id": template_id,
        "resolved": ok,
        "item": _serialize_template(template),
    }
    send_json(ctx.request_handler, payload)


def handle_report_generate(ctx: RouteContext) -> None:
    """Execute the end-to-end report pipeline and return the result."""
    try:
        payload = _coerce_object(ctx.read_json())
        request = ReportRequest.from_mapping(payload)
        result = generate_report(request, config=ctx.config)
        response: dict[str, Any] = result.to_dict()
        response.setdefault("ok", result.ok)

        if bool(payload.get("include_guidelines")):
            guideline_payload = _merge_template_context(payload)
            guideline_result = generate_guideline_text(
                config=ctx.config,
                context=guideline_payload,
                use_model=bool(payload.get("guidelines_use_model", True)),
            )
            response["guidelines"] = guideline_result.to_dict()

        _log_event(
            "REPORT_GENERATE",
            detail="Generated report.",
            ctx=ctx,
            level="INFO" if result.ok else "WARNING",
            context={
                "template_id": result.template_id,
                "model": result.model_used,
                "warnings": len(result.warnings),
                "validation_score": round(result.validation.score, 4) if result.validation else None,
                "trace_steps": len(result.trace),
            },
        )
        send_json(ctx.request_handler, response, status=200 if result.ok else 422)
    except Exception as exc:
        _log_exception(exc, ctx=ctx, event="report_generate")
        send_json(
            ctx.request_handler,
            {
                "ok": False,
                "error": f"Report generation failed: {type(exc).__name__}: {exc}",
            },
            status=500,
        )


def handle_report_guidelines(ctx: RouteContext) -> None:
    """Generate guideline-oriented recommendation text for a case."""
    try:
        payload = _merge_template_context(_coerce_object(ctx.read_json()))
        use_model = bool(payload.pop("use_model", True))
        result = generate_guideline_text(config=ctx.config, context=payload, use_model=use_model)
        _log_event(
            "REPORT_GUIDELINES",
            detail="Generated guideline recommendations.",
            ctx=ctx,
            level="INFO",
            context={
                "recommendations": len(result.recommendations),
                "model": result.model_used,
                "warnings": len(result.warnings),
            },
        )
        send_json(ctx.request_handler, result.to_dict())
    except Exception as exc:
        _log_exception(exc, ctx=ctx, event="report_guidelines")
        send_json(
            ctx.request_handler,
            {
                "ok": False,
                "error": f"Guideline generation failed: {type(exc).__name__}: {exc}",
            },
            status=500,
        )


def handle_report_validate(ctx: RouteContext) -> None:
    """Validate impression text using deterministic report rules."""
    try:
        payload = _merge_template_context(_coerce_object(ctx.read_json()))
        impression = str(payload.get("impression", "") or "")
        findings = str(payload.get("findings", "") or "")
        section_map = _coerce_abnormals(payload.get("abnormals") or payload.get("section_map"))
        template_id = str(payload.get("template_id", "") or "")
        allow_negatives = bool(payload.get("allow_negatives", False))

        if not impression.strip():
            send_json(
                ctx.request_handler,
                {"ok": False, "error": "Missing required field: impression"},
                status=400,
            )
            return

        result = validate_impression(
            impression,
            findings_text=findings,
            abnormals=section_map,
            template_id=template_id,
            allow_negatives=allow_negatives,
        )
        response = {
            "ok": True,
            "valid": result.valid,
            "summary": summarize_issues(result),
            "validation": result.to_dict(),
        }
        _log_event(
            "REPORT_VALIDATE",
            detail="Validated impression.",
            ctx=ctx,
            level="INFO" if result.valid else "WARNING",
            context={
                "template_id": template_id,
                "score": round(result.score, 4),
                "errors": result.error_count(),
                "warnings": result.warning_count(),
            },
        )
        send_json(ctx.request_handler, response)
    except Exception as exc:
        _log_exception(exc, ctx=ctx, event="report_validate")
        send_json(
            ctx.request_handler,
            {
                "ok": False,
                "error": f"Report validation failed: {type(exc).__name__}: {exc}",
            },
            status=500,
        )


def _serialize_template(template: ReportTemplate) -> dict[str, Any]:
    return {
        "id": template.id,
        "label": template.label,
        "modality": template.modality,
        "section_order": list(template.section_order),
        "section_labels": dict(template.section_labels),
        "section_defaults": dict(template.section_defaults),
        "guideline_hint": template.guideline_hint,
        "allow_negatives_in_impression": bool(template.allow_negatives_in_impression),
    }


def _merge_template_context(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Augment payload with template label and guideline hint when possible."""
    merged = dict(payload)
    template_id = str(merged.get("template_id", "") or "").strip().lower()
    if not template_id:
        return merged
    template = get_template(template_id)
    merged.setdefault("template_label", template.label)
    if template.guideline_hint and not merged.get("guideline_hint"):
        merged["guideline_hint"] = template.guideline_hint
    if "allow_negatives" not in merged:
        merged["allow_negatives"] = template.allow_negatives_in_impression
    return merged


def _coerce_object(payload: Mapping[str, Any] | dict[str, Any]) -> dict[str, Any]:
    if isinstance(payload, dict):
        if "_root" in payload and len(payload) == 1 and isinstance(payload["_root"], Mapping):
            return dict(payload["_root"])
        return dict(payload)
    return dict(payload)


def _coerce_abnormals(value: Any) -> dict[str, list[str]] | None:
    if not value:
        return None
    if not isinstance(value, Mapping):
        return None
    result: dict[str, list[str]] = {}
    for key, raw in value.items():
        section = str(key).strip()
        if not section:
            continue
        if isinstance(raw, str):
            items = [raw.strip()] if raw.strip() else []
        elif isinstance(raw, Sequence) and not isinstance(raw, (bytes, bytearray, str)):
            items = [str(item).strip() for item in raw if str(item).strip()]
        else:
            text = str(raw).strip()
            items = [text] if text else []
        if items:
            result[section] = items
    return result or None


def _log_event(
    type_name: str,
    *,
    detail: str,
    ctx: RouteContext,
    level: str = "INFO",
    context: Mapping[str, Any] | None = None,
    event: str = "",
) -> None:
    if log_event is None:
        return
    try:
        log_event(
            type_name,
            detail,
            config=ctx.config,
            source="report_routes",
            level=level,
            event=event,
            route_path=ctx.path,
            context=context,
        )
    except Exception:
        return


def _log_exception(exc: BaseException, *, ctx: RouteContext, event: str = "") -> None:
    if log_exception is None:
        return
    try:
        log_exception(
            exc,
            config=ctx.config,
            source="report_routes",
            route_path=ctx.path,
            event=event,
            detail="Report route exception.",
            context={"method": ctx.method},
        )
    except Exception:
        return
