from __future__ import annotations

"""
radcopilot.server.app

Integrated HTTP server for the modular RadCopilot refactor.

This version replaces most inline route logic with the reusable server modules
that now exist in the package:
- `server.routes` for registry-based dispatch
- `server.report_routes` for report-focused endpoints
- `server.proxy` for Ollama forwarding
- `services.whisper_service` for speech-to-text handling
- `services.logging_service` for JSONL logging
- `rag.library` / `rag.trainer` for retrieval operations

The goal is to keep the runtime contract expected by `main.py` while moving the
implementation toward a modular monolith.
"""

import json
import mimetypes
from pathlib import Path
import traceback
from http.server import BaseHTTPRequestHandler
from typing import Any, Protocol, Type

from radcopilot.server.proxy import proxy_route_handler
from radcopilot.server.report_routes import register_report_routes
from radcopilot.server.routes import (
    CoreRouteHandlers,
    MethodNotAllowed,
    RouteContext,
    RouteNotFound,
    RouteRegistry,
    build_core_registry,
    dispatch_request,
    send_bytes,
    send_json,
    send_text,
)
from radcopilot.services.logging_service import log_event, log_exception, read_recent
from radcopilot.services.whisper_service import get_status as whisper_get_status
from radcopilot.services.whisper_service import transcribe_request
from radcopilot.rag.library import add_record, get_status as rag_get_status, query_records
from radcopilot.rag.trainer import train_path


class ConfigLike(Protocol):
    """Duck-typed runtime config expected from main.AppConfig."""

    app_name: str
    host: str
    port: int
    ollama_url: str
    base_dir: Path
    ui_dir: Path
    log_file: Path
    data_dir: Path

    @property
    def base_url(self) -> str:  # pragma: no cover - protocol only
        ...


def create_handler(config: ConfigLike) -> Type[BaseHTTPRequestHandler]:
    """Create a request handler bound to runtime config and route registry."""

    def serve_index(ctx: RouteContext) -> None:
        ui_index = config.ui_dir / "index.html"
        if ui_index.exists() and ui_index.is_file():
            send_text(
                ctx.request_handler,
                ui_index.read_text(encoding="utf-8", errors="replace"),
                content_type="text/html; charset=utf-8",
            )
            return
        send_text(ctx.request_handler, _fallback_index_html(config), content_type="text/html; charset=utf-8")

    def serve_static(ctx: RouteContext) -> None:
        rel = (ctx.params.get("path") or "").strip()
        _serve_ui_asset(ctx, rel)

    def health(ctx: RouteContext) -> None:
        send_json(ctx.request_handler, _health_payload(config))

    def config_view(ctx: RouteContext) -> None:
        send_json(ctx.request_handler, _config_payload(config))

    def logs_recent(ctx: RouteContext) -> None:
        limit = max(1, min(500, _safe_int(ctx.query_first("limit", "50"), 50)))
        send_json(ctx.request_handler, {"ok": True, "items": read_recent(config=config, limit=limit)})

    def whisper_status(ctx: RouteContext) -> None:
        send_json(ctx.request_handler, {"ok": True, **whisper_get_status()})

    def rag_status(ctx: RouteContext) -> None:
        payload = rag_get_status(config=config)
        ratings_file = config.base_dir / "rag_ratings.jsonl"
        payload["ratings_file"] = str(ratings_file)
        payload["ratings_count"] = _count_lines(ratings_file)
        send_json(ctx.request_handler, payload)

    def rag_query(ctx: RouteContext) -> None:
        findings = ctx.query_first("findings", "").strip()
        modality = ctx.query_first("modality", "").strip() or None
        k = max(1, min(10, _safe_int(ctx.query_first("k", "3"), 3)))
        items = query_records(config=config, findings=findings, modality=modality, k=k) if findings else []
        send_json(ctx.request_handler, {"ok": True, "items": items, "count": len(items)})

    def rag_examples(ctx: RouteContext) -> None:
        findings = ctx.query_first("findings", "").strip()
        modality = ctx.query_first("modality", "").strip() or None
        k = max(1, min(10, _safe_int(ctx.query_first("k", "3"), 3)))
        items = query_records(config=config, findings=findings, modality=modality, k=k) if findings else []
        send_json(
            ctx.request_handler,
            {
                "ok": True,
                "items": items,
                "count": len(items),
                "examples": _format_examples_block(items),
            },
        )

    def benchmark_datasets(ctx: RouteContext) -> None:
        send_json(ctx.request_handler, {"ok": True, "items": _list_benchmark_datasets(config.data_dir)})

    def benchmark_load_path_get(ctx: RouteContext) -> None:
        source_path = ctx.query_first("path", "").strip()
        if not source_path:
            send_json(ctx.request_handler, {"ok": False, "error": "Missing query parameter: path"}, status=400)
            return
        send_json(ctx.request_handler, _benchmark_load_path(config, source_path))

    def rag_rate(ctx: RouteContext) -> None:
        payload = _coerce_object(ctx.read_json())
        rating = str(payload.get("rating", "")).strip().lower()
        line = str(payload.get("line", "")).strip()
        findings = str(payload.get("findings", "")).strip()
        modality = str(payload.get("modality", "unknown") or "unknown").strip() or "unknown"
        template = str(payload.get("template", "") or "").strip()

        if rating not in {"good", "bad", "up", "down", "thumbs_up", "thumbs_down"}:
            send_json(ctx.request_handler, {"ok": False, "error": "rating must be good/bad or equivalent"}, status=400)
            return

        ratings_file = config.base_dir / "rag_ratings.jsonl"
        record = {
            "type": "RAG_RATING",
            "rating": rating,
            "line": line,
            "findings": findings,
            "template": template,
            "modality": modality,
            "context": _coerce_object(payload.get("context", {})),
        }
        log_event(
            "RAG_RATING",
            detail=f"rating={rating} template={template}",
            config=config,
            path=ratings_file,
            source="server.app",
            route_path=ctx.path,
            context=record,
        )

        added = False
        if rating in {"good", "up", "thumbs_up"} and line and findings:
            added = add_record(
                {
                    "findings": findings,
                    "impression": line,
                    "modality": modality,
                    "source": "user_rating",
                },
                config=config,
            )

        send_json(
            ctx.request_handler,
            {
                "ok": True,
                "saved": True,
                "added_to_library": added,
                "ratings_file": str(ratings_file),
            },
        )

    def rag_train(ctx: RouteContext) -> None:
        payload = _coerce_object(ctx.read_json())
        source_path = str(payload.get("path", "") or "").strip()
        if not source_path:
            send_json(ctx.request_handler, {"ok": False, "error": "Missing JSON field: path"}, status=400)
            return
        result = train_path(config=config, path=source_path)
        send_json(ctx.request_handler, {"ok": True, **result})

    def benchmark_load(ctx: RouteContext) -> None:
        send_json(
            ctx.request_handler,
            {
                "ok": False,
                "error": "Benchmark upload parsing is not implemented yet in the modular server.",
                "hint": "Use GET/POST /benchmark/load-path with a local file path for now.",
            },
            status=501,
        )

    def benchmark_load_path_post(ctx: RouteContext) -> None:
        payload = _coerce_object(ctx.read_json())
        source_path = str(payload.get("path", "") or "").strip()
        if not source_path:
            send_json(ctx.request_handler, {"ok": False, "error": "Missing JSON field: path"}, status=400)
            return
        send_json(ctx.request_handler, _benchmark_load_path(config, source_path))

    def whisper_transcribe(ctx: RouteContext) -> None:
        result = transcribe_request(config, ctx.request_handler)
        send_json(ctx.request_handler, result, status=200 if result.get("ok") else 400)

    def ollama_proxy_get(ctx: RouteContext) -> None:
        proxy_route_handler(ctx)

    def ollama_proxy_post(ctx: RouteContext) -> None:
        proxy_route_handler(ctx)

    registry = _build_registry(
        config,
        CoreRouteHandlers(
            serve_index=serve_index,
            serve_static=serve_static,
            health=health,
            config_view=config_view,
            logs_recent=logs_recent,
            whisper_status=whisper_status,
            rag_status=rag_status,
            rag_query=rag_query,
            rag_examples=rag_examples,
            benchmark_datasets=benchmark_datasets,
            benchmark_load_path_get=benchmark_load_path_get,
            rag_rate=rag_rate,
            rag_train=rag_train,
            benchmark_load=benchmark_load,
            benchmark_load_path_post=benchmark_load_path_post,
            whisper_transcribe=whisper_transcribe,
            ollama_proxy_get=ollama_proxy_get,
            ollama_proxy_post=ollama_proxy_post,
        ),
    )

    class Handler(BaseHTTPRequestHandler):
        server_version = "RadCopilotLocal/0.4"

        def log_message(self, fmt: str, *args: object) -> None:  # noqa: A003
            try:
                log_event(
                    "HTTP",
                    detail=(fmt % args)[:1000],
                    config=config,
                    source="server.app",
                    route_path=self.path,
                )
            except Exception:
                return

        def _dispatch(self, method: str) -> None:
            try:
                dispatch_request(
                    registry=registry,
                    config=config,
                    request_handler=self,
                    method=method,
                    extras={"timeout": 300.0},
                )
            except RouteNotFound:
                if method == "GET" and _try_serve_root_ui_asset(self):
                    return
                send_json(self, {"ok": False, "error": "Not Found", "path": self.path}, status=404)
            except MethodNotAllowed as exc:
                send_json(self, {"ok": False, "error": str(exc), "path": self.path}, status=405)
            except json.JSONDecodeError:
                send_json(self, {"ok": False, "error": "Invalid JSON body", "path": self.path}, status=400)
            except Exception as exc:  # pragma: no cover - defensive HTTP path
                try:
                    log_exception(
                        exc,
                        type="ROUTE_EXCEPTION",
                        detail="Unhandled route exception",
                        config=config,
                        source="server.app",
                        route_path=self.path,
                        context={"method": method},
                    )
                except Exception:
                    pass
                send_json(
                    self,
                    {
                        "ok": False,
                        "error": f"{type(exc).__name__}: {exc}",
                        "path": self.path,
                        "trace": traceback.format_exc()[-4000:],
                    },
                    status=500,
                )

        def do_GET(self) -> None:  # noqa: N802
            self._dispatch("GET")

        def do_POST(self) -> None:  # noqa: N802
            self._dispatch("POST")

        def do_PUT(self) -> None:  # noqa: N802
            self._dispatch("PUT")

        def do_PATCH(self) -> None:  # noqa: N802
            self._dispatch("PATCH")

        def do_DELETE(self) -> None:  # noqa: N802
            self._dispatch("DELETE")

    def _try_serve_root_ui_asset(handler: BaseHTTPRequestHandler) -> bool:
        path = _normalize_root_path(handler.path)
        if path in {"", "/", "/index.html"}:
            return False
        rel = path.lstrip("/")
        return _serve_ui_asset_direct(handler, rel)

    return Handler


def _build_registry(config: ConfigLike, handlers: CoreRouteHandlers) -> RouteRegistry:
    registry = build_core_registry(handlers)
    register_report_routes(registry)
    # Support full proxy forwarding, not just GET/POST.
    registry.put("/api/*", proxy_route_handler, name="ollama-proxy-put", tags=("ollama", "proxy"))
    registry.patch("/api/*", proxy_route_handler, name="ollama-proxy-patch", tags=("ollama", "proxy"))
    registry.delete("/api/*", proxy_route_handler, name="ollama-proxy-delete", tags=("ollama", "proxy"))

    def routes_view(ctx: RouteContext) -> None:
        send_json(ctx.request_handler, {"ok": True, "items": registry.describe()})

    registry.get("/routes", routes_view, name="routes", description="Describe registered routes.", tags=("system",))
    return registry


def _health_payload(config: ConfigLike) -> dict[str, Any]:
    return {
        "ok": True,
        "app": config.app_name,
        "base_url": config.base_url,
        "ollama_url": config.ollama_url,
        "ui_dir": str(config.ui_dir),
        "data_dir": str(config.data_dir),
        "log_file": str(config.log_file),
    }


def _config_payload(config: ConfigLike) -> dict[str, Any]:
    return {
        "app_name": config.app_name,
        "host": config.host,
        "port": config.port,
        "base_url": config.base_url,
        "ollama_url": config.ollama_url,
        "ui_dir": str(config.ui_dir),
        "data_dir": str(config.data_dir),
        "log_file": str(config.log_file),
    }


def _serve_ui_asset(ctx: RouteContext, rel: str) -> None:
    handler = ctx.request_handler
    if not _serve_ui_asset_direct(handler, rel, ui_dir=ctx.config.ui_dir):
        send_json(handler, {"ok": False, "error": "Not Found", "path": ctx.raw_path}, status=404)


def _serve_ui_asset_direct(handler: BaseHTTPRequestHandler, rel: str, ui_dir: Path | None = None) -> bool:
    if ui_dir is None:
        return False
    rel = str(rel or "").strip().lstrip("/")
    if not rel:
        return False

    target = (ui_dir / rel).resolve()
    try:
        target.relative_to(ui_dir.resolve())
    except ValueError:
        send_json(handler, {"ok": False, "error": "Invalid static path", "path": handler.path}, status=400)
        return True

    if not target.exists() or not target.is_file():
        return False

    mime, _ = mimetypes.guess_type(str(target))
    send_bytes(handler, target.read_bytes(), content_type=mime or "application/octet-stream")
    return True


def _list_benchmark_datasets(data_dir: Path) -> list[dict[str, Any]]:
    root = Path(data_dir).expanduser().resolve()
    if not root.exists() or not root.is_dir():
        return []

    items: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        if suffix not in {".txt", ".csv", ".xml", ".json", ".jsonl", ".tgz", ".tar", ".gz"}:
            continue
        try:
            rel = str(path.relative_to(root))
        except ValueError:
            rel = path.name
        items.append(
            {
                "name": path.name,
                "relative_path": rel,
                "path": str(path),
                "size_bytes": path.stat().st_size,
                "suffix": suffix,
            }
        )
    return items


def _benchmark_load_path(config: ConfigLike, source_path: str) -> dict[str, Any]:
    path = Path(source_path).expanduser().resolve()
    if not path.exists():
        return {"ok": False, "error": f"Path does not exist: {path}"}

    if path.is_dir():
        files = [p for p in sorted(path.rglob("*")) if p.is_file()]
        return {
            "ok": True,
            "path": str(path),
            "kind": "directory",
            "file_count": len(files),
            "items": [{"name": p.name, "path": str(p), "size_bytes": p.stat().st_size} for p in files[:200]],
        }

    preview = ""
    suffix = path.suffix.lower()
    if suffix in {".txt", ".md", ".json", ".jsonl", ".csv", ".xml"}:
        try:
            preview = path.read_text(encoding="utf-8", errors="replace")[:4000]
        except Exception:
            preview = ""

    return {
        "ok": True,
        "path": str(path),
        "kind": "file",
        "name": path.name,
        "suffix": suffix,
        "size_bytes": path.stat().st_size,
        "preview": preview,
    }


def _format_examples_block(items: list[dict[str, Any]]) -> str:
    if not items:
        return ""
    lines: list[str] = []
    for idx, item in enumerate(items, start=1):
        impression = str(item.get("impression", "") or "").strip()
        findings = str(item.get("findings", "") or "").strip()
        score = item.get("score")
        prefix = f"Example {idx}"
        if score is not None:
            prefix += f" (score={score})"
        lines.append(prefix)
        if findings:
            lines.append(f"Findings: {findings}")
        if impression:
            lines.append(f"Impression: {impression}")
        lines.append("")
    return "\n".join(lines).strip()


def _fallback_index_html(config: ConfigLike) -> str:
    return f"""<!DOCTYPE html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\">
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">
  <title>{config.app_name}</title>
  <style>
    body {{ font-family: Arial, sans-serif; background: #0b1220; color: #e5eef8; margin: 0; display:grid; place-items:center; min-height:100vh; }}
    .card {{ width:min(880px,92vw); background:#101a2b; border:1px solid #1f3150; border-radius:18px; padding:28px; }}
    code {{ background:#0b1220; padding:2px 6px; border-radius:6px; }}
    a {{ color:#7dd3fc; }}
  </style>
</head>
<body>
  <main class=\"card\">
    <h1>{config.app_name}</h1>
    <p>The modular server is running. The UI fallback page is being served because <code>ui/index.html</code> was not found.</p>
    <ul>
      <li>Base URL: <code>{config.base_url}</code></li>
      <li>Ollama URL: <code>{config.ollama_url}</code></li>
      <li>Log file: <code>{config.log_file}</code></li>
      <li>Routes: <a href=\"/routes\">/routes</a></li>
      <li>Health: <a href=\"/health\">/health</a></li>
    </ul>
  </main>
</body>
</html>"""


def _normalize_root_path(raw_path: str) -> str:
    text = str(raw_path or "").split("?", 1)[0]
    if not text.startswith("/"):
        text = "/" + text
    return text


def _safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _count_lines(path: Path) -> int:
    try:
        if not path.exists() or not path.is_file():
            return 0
        with path.open("r", encoding="utf-8", errors="replace") as fh:
            return sum(1 for _ in fh)
    except Exception:
        return 0


def _coerce_object(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    return {}
