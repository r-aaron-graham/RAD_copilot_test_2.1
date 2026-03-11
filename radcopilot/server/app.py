from __future__ import annotations

"""
radcopilot.server.app

Modular HTTP server for RadCopilot Local.

This file is the first real server module in the refactor from the original
single-file application into a modular local package. It is designed to work
immediately with the previously created `main.py` entrypoint.

Current goals:
- provide a production-quality local HTTP handler factory
- serve the UI when available
- expose health/config/log routes
- expose RAG utility routes
- expose working benchmark dataset loading and scoring routes
- expose working report routes
- expose working history routes
- proxy `/api/*` requests to local Ollama
- keep optional feature routes available even before every service module exists
- fail gracefully with explicit JSON errors instead of crashing

This module intentionally uses only the Python standard library so it can run
before the rest of the refactor is complete.
"""

import base64
import json
import mimetypes
from pathlib import Path
import traceback
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler
from typing import Any, Protocol, Type


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
    """
    Create a request handler bound to the supplied runtime config.

    This shape matches the expectation in `main.py`:
        from radcopilot.server.app import create_handler
        handler = create_handler(config)
    """

    class Handler(BaseHTTPRequestHandler):
        server_version = "RadCopilotLocal/0.4"

        # -----------------------------
        # Generic helpers
        # -----------------------------
        def log_message(self, fmt: str, *args: object) -> None:  # noqa: A003 - stdlib signature
            _append_jsonl(
                config.log_file,
                {
                    "ts": _utc_now(),
                    "type": "HTTP",
                    "path": self.path,
                    "detail": (fmt % args)[:1000],
                },
            )

        def _json(self, payload: Any, status: int = 200) -> None:
            body = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _text(self, text: str, status: int = 200, content_type: str = "text/plain; charset=utf-8") -> None:
            body = text.encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _bytes(self, data: bytes, status: int = 200, content_type: str = "application/octet-stream") -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _read_json_body(self) -> dict[str, Any]:
            length = int(self.headers.get("Content-Length", "0") or "0")
            if length <= 0:
                return {}
            raw = self.rfile.read(length)
            if not raw:
                return {}
            return json.loads(raw.decode("utf-8"))

        def _query(self) -> dict[str, list[str]]:
            return urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)

        def _query_first(self, key: str, default: str = "") -> str:
            return self._query().get(key, [default])[0]

        def _send_not_found(self) -> None:
            self._json({"ok": False, "error": "Not Found", "path": self.path}, status=404)

        # -----------------------------
        # GET routes
        # -----------------------------
        def do_GET(self) -> None:  # noqa: N802 - stdlib handler name
            try:
                parsed = urllib.parse.urlparse(self.path)
                path = parsed.path

                if path in {"/", "/index.html"}:
                    self._serve_index()
                    return

                if path.startswith("/static/"):
                    self._serve_static(path)
                    return

                if path == "/health":
                    self._json(_health_payload(config))
                    return

                if path == "/config":
                    self._json(_config_payload(config))
                    return

                if path == "/logs/recent":
                    limit = max(1, min(500, _safe_int(self._query_first("limit", "50"), 50)))
                    self._json({"ok": True, "items": _tail_jsonl(config.log_file, limit)})
                    return

                if path == "/whisper/status":
                    self._json({"ok": True, "available": _whisper_available()})
                    return

                if path == "/rag/status":
                    self._json(_rag_status_payload(config))
                    return

                if path == "/rag/query":
                    findings = self._query_first("findings", "").strip()
                    modality = self._query_first("modality", "").strip() or None
                    k = max(1, min(10, _safe_int(self._query_first("k", "3"), 3)))
                    self._json(_rag_query_payload(config, findings=findings, modality=modality, k=k))
                    return

                if path == "/rag/examples":
                    findings = self._query_first("findings", "").strip()
                    modality = self._query_first("modality", "").strip() or None
                    k = max(1, min(10, _safe_int(self._query_first("k", "3"), 3)))
                    data = _rag_query_payload(config, findings=findings, modality=modality, k=k)
                    examples_text = _format_examples_block(data.get("items", []))
                    self._json({"ok": True, "examples": examples_text, "items": data.get("items", [])})
                    return

                if path == "/benchmark/datasets":
                    self._json({"ok": True, "items": _list_benchmark_datasets(config)})
                    return

                if path == "/benchmark/load-path":
                    source_path = self._query_first("path", "").strip()
                    limit = max(1, min(25_000, _safe_int(self._query_first("limit", "1000"), 1000)))
                    if not source_path:
                        self._json({"ok": False, "error": "Missing query parameter: path"}, status=400)
                        return
                    self._json(_benchmark_load_path(config, source_path, limit=limit))
                    return

                if path == "/report/templates":
                    self._json(_report_templates())
                    return

                if path.startswith("/report/templates/"):
                    template_id = path.removeprefix("/report/templates/").strip("/")
                    self._json(_report_template_detail(template_id))
                    return

                if path == "/history":
                    limit = max(1, min(500, _safe_int(self._query_first("limit", "50"), 50)))
                    offset = max(0, _safe_int(self._query_first("offset", "0"), 0))
                    query = self._query_first("q", "").strip()
                    self._json(_history_list(config, limit=limit, offset=offset, query=query))
                    return

                if path.startswith("/history/") and not path.startswith("/history/search"):
                    entry_id = path.removeprefix("/history/").strip("/")
                    self._json(_history_get(config, entry_id))
                    return

                if path == "/history/search":
                    query = self._query_first("q", "").strip()
                    limit = max(1, min(500, _safe_int(self._query_first("limit", "20"), 20)))
                    self._json(_history_list(config, limit=limit, query=query))
                    return

                if path.startswith("/api/"):
                    self._proxy_to_ollama("GET")
                    return

                self._send_not_found()
            except Exception as exc:  # pragma: no cover
                self._handle_route_exception(exc)

        # -----------------------------
        # POST routes
        # -----------------------------
        def do_POST(self) -> None:  # noqa: N802 - stdlib handler name
            try:
                parsed = urllib.parse.urlparse(self.path)
                path = parsed.path

                if path == "/rag/rate":
                    payload = self._read_json_body()
                    self._json(_rag_rate(config, payload))
                    return

                if path == "/rag/train":
                    payload = self._read_json_body()
                    self._json(_rag_train(config, payload))
                    return

                if path == "/benchmark/load":
                    payload = self._read_json_body()
                    self._json(_benchmark_load_upload(config, payload))
                    return

                if path == "/benchmark/load-path":
                    payload = self._read_json_body()
                    source_path = str(payload.get("path", "")).strip()
                    limit = max(1, min(25_000, _safe_int(str(payload.get("limit", 1000)), 1000)))
                    if not source_path:
                        self._json({"ok": False, "error": "Missing JSON field: path"}, status=400)
                        return
                    self._json(_benchmark_load_path(config, source_path, limit=limit))
                    return

                if path == "/benchmark/score":
                    payload = self._read_json_body()
                    self._json(_benchmark_score(config, payload))
                    return

                if path == "/whisper/transcribe":
                    self._json(_whisper_transcribe(config, self))
                    return

                if path == "/report/generate":
                    payload = self._read_json_body()
                    self._json(_report_generate(config, payload))
                    return

                if path == "/report/guidelines":
                    payload = self._read_json_body()
                    self._json(_report_guidelines(config, payload))
                    return

                if path == "/report/validate":
                    payload = self._read_json_body()
                    self._json(_report_validate(config, payload))
                    return

                if path == "/history":
                    payload = self._read_json_body()
                    self._json(_history_append(config, payload))
                    return

                if path.startswith("/history/") and path.endswith("/star"):
                    entry_id = path.removeprefix("/history/").removesuffix("/star").strip("/")
                    self._json(_history_toggle_star(config, entry_id))
                    return

                if path.startswith("/api/"):
                    self._proxy_to_ollama("POST")
                    return

                self._send_not_found()
            except json.JSONDecodeError:
                self._json({"ok": False, "error": "Invalid JSON body", "path": self.path}, status=400)
            except Exception as exc:  # pragma: no cover
                self._handle_route_exception(exc)

        def do_DELETE(self) -> None:  # noqa: N802 - stdlib handler name
            try:
                parsed = urllib.parse.urlparse(self.path)
                path = parsed.path

                if path.startswith("/history/"):
                    entry_id = path.removeprefix("/history/").strip("/")
                    self._json(_history_delete(config, entry_id))
                    return

                self._send_not_found()
            except Exception as exc:  # pragma: no cover
                self._handle_route_exception(exc)

        # -----------------------------
        # Route implementations
        # -----------------------------
        def _serve_index(self) -> None:
            ui_index = config.ui_dir / "index.html"
            if ui_index.exists() and ui_index.is_file():
                self._text(ui_index.read_text(encoding="utf-8"), content_type="text/html; charset=utf-8")
                return
            self._text(_fallback_index_html(config), content_type="text/html; charset=utf-8")

        def _serve_static(self, path: str) -> None:
            rel = path.removeprefix("/static/").strip("/")
            target = (config.ui_dir / rel).resolve()
            try:
                target.relative_to(config.ui_dir.resolve())
            except ValueError:
                self._json({"ok": False, "error": "Invalid static path"}, status=400)
                return

            if not target.exists() or not target.is_file():
                self._send_not_found()
                return

            mime, _ = mimetypes.guess_type(str(target))
            content_type = mime or "application/octet-stream"
            if content_type.startswith("text/"):
                content_type = f"{content_type}; charset=utf-8"
            self._bytes(target.read_bytes(), content_type=content_type)

        def _proxy_to_ollama(self, method: str) -> None:
            # Note: proxy.py remains a richer streaming-capable implementation.
            # This inline proxy is kept for minimal local-server compatibility.
            parsed = urllib.parse.urlparse(self.path)
            target_url = f"{config.ollama_url}{parsed.path}"
            if parsed.query:
                target_url = f"{target_url}?{parsed.query}"

            body: bytes | None = None
            if method in {"POST", "PUT", "PATCH"}:
                length = int(self.headers.get("Content-Length", "0") or "0")
                body = self.rfile.read(length) if length > 0 else b""

            headers: dict[str, str] = {}
            if self.headers.get("Content-Type"):
                headers["Content-Type"] = str(self.headers["Content-Type"])
            if self.headers.get("Accept"):
                headers["Accept"] = str(self.headers["Accept"])

            req = urllib.request.Request(target_url, data=body, method=method, headers=headers)
            try:
                with urllib.request.urlopen(req, timeout=300) as resp:
                    payload = resp.read()
                    self.send_response(resp.status)
                    resp_content_type = resp.headers.get("Content-Type", "application/json; charset=utf-8")
                    self.send_header("Content-Type", resp_content_type)
                    self.send_header("Content-Length", str(len(payload)))
                    self.end_headers()
                    self.wfile.write(payload)
            except urllib.error.HTTPError as exc:
                payload = exc.read() or json.dumps({"ok": False, "error": str(exc)}).encode("utf-8")
                self.send_response(exc.code)
                self.send_header("Content-Type", exc.headers.get("Content-Type", "application/json; charset=utf-8"))
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
            except Exception as exc:
                self._json(
                    {
                        "ok": False,
                        "error": f"Ollama proxy failure: {type(exc).__name__}: {exc}",
                        "target_url": target_url,
                    },
                    status=502,
                )

        def _handle_route_exception(self, exc: Exception) -> None:
            _append_jsonl(
                config.log_file,
                {
                    "ts": _utc_now(),
                    "type": "ROUTE_EXCEPTION",
                    "path": self.path,
                    "detail": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc()[-8000:],
                },
            )
            self._json(
                {"ok": False, "error": f"{type(exc).__name__}: {exc}", "path": self.path},
                status=500,
            )

    return Handler


# -----------------------------
# Payload builders / service hooks
# -----------------------------


def _health_payload(config: ConfigLike) -> dict[str, Any]:
    return {
        "ok": True,
        "app": config.app_name,
        "base_url": config.base_url,
        "ollama_url": config.ollama_url,
        "ollama_up": _ollama_available(config.ollama_url),
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


def _rag_status_payload(config: ConfigLike) -> dict[str, Any]:
    library_file = config.base_dir / "rag_library.json"
    ratings_file = config.base_dir / "rag_ratings.jsonl"
    items = _read_json_file(library_file, default=[])
    if not isinstance(items, list):
        items = []

    modality_counts: dict[str, int] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        mod = str(item.get("modality") or "unknown")
        modality_counts[mod] = modality_counts.get(mod, 0) + 1

    ratings_count = _count_lines(ratings_file)
    return {
        "ok": True,
        "library_file": str(library_file),
        "ratings_file": str(ratings_file),
        "count": len(items),
        "ratings_count": ratings_count,
        "modalities": modality_counts,
    }


def _rag_query_payload(config: ConfigLike, *, findings: str, modality: str | None, k: int) -> dict[str, Any]:
    if not findings:
        return {"ok": True, "items": [], "count": 0}

    try:
        from radcopilot.rag.library import query_records  # type: ignore

        items = query_records(config=config, findings=findings, modality=modality, k=k)
        return {"ok": True, "items": items, "count": len(items), "source": "module"}
    except ModuleNotFoundError:
        pass
    except Exception as exc:
        return {
            "ok": False,
            "error": f"RAG query module error: {type(exc).__name__}: {exc}",
            "items": [],
            "count": 0,
        }

    library_file = config.base_dir / "rag_library.json"
    items = _read_json_file(library_file, default=[])
    if not isinstance(items, list):
        items = []

    findings_tokens = _tokenize(findings)
    scored: list[tuple[float, dict[str, Any]]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        if modality and item.get("modality") not in {modality, None, "", "unknown"}:
            continue
        item_findings = str(item.get("findings", ""))
        score = _jaccard(findings_tokens, _tokenize(item_findings))
        if score <= 0:
            continue
        scored.append((score, item))

    scored.sort(key=lambda x: x[0], reverse=True)
    best = [
        {
            "score": round(score, 4),
            "findings": item.get("findings", ""),
            "impression": item.get("impression", ""),
            "modality": item.get("modality", "unknown"),
        }
        for score, item in scored[:k]
    ]
    return {"ok": True, "items": best, "count": len(best), "source": "json-fallback"}


def _rag_rate(config: ConfigLike, payload: dict[str, Any]) -> dict[str, Any]:
    entry = {
        "ts": _utc_now(),
        "rating": str(payload.get("rating", "")).strip().lower(),
        "line": str(payload.get("line", "")).strip(),
        "findings": str(payload.get("findings", "")).strip(),
        "template": str(payload.get("template", "")).strip(),
        "context": payload.get("context", {}),
    }

    if entry["rating"] not in {"good", "bad", "up", "down", "thumbs_up", "thumbs_down"}:
        return {"ok": False, "error": "rating must be good/bad or equivalent"}

    ratings_file = config.base_dir / "rag_ratings.jsonl"
    _append_jsonl(ratings_file, entry)

    added_to_library = False
    if entry["rating"] in {"good", "up", "thumbs_up"} and entry["findings"] and entry["line"]:
        added_to_library = _add_rag_record(
            config,
            {
                "findings": entry["findings"],
                "impression": entry["line"],
                "modality": payload.get("modality") or "unknown",
                "source": "user_rating",
                "created_at": entry["ts"],
            },
        )

    return {
        "ok": True,
        "saved": True,
        "added_to_library": added_to_library,
        "ratings_file": str(ratings_file),
    }


def _rag_train(config: ConfigLike, payload: dict[str, Any]) -> dict[str, Any]:
    source_path = str(payload.get("path", "")).strip()
    if not source_path:
        return {"ok": False, "error": "Missing JSON field: path"}

    try:
        from radcopilot.rag.trainer import train_path  # type: ignore

        result = train_path(config=config, path=source_path)
        return {"ok": True, "source": "module", **result}
    except ModuleNotFoundError:
        pass
    except Exception as exc:
        return {"ok": False, "error": f"RAG trainer module error: {type(exc).__name__}: {exc}"}

    path = Path(source_path).expanduser().resolve()
    if not path.exists():
        return {"ok": False, "error": f"Path does not exist: {path}"}

    records: list[dict[str, Any]] = []
    if path.is_dir():
        for child in sorted(path.rglob("*.txt")):
            rec = _parse_txt_case(child.read_text(encoding="utf-8", errors="replace"))
            if rec:
                rec["source_file"] = str(child)
                records.append(rec)
    elif path.suffix.lower() == ".txt":
        rec = _parse_txt_case(path.read_text(encoding="utf-8", errors="replace"))
        if rec:
            rec["source_file"] = str(path)
            records.append(rec)
    elif path.suffix.lower() == ".csv":
        records.extend(_parse_csv_cases(path))
    else:
        return {
            "ok": False,
            "error": "Fallback trainer currently supports only .txt, .csv, or directories of .txt files",
        }

    added = 0
    for record in records:
        if _add_rag_record(config, record):
            added += 1

    return {
        "ok": True,
        "source": "fallback",
        "path": str(path),
        "records_seen": len(records),
        "records_added": added,
    }


def _whisper_transcribe(config: ConfigLike, handler: BaseHTTPRequestHandler) -> dict[str, Any]:
    try:
        from radcopilot.services.whisper_service import transcribe_request  # type: ignore

        return transcribe_request(config=config, handler=handler)
    except ModuleNotFoundError:
        pass
    except Exception as exc:
        return {"ok": False, "error": f"Whisper service module error: {type(exc).__name__}: {exc}"}

    return {
        "ok": False,
        "error": "Whisper transcription is not implemented yet in the modular server",
        "available": _whisper_available(),
    }


def _list_benchmark_datasets(config: ConfigLike) -> list[dict[str, Any]]:
    try:
        list_fn = _get_benchmark_loader_symbol("list_benchmark_datasets")
        items = list_fn(config=config)
        return items if isinstance(items, list) else []
    except Exception:
        pass

    data_dir = config.data_dir
    if not data_dir.exists() or not data_dir.is_dir():
        return []

    items: list[dict[str, Any]] = []
    for path in sorted(data_dir.rglob("*")):
        if not path.is_file():
            continue
        try:
            rel = path.relative_to(data_dir)
        except ValueError:
            rel = path
        items.append(
            {
                "name": path.name,
                "relative_path": str(rel),
                "path": str(path),
                "suffix": path.suffix.lower(),
                "size_bytes": path.stat().st_size,
            }
        )
    return items


def _benchmark_load_path(config: ConfigLike, source_path: str, *, limit: int = 1000) -> dict[str, Any]:
    try:
        load_path_fn = _get_benchmark_loader_symbol("load_path")
        return load_path_fn(config=config, path=source_path, limit=limit)
    except Exception as exc:
        path = Path(source_path).expanduser().resolve()
        if not path.exists():
            return {"ok": False, "error": f"Path does not exist: {path}"}

        if path.is_dir():
            files = [str(p) for p in sorted(path.iterdir()) if p.is_file()]
            return {
                "ok": True,
                "path": str(path),
                "kind": "directory",
                "files": files[:200],
                "count": len(files),
                "warning": f"Benchmark loader unavailable, falling back to directory listing: {type(exc).__name__}: {exc}",
            }

        stat = path.stat()
        preview = ""
        if path.suffix.lower() in {".txt", ".md", ".json", ".jsonl", ".csv", ".xml"}:
            preview = path.read_text(encoding="utf-8", errors="replace")[:4000]

        return {
            "ok": True,
            "path": str(path),
            "kind": "file",
            "name": path.name,
            "suffix": path.suffix.lower(),
            "size_bytes": stat.st_size,
            "preview": preview,
            "warning": f"Benchmark loader unavailable, falling back to file preview: {type(exc).__name__}: {exc}",
        }


def _benchmark_load_upload(config: ConfigLike, payload: dict[str, Any]) -> dict[str, Any]:
    filename = str(payload.get("filename", "")).strip() or "upload.txt"
    limit = max(1, min(25_000, _safe_int(str(payload.get("limit", 1000)), 1000)))

    raw_data = payload.get("content_base64")
    if raw_data:
        try:
            data = base64.b64decode(str(raw_data), validate=True)
        except Exception as exc:
            return {"ok": False, "error": f"Invalid content_base64 payload: {type(exc).__name__}: {exc}"}
    else:
        content = payload.get("content", "")
        if isinstance(content, str):
            data = content.encode("utf-8")
        else:
            return {"ok": False, "error": "Missing content or content_base64"}

    try:
        load_bytes_fn = _get_benchmark_loader_symbol("load_bytes")
        return load_bytes_fn(config=config, filename=filename, data=data, limit=limit)
    except Exception as exc:
        return {
            "ok": False,
            "error": f"Benchmark upload parsing is unavailable: {type(exc).__name__}: {exc}",
        }


def _benchmark_score(config: ConfigLike, payload: dict[str, Any]) -> dict[str, Any]:
    try:
        score_cases_fn = _get_benchmark_scorer_symbol("score_cases")
    except Exception as exc:
        return {"ok": False, "error": f"Benchmark scorer unavailable: {type(exc).__name__}: {exc}"}

    cases: list[dict[str, Any]] = []
    if isinstance(payload.get("cases"), list):
        cases = [dict(item) for item in payload["cases"] if isinstance(item, dict)]
    elif isinstance(payload.get("items"), list):
        cases = [dict(item) for item in payload["items"] if isinstance(item, dict)]
    else:
        source_path = str(payload.get("path", "")).strip()
        if source_path:
            load_result = _benchmark_load_path(
                config,
                source_path,
                limit=max(1, min(25_000, _safe_int(str(payload.get("limit", 1000)), 1000))),
            )
            if not load_result.get("ok"):
                return load_result
            loaded_items = load_result.get("items", [])
            if isinstance(loaded_items, list):
                cases = [dict(item) for item in loaded_items if isinstance(item, dict)]

    if not cases:
        return {"ok": False, "error": "No benchmark cases available to score"}

    prepared_cases = [_normalize_benchmark_case_for_scoring(case) for case in cases]
    predictions = payload.get("predictions")
    pass_threshold = _safe_float(payload.get("pass_threshold"), 0.72)
    strict_count = bool(payload.get("strict_count", False))
    max_cases_default = len(prepared_cases) if prepared_cases else 1
    max_cases = max(1, min(50_000, _safe_int(str(payload.get("max_cases", max_cases_default)), max_cases_default)))

    result = score_cases_fn(
        cases=prepared_cases,
        predictions=predictions,
        pass_threshold=pass_threshold,
        strict_count=strict_count,
        max_cases=max_cases,
    )
    if isinstance(result, dict):
        return result
    return {"ok": False, "error": "Benchmark scorer returned an unexpected result type"}


# -----------------------------
# Report route payload builders
# -----------------------------


def _report_templates() -> dict[str, Any]:
    try:
        from radcopilot.report.generator import list_templates  # type: ignore

        return {"ok": True, "default_template_id": "ct-chest", "items": list_templates()}
    except ModuleNotFoundError:
        return {"ok": False, "error": "Report generator module not available"}
    except Exception as exc:
        return {"ok": False, "error": f"Failed to list templates: {type(exc).__name__}: {exc}"}


def _report_template_detail(template_id: str) -> dict[str, Any]:
    try:
        from radcopilot.report.generator import get_template  # type: ignore

        template = get_template(template_id or None)
        return {
            "ok": True,
            "requested_template_id": template_id,
            "item": {
                "id": template.id,
                "label": template.label,
                "modality": template.modality,
                "section_order": list(template.section_order),
                "section_labels": dict(template.section_labels),
                "section_defaults": dict(template.section_defaults),
                "guideline_hint": template.guideline_hint,
                "allow_negatives_in_impression": bool(template.allow_negatives_in_impression),
            },
        }
    except ModuleNotFoundError:
        return {"ok": False, "error": "Report generator module not available"}
    except Exception as exc:
        return {"ok": False, "error": f"Failed to get template: {type(exc).__name__}: {exc}"}


def _report_generate(config: ConfigLike, payload: dict[str, Any]) -> dict[str, Any]:
    try:
        from radcopilot.report.generator import ReportRequest, generate_report  # type: ignore

        request = ReportRequest.from_mapping(payload)
        result = generate_report(request, config=config)
        response: dict[str, Any] = result.to_dict()
        response.setdefault("ok", result.ok)
        return response
    except ModuleNotFoundError:
        return {"ok": False, "error": "Report generator module not available"}
    except Exception as exc:
        return {"ok": False, "error": f"Report generation failed: {type(exc).__name__}: {exc}"}


def _report_guidelines(config: ConfigLike, payload: dict[str, Any]) -> dict[str, Any]:
    try:
        from radcopilot.report.guidelines import generate_guideline_text  # type: ignore

        use_model = bool(payload.pop("use_model", True))
        result = generate_guideline_text(config=config, context=payload, use_model=use_model)
        return result.to_dict()
    except ModuleNotFoundError:
        return {"ok": False, "error": "Report guidelines module not available"}
    except Exception as exc:
        return {"ok": False, "error": f"Guideline generation failed: {type(exc).__name__}: {exc}"}


def _report_validate(config: ConfigLike, payload: dict[str, Any]) -> dict[str, Any]:
    try:
        from radcopilot.report.validator import validate_impression, summarize_issues  # type: ignore

        impression = str(payload.get("impression", "") or "")
        findings = str(payload.get("findings", "") or "")
        if not impression.strip():
            return {"ok": False, "error": "Missing required field: impression"}
        result = validate_impression(
            impression,
            findings_text=findings,
            template_id=str(payload.get("template_id", "") or ""),
            allow_negatives=bool(payload.get("allow_negatives", False)),
        )
        return {
            "ok": True,
            "valid": result.valid,
            "summary": summarize_issues(result),
            "validation": result.to_dict(),
        }
    except ModuleNotFoundError:
        return {"ok": False, "error": "Report validator module not available"}
    except Exception as exc:
        return {"ok": False, "error": f"Report validation failed: {type(exc).__name__}: {exc}"}


# -----------------------------
# History route payload builders
# -----------------------------


def _history_list(
    config: ConfigLike,
    *,
    limit: int = 50,
    offset: int = 0,
    query: str = "",
) -> dict[str, Any]:
    try:
        from radcopilot.services.history_service import list_history  # type: ignore

        return list_history(config=config, limit=limit, offset=offset, query=query)
    except ModuleNotFoundError:
        return {"ok": False, "error": "History service not available"}
    except Exception as exc:
        return {"ok": False, "error": f"History list failed: {type(exc).__name__}: {exc}"}


def _history_get(config: ConfigLike, entry_id: str) -> dict[str, Any]:
    try:
        from radcopilot.services.history_service import get_history_entry  # type: ignore

        result = get_history_entry(entry_id=entry_id, config=config)
        if result is None:
            return {"ok": False, "error": f"History entry not found: {entry_id}"}
        return {"ok": True, "item": result}
    except ModuleNotFoundError:
        return {"ok": False, "error": "History service not available"}
    except Exception as exc:
        return {"ok": False, "error": f"History get failed: {type(exc).__name__}: {exc}"}


def _history_append(config: ConfigLike, payload: dict[str, Any]) -> dict[str, Any]:
    try:
        from radcopilot.services.history_service import append_history_entry  # type: ignore

        return append_history_entry(payload, config=config)
    except ModuleNotFoundError:
        return {"ok": False, "error": "History service not available"}
    except Exception as exc:
        return {"ok": False, "error": f"History append failed: {type(exc).__name__}: {exc}"}


def _history_delete(config: ConfigLike, entry_id: str) -> dict[str, Any]:
    try:
        from radcopilot.services.history_service import delete_history_entry  # type: ignore

        return delete_history_entry(entry_id=entry_id, config=config)
    except ModuleNotFoundError:
        return {"ok": False, "error": "History service not available"}
    except Exception as exc:
        return {"ok": False, "error": f"History delete failed: {type(exc).__name__}: {exc}"}


def _history_toggle_star(config: ConfigLike, entry_id: str) -> dict[str, Any]:
    try:
        from radcopilot.services.history_service import toggle_star  # type: ignore

        return toggle_star(entry_id=entry_id, config=config)
    except ModuleNotFoundError:
        return {"ok": False, "error": "History service not available"}
    except Exception as exc:
        return {"ok": False, "error": f"History star toggle failed: {type(exc).__name__}: {exc}"}


# -----------------------------
# Benchmark helpers
# -----------------------------


def _get_benchmark_loader_symbol(name: str) -> Any:
    try:
        from radcopilot.benchmark import loader as loader_module  # type: ignore
    except Exception:
        from ..benchmark import loader as loader_module  # type: ignore
    return getattr(loader_module, name)


def _get_benchmark_scorer_symbol(name: str) -> Any:
    try:
        from radcopilot.benchmark import scorer as scorer_module  # type: ignore
    except Exception:
        from ..benchmark import scorer as scorer_module  # type: ignore
    return getattr(scorer_module, name)


def _normalize_benchmark_case_for_scoring(case: dict[str, Any]) -> dict[str, Any]:
    prepared = dict(case)
    if _case_has_prediction(prepared):
        return prepared

    metadata = prepared.get("metadata", {})
    if not isinstance(metadata, dict):
        return prepared

    for key in (
        "predicted_text",
        "prediction",
        "generated_impression",
        "generated_text",
        "output",
        "result",
        "response",
    ):
        value = metadata.get(key)
        if isinstance(value, str) and value.strip():
            prepared["prediction"] = value.strip()
            break
    return prepared


def _case_has_prediction(case: dict[str, Any]) -> bool:
    for key in (
        "predicted_text",
        "prediction",
        "generated_impression",
        "generated_text",
        "output",
        "result",
        "response",
    ):
        value = case.get(key)
        if isinstance(value, str) and value.strip():
            return True
    return False


# -----------------------------
# Local helpers
# -----------------------------


def _fallback_index_html(config: ConfigLike) -> str:
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{config.app_name}</title>
  <style>
    body {{
      margin: 0; font-family: Inter, Arial, sans-serif; background: #0b1220; color: #e5eef8;
      display: grid; place-items: center; min-height: 100vh;
    }}
    .card {{
      width: min(860px, 92vw); background: #101a2b; border: 1px solid #1f3150;
      border-radius: 18px; padding: 28px; box-shadow: 0 20px 60px rgba(0,0,0,.35);
    }}
    h1 {{ margin-top: 0; font-size: 1.8rem; }}
    code {{ background: #0b1220; padding: 2px 6px; border-radius: 6px; }}
    .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-top: 18px; }}
    .tile {{ background: #0d1728; border: 1px solid #1f3150; border-radius: 12px; padding: 14px; }}
    a {{ color: #7dd3fc; }}
    .muted {{ color: #9fb0c9; }}
  </style>
</head>
<body>
  <main class="card">
    <h1>{config.app_name}</h1>
    <p>This modular server is running. The full browser UI has not been moved into <code>ui/index.html</code> yet, so you are seeing the fallback page from <code>radcopilot/server/app.py</code>.</p>
    <div class="grid">
      <section class="tile"><strong>Server</strong><p class="muted">{config.base_url}</p></section>
      <section class="tile"><strong>Ollama</strong><p class="muted">{config.ollama_url}</p></section>
      <section class="tile"><strong>RAG Data Dir</strong><p class="muted">{config.data_dir}</p></section>
      <section class="tile"><strong>Log File</strong><p class="muted">{config.log_file}</p></section>
    </div>
    <p style="margin-top: 18px;">
      <a href="/health">/health</a> ·
      <a href="/config">/config</a> ·
      <a href="/rag/status">/rag/status</a> ·
      <a href="/benchmark/datasets">/benchmark/datasets</a> ·
      <a href="/report/templates">/report/templates</a> ·
      <a href="/history">/history</a>
    </p>
  </main>
</body>
</html>
"""


def _ollama_available(ollama_url: str, timeout: float = 2.0) -> bool:
    try:
        urllib.request.urlopen(f"{ollama_url.rstrip('/')}/api/tags", timeout=timeout)
        return True
    except Exception:
        return False


def _whisper_available() -> bool:
    try:
        import whisper  # noqa: F401

        return True
    except Exception:
        return False


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _safe_int(value: str, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _tail_jsonl(path: Path, limit: int) -> list[dict[str, Any]]:
    if not path.exists() or not path.is_file():
        return []
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()[-limit:]
    items: list[dict[str, Any]] = []
    for line in reversed(lines):
        try:
            obj = json.loads(line)
            if isinstance(obj, dict):
                items.append(obj)
        except Exception:
            items.append({"raw": line})
    return items


def _read_json_file(path: Path, default: Any) -> Any:
    if not path.exists() or not path.is_file():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _count_lines(path: Path) -> int:
    if not path.exists() or not path.is_file():
        return 0
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        return sum(1 for _ in fh)


def _format_examples_block(items: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for idx, item in enumerate(items, start=1):
        findings = str(item.get("findings", "")).strip()
        impression = str(item.get("impression", "")).strip()
        if not findings and not impression:
            continue
        parts.append(f"Example {idx}:\nFindings: {findings[:400]}\nImpression:\n{impression[:300]}")
    if not parts:
        return ""
    return "\n\nReal radiologist examples for similar cases:\n\n" + "\n\n".join(parts)


def _tokenize(text: str) -> set[str]:
    cleaned = "".join(ch.lower() if ch.isalnum() else " " for ch in text)
    return {tok for tok in cleaned.split() if len(tok) >= 2}


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


def _add_rag_record(config: ConfigLike, record: dict[str, Any]) -> bool:
    findings = str(record.get("findings", "")).strip()
    impression = str(record.get("impression", "")).strip()
    if not findings or not impression:
        return False

    library_file = config.base_dir / "rag_library.json"
    items = _read_json_file(library_file, default=[])
    if not isinstance(items, list):
        items = []

    signature = findings[:500].strip().lower()
    for item in items:
        if not isinstance(item, dict):
            continue
        existing = str(item.get("findings", "")).strip().lower()
        if existing[:500] == signature:
            return False

    payload = {
        "findings": findings,
        "impression": impression,
        "modality": str(record.get("modality", "unknown") or "unknown"),
        "source": str(record.get("source", "manual") or "manual"),
        "created_at": str(record.get("created_at", _utc_now())),
    }
    items.append(payload)
    library_file.write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")
    return True


def _parse_txt_case(text: str) -> dict[str, Any] | None:
    upper = text.upper()
    findings_idx = upper.find("FINDINGS")
    impression_idx = upper.find("IMPRESSION")
    conclusion_idx = upper.find("CONCLUSION")

    if findings_idx == -1:
        return None

    end_idx = impression_idx if impression_idx != -1 else conclusion_idx
    findings = (
        text[findings_idx:end_idx].split(":", 1)[-1].strip()
        if end_idx != -1
        else text[findings_idx:].split(":", 1)[-1].strip()
    )

    impression = ""
    if impression_idx != -1:
        impression = text[impression_idx:].split(":", 1)[-1].strip()
    elif conclusion_idx != -1:
        impression = text[conclusion_idx:].split(":", 1)[-1].strip()

    if not findings or not impression:
        return None

    return {
        "findings": findings,
        "impression": impression,
        "modality": "unknown",
        "source": "txt",
    }


def _parse_csv_cases(path: Path) -> list[dict[str, Any]]:
    import csv

    items: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", errors="replace", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            findings = (
                row.get("Findings_EN")
                or row.get("findings")
                or row.get("Findings")
                or row.get("FINDINGS")
                or ""
            )
            impression = (
                row.get("Impression_EN")
                or row.get("impression")
                or row.get("Impression")
                or row.get("IMPRESSION")
                or ""
            )
            findings = str(findings).strip()
            impression = str(impression).strip()
            if findings and impression:
                items.append(
                    {
                        "findings": findings,
                        "impression": impression,
                        "modality": "unknown",
                        "source": "csv",
                        "source_file": str(path),
                    }
                )
    return items


__all__ = [
    "ConfigLike",
    "create_handler",
]
