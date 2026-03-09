from __future__ import annotations

"""
radcopilot.server.app

Modular HTTP server for RadCopilot Local.

This file is the first real server module in the refactor from the original
single-file application into a modular local package. It is designed to work
immediately with the previously created `main.py` entrypoint.

Current goals:
- provide a production-quality local HTTP handler factory
- serve the UI from /ui when available
- expose health/config/log routes
- expose the core route surface used by the original app
- proxy `/api/*` requests to local Ollama
- keep optional feature routes available even before their service modules exist
- fail gracefully with explicit JSON errors instead of crashing

This module intentionally uses only the Python standard library so it can run
before the rest of the refactor is complete.
"""

import json
import mimetypes
import os
from pathlib import Path
import tempfile
import traceback
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler
from typing import Any, Callable, Protocol, Type


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
        server_version = "RadCopilotLocal/0.2"

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

        def _send_not_implemented(self, message: str) -> None:
            self._json(
                {
                    "ok": False,
                    "error": message,
                    "path": self.path,
                    "hint": "Create the corresponding service module to enable this route.",
                },
                status=501,
            )

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
                    self._json({"ok": True, "items": _list_benchmark_datasets(config.data_dir)})
                    return

                if path == "/benchmark/load-path":
                    source_path = self._query_first("path", "").strip()
                    if not source_path:
                        self._json({"ok": False, "error": "Missing query parameter: path"}, status=400)
                        return
                    self._json(_benchmark_load_path(config, source_path))
                    return

                if path.startswith("/api/"):
                    self._proxy_to_ollama("GET")
                    return

                self._send_not_found()
            except Exception as exc:  # pragma: no cover - defensive HTTP path
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
                    self._json(_rag_rate(config, payload), status=200)
                    return

                if path == "/rag/train":
                    payload = self._read_json_body()
                    self._json(_rag_train(config, payload))
                    return

                if path == "/benchmark/load":
                    self._send_not_implemented("Benchmark upload parsing is not implemented in this module yet")
                    return

                if path == "/benchmark/load-path":
                    payload = self._read_json_body()
                    source_path = str(payload.get("path", "")).strip()
                    if not source_path:
                        self._json({"ok": False, "error": "Missing JSON field: path"}, status=400)
                        return
                    self._json(_benchmark_load_path(config, source_path))
                    return

                if path == "/whisper/transcribe":
                    self._json(_whisper_transcribe(config, self))
                    return

                if path.startswith("/api/"):
                    self._proxy_to_ollama("POST")
                    return

                self._send_not_found()
            except json.JSONDecodeError:
                self._json({"ok": False, "error": "Invalid JSON body", "path": self.path}, status=400)
            except Exception as exc:  # pragma: no cover - defensive HTTP path
                self._handle_route_exception(exc)

        # -----------------------------
        # Route implementations
        # -----------------------------
        def _serve_index(self) -> None:
            ui_index = config.ui_dir / "index.html"
            if ui_index.exists() and ui_index.is_file():
                text = ui_index.read_text(encoding="utf-8")
                self._text(text, content_type="text/html; charset=utf-8")
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
            self._bytes(target.read_bytes(), content_type=mime or "application/octet-stream")

        def _proxy_to_ollama(self, method: str) -> None:
            target_url = f"{config.ollama_url}{urllib.parse.urlparse(self.path).path}"
            query = urllib.parse.urlparse(self.path).query
            if query:
                target_url = f"{target_url}?{query}"

            body: bytes | None = None
            if method in {"POST", "PUT", "PATCH"}:
                length = int(self.headers.get("Content-Length", "0") or "0")
                body = self.rfile.read(length) if length > 0 else b""

            headers = {}
            content_type = self.headers.get("Content-Type")
            if content_type:
                headers["Content-Type"] = content_type
            accept = self.headers.get("Accept")
            if accept:
                headers["Accept"] = accept

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
                {
                    "ok": False,
                    "error": f"{type(exc).__name__}: {exc}",
                    "path": self.path,
                },
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

    # Future modular implementation hook.
    try:
        from radcopilot.rag.library import query_records  # type: ignore

        items = query_records(config=config, findings=findings, modality=modality, k=k)
        return {"ok": True, "items": items, "count": len(items), "source": "module"}
    except ModuleNotFoundError:
        pass
    except Exception as exc:
        return {"ok": False, "error": f"RAG query module error: {type(exc).__name__}: {exc}", "items": [], "count": 0}

    # Lightweight fallback search against a JSON library file.
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

    # Future modular implementation hook.
    try:
        from radcopilot.rag.trainer import train_path  # type: ignore

        result = train_path(config=config, path=source_path)
        return {"ok": True, "source": "module", **result}
    except ModuleNotFoundError:
        pass
    except Exception as exc:
        return {"ok": False, "error": f"RAG trainer module error: {type(exc).__name__}: {exc}"}

    # Standard-library fallback: only basic TXT/CSV ingestion for now.
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
    # Future modular implementation hook.
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



def _benchmark_load_path(config: ConfigLike, source_path: str) -> dict[str, Any]:
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
        }

    stat = path.stat()
    preview = ""
    if path.suffix.lower() in {".txt", ".md", ".json", ".csv", ".xml"}:
        preview = path.read_text(encoding="utf-8", errors="replace")[:4000]

    return {
        "ok": True,
        "path": str(path),
        "kind": "file",
        "name": path.name,
        "suffix": path.suffix.lower(),
        "size_bytes": stat.st_size,
        "preview": preview,
    }


# -----------------------------
# Local helpers
# -----------------------------

def _fallback_index_html(config: ConfigLike) -> str:
    return f"""<!DOCTYPE html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\">
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">
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
  <main class=\"card\">
    <h1>{config.app_name}</h1>
    <p>This modular server is running. The full browser UI has not been moved into <code>ui/index.html</code> yet, so you are seeing the fallback page from <code>radcopilot/server/app.py</code>.</p>
    <div class=\"grid\">
      <section class=\"tile\"><strong>Server</strong><p class=\"muted\">{config.base_url}</p></section>
      <section class=\"tile\"><strong>Ollama</strong><p class=\"muted\">{config.ollama_url}</p></section>
      <section class=\"tile\"><strong>RAG Data Dir</strong><p class=\"muted\">{config.data_dir}</p></section>
      <section class=\"tile\"><strong>Log File</strong><p class=\"muted\">{config.log_file}</p></section>
    </div>
    <p style=\"margin-top: 18px;\">
      <a href=\"/health\">/health</a> ·
      <a href=\"/config\">/config</a> ·
      <a href=\"/rag/status\">/rag/status</a> ·
      <a href=\"/benchmark/datasets\">/benchmark/datasets</a>
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



def _list_benchmark_datasets(data_dir: Path) -> list[dict[str, Any]]:
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
    findings = text[findings_idx:end_idx].split(":", 1)[-1].strip() if end_idx != -1 else text[findings_idx:].split(":", 1)[-1].strip()
    impression = ""
    if impression_idx != -1:
        impression = text[impression_idx:].split(":", 1)[-1].strip()
    elif conclusion_idx != -1:
        impression = text[conclusion_idx:].split(":", 1)[-1].strip()

    if not findings or not impression:
        return None
    return {"findings": findings, "impression": impression, "modality": "unknown", "source": "txt"}



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

