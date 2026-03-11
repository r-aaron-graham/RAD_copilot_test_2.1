#!/usr/bin/env python3
"""
RadCopilot Local - modular entrypoint

This file is intended to live at:

    radcopilot/main.py

Responsibilities:
- load runtime configuration
- perform crash-safe startup handling
- verify or start Ollama
- optionally bootstrap a startup RAG index
- resolve the HTTP handler from modular server code
- fall back to a safe minimal handler if the rest of the package is not yet built
- launch the local server and open the browser

This file is intentionally self-contained so it can run while the rest of the
refactor is still being completed.
"""

from __future__ import annotations

import datetime as _dt
import http.server
import json
import os
from pathlib import Path
import socketserver
import subprocess
import sys
import threading
import time
import traceback
import urllib.request
import webbrowser
from dataclasses import dataclass
from typing import Optional, Type


APP_NAME = "RadCopilot Local"
DEFAULT_PORT = 7432
DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_BROWSER_DELAY_SECONDS = 1.25
DEFAULT_HOST = "127.0.0.1"

# This file is expected to live in radcopilot/main.py, so the repository root
# is one directory above this file.
DEFAULT_BASE_DIR = Path(__file__).resolve().parent.parent


@dataclass(slots=True)
class AppConfig:
    """Runtime configuration for the local launcher."""

    app_name: str = APP_NAME
    host: str = DEFAULT_HOST
    port: int = DEFAULT_PORT
    ollama_url: str = DEFAULT_OLLAMA_URL
    open_browser: bool = True
    browser_delay_seconds: float = DEFAULT_BROWSER_DELAY_SECONDS
    auto_start_ollama: bool = True
    build_startup_rag: bool = True

    # Project-root-relative paths
    base_dir: Path = DEFAULT_BASE_DIR
    ui_dir: Path = DEFAULT_BASE_DIR / "radcopilot" / "ui"
    log_file: Path = DEFAULT_BASE_DIR / "radcopilot_errors.jsonl"
    data_dir: Path = DEFAULT_BASE_DIR / "radcopilot_datasets"

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    @classmethod
    def from_env(cls) -> "AppConfig":
        env = os.environ

        default_base_dir = DEFAULT_BASE_DIR
        base_dir = Path(env.get("RADCOPILOT_BASE_DIR", default_base_dir)).resolve()
        ui_dir = Path(env.get("RADCOPILOT_UI_DIR", base_dir / "radcopilot" / "ui")).resolve()
        log_file = Path(env.get("RADCOPILOT_LOG_FILE", base_dir / "radcopilot_errors.jsonl")).resolve()
        data_dir = Path(env.get("RADCOPILOT_DATA_DIR", base_dir / "radcopilot_datasets")).resolve()

        return cls(
            app_name=env.get("RADCOPILOT_APP_NAME", APP_NAME),
            host=env.get("RADCOPILOT_HOST", DEFAULT_HOST),
            port=int(env.get("RADCOPILOT_PORT", DEFAULT_PORT)),
            ollama_url=env.get("RADCOPILOT_OLLAMA_URL", DEFAULT_OLLAMA_URL).rstrip("/"),
            open_browser=_env_bool("RADCOPILOT_OPEN_BROWSER", True),
            browser_delay_seconds=float(env.get("RADCOPILOT_BROWSER_DELAY", DEFAULT_BROWSER_DELAY_SECONDS)),
            auto_start_ollama=_env_bool("RADCOPILOT_AUTO_START_OLLAMA", True),
            build_startup_rag=_env_bool("RADCOPILOT_BUILD_STARTUP_RAG", True),
            base_dir=base_dir,
            ui_dir=ui_dir,
            log_file=log_file,
            data_dir=data_dir,
        )


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on", "y"}


def log_event(config: AppConfig, event_type: str, detail: object, *, context: str = "") -> None:
    """Append a JSONL log entry. Logging must never crash startup."""
    try:
        config.log_file.parent.mkdir(parents=True, exist_ok=True)
        entry = {
            "ts": _dt.datetime.now(_dt.timezone.utc).isoformat().replace("+00:00", "Z"),
            "type": event_type,
            "detail": detail if isinstance(detail, (dict, list)) else str(detail)[:1000],
            "context": context[:200],
        }
        with config.log_file.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception:
        pass


def show_error_and_exit(config: AppConfig, message: str) -> "None":
    """Crash-safe startup message. Shows a Windows dialog when possible."""
    log_event(config, "STARTUP_CRASH", message)
    print(f"\nFATAL ERROR: {message}", file=sys.stderr)

    try:
        if sys.platform == "win32":
            import ctypes  # pylint: disable=import-outside-toplevel

            ctypes.windll.user32.MessageBoxW(
                0,
                f"{config.app_name} failed to start:\n\n{message}\n\nSee {config.log_file.name} for details.",
                f"{config.app_name} — Startup Error",
                0x10,
            )
        else:
            print("Check the log file for details:", config.log_file, file=sys.stderr)
    except Exception:
        pass

    raise SystemExit(1)


def ollama_available(config: AppConfig, timeout: float = 2.0) -> bool:
    """Return True when Ollama responds to /api/tags."""
    try:
        urllib.request.urlopen(f"{config.ollama_url}/api/tags", timeout=timeout)
        return True
    except Exception:
        return False


def start_ollama(config: AppConfig) -> bool:
    """Start local Ollama if needed. Returns True if available or started."""
    if ollama_available(config):
        print("  Ollama already running.")
        return True

    if not config.auto_start_ollama:
        print("  Ollama is not reachable and auto-start is disabled.")
        return False

    print("  Starting Ollama...")
    env = os.environ.copy()
    env["OLLAMA_ORIGINS"] = "*"

    try:
        kwargs: dict[str, object] = {"env": env}
        if sys.platform == "win32":
            kwargs["creationflags"] = subprocess.CREATE_NEW_CONSOLE  # type: ignore[attr-defined]

        subprocess.Popen(["ollama", "serve"], **kwargs)  # noqa: S603,S607
        time.sleep(3)

        ok = ollama_available(config, timeout=3.0)
        if ok:
            log_event(config, "OLLAMA_START", "Started local Ollama server")
        else:
            log_event(config, "OLLAMA_WARN", "Ollama process launched but endpoint not yet responding")
        return ok

    except FileNotFoundError:
        msg = "ollama command not found. Install Ollama or disable auto-start."
        print(f"  WARNING: {msg}")
        log_event(config, "OLLAMA_MISSING", msg)
        return False
    except Exception as exc:  # pragma: no cover
        log_event(config, "OLLAMA_START_ERR", f"{type(exc).__name__}: {exc}")
        return False


def bootstrap_startup_rag(config: AppConfig) -> None:
    """
    Best-effort hook for modular RAG bootstrap.

    Expected import target:
        radcopilot.rag.startup_index.build_startup_index(config)
    """
    if not config.build_startup_rag:
        print("  Startup RAG bootstrap disabled.")
        return

    try:
        from radcopilot.rag.startup_index import build_startup_index  # type: ignore

        print("  Building startup RAG index...")
        result = build_startup_rag_result_safe(build_startup_index, config)
        log_event(config, "RAG_BOOTSTRAP", result)
    except ModuleNotFoundError:
        print("  RAG bootstrap module not created yet — skipping.")
    except Exception as exc:
        print(f"  RAG bootstrap failed: {type(exc).__name__}: {exc}")
        log_event(config, "RAG_BOOTSTRAP_ERR", f"{type(exc).__name__}: {exc}")


def build_startup_rag_result_safe(builder: object, config: AppConfig) -> object:
    """
    Call the startup-index builder and always return something loggable.
    """
    try:
        result = builder(config)  # type: ignore[misc]
        return result if result is not None else "Startup RAG index built"
    except Exception:
        raise


def open_browser_delayed(config: AppConfig) -> None:
    """Open the local UI after a short delay."""
    if not config.open_browser:
        return

    time.sleep(config.browser_delay_seconds)
    try:
        webbrowser.open(config.base_url)
    except Exception as exc:
        log_event(config, "BROWSER_OPEN_ERR", f"{type(exc).__name__}: {exc}")


def default_index_html(config: AppConfig) -> str:
    """Fallback UI shown before the full front-end is available."""
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{config.app_name}</title>
  <style>
    body {{
      margin: 0;
      font-family: Inter, Arial, sans-serif;
      background: #0b1220;
      color: #e5eef8;
      display: grid;
      place-items: center;
      min-height: 100vh;
    }}
    .card {{
      width: min(860px, 92vw);
      background: #101a2b;
      border: 1px solid #1f3150;
      border-radius: 18px;
      padding: 28px;
      box-shadow: 0 20px 60px rgba(0,0,0,.35);
    }}
    h1 {{
      margin-top: 0;
      font-size: 1.8rem;
    }}
    code {{
      background: #0b1220;
      padding: 2px 6px;
      border-radius: 6px;
    }}
    .grid {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 12px;
      margin-top: 18px;
    }}
    .tile {{
      background: #0d1728;
      border: 1px solid #1f3150;
      border-radius: 12px;
      padding: 14px;
    }}
    a {{
      color: #7dd3fc;
    }}
    .muted {{
      color: #9fb0c9;
    }}
  </style>
</head>
<body>
  <main class="card">
    <h1>{config.app_name}</h1>
    <p>The modular launcher is running. If the full browser UI is not ready yet, this fallback page is shown instead.</p>
    <div class="grid">
      <section class="tile">
        <strong>Server</strong>
        <p class="muted">Bound to <code>{config.base_url}</code></p>
      </section>
      <section class="tile">
        <strong>Ollama</strong>
        <p class="muted">Configured at <code>{config.ollama_url}</code></p>
      </section>
      <section class="tile">
        <strong>UI Directory</strong>
        <p class="muted"><code>{config.ui_dir}</code></p>
      </section>
      <section class="tile">
        <strong>Log File</strong>
        <p class="muted"><code>{config.log_file.name}</code></p>
      </section>
    </div>
    <p style="margin-top: 18px;">
      <a href="/health">Health endpoint</a>
    </p>
  </main>
</body>
</html>
"""


def _read_text_if_exists(path: Path) -> Optional[str]:
    if not path.exists() or not path.is_file():
        return None
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return None


def create_fallback_handler(config: AppConfig) -> Type[http.server.BaseHTTPRequestHandler]:
    """Minimal handler used until modular server routes are implemented."""

    class FallbackHandler(http.server.BaseHTTPRequestHandler):
        server_version = "RadCopilotLocal/0.1"

        def _json(self, payload: object, status: int = 200) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _html(self, html: str, status: int = 200) -> None:
            body = html.encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, fmt: str, *args: object) -> None:  # noqa: A003
            log_event(config, "HTTP", fmt % args, context=self.path)

        def do_GET(self) -> None:  # noqa: N802
            if self.path in {"/", "/index.html"}:
                html = _read_text_if_exists(config.ui_dir / "index.html") or default_index_html(config)
                self._html(html)
                return

            if self.path == "/health":
                self._json(
                    {
                        "ok": True,
                        "app": config.app_name,
                        "base_url": config.base_url,
                        "ollama_url": config.ollama_url,
                        "ollama_up": ollama_available(config),
                        "ui_dir": str(config.ui_dir),
                        "data_dir": str(config.data_dir),
                        "log_file": str(config.log_file),
                    }
                )
                return

            if self.path == "/config":
                self._json(
                    {
                        "app_name": config.app_name,
                        "host": config.host,
                        "port": config.port,
                        "base_url": config.base_url,
                        "ollama_url": config.ollama_url,
                        "open_browser": config.open_browser,
                        "data_dir": str(config.data_dir),
                        "log_file": str(config.log_file),
                    }
                )
                return

            self._json({"ok": False, "error": "Not Found", "path": self.path}, status=404)

        def do_POST(self) -> None:  # noqa: N802
            self._json(
                {
                    "ok": False,
                    "error": "POST route not implemented yet in fallback server",
                    "path": self.path,
                },
                status=501,
            )

    return FallbackHandler


def resolve_handler(config: AppConfig) -> Type[http.server.BaseHTTPRequestHandler]:
    """
    Prefer the modular server app if it exists.

    Expected shapes:
      - radcopilot.server.app.create_handler(config) -> HandlerClass
      - radcopilot.server.app.Handler
    """
    try:
        from radcopilot.server.app import create_handler  # type: ignore

        handler = create_handler(config)
        log_event(config, "SERVER_HANDLER", "Loaded create_handler from radcopilot.server.app")
        return handler
    except ModuleNotFoundError:
        pass
    except Exception as exc:
        log_event(config, "SERVER_HANDLER_ERR", f"create_handler failed: {type(exc).__name__}: {exc}")

    try:
        from radcopilot.server.app import Handler  # type: ignore

        log_event(config, "SERVER_HANDLER", "Loaded Handler from radcopilot.server.app")
        return Handler
    except ModuleNotFoundError:
        pass
    except Exception as exc:
        log_event(config, "SERVER_HANDLER_ERR", f"Handler import failed: {type(exc).__name__}: {exc}")

    log_event(config, "SERVER_HANDLER", "Using fallback handler")
    return create_fallback_handler(config)


class ThreadingHTTPServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def run() -> int:
    config = AppConfig.from_env()

    try:
        print()
        print("  ============================================")
        print(f"   {config.app_name} — Modular Launcher")
        print("  ============================================")
        print()

        config.data_dir.mkdir(parents=True, exist_ok=True)

        ollama_ok = start_ollama(config)
        if not ollama_ok:
            print("  WARNING: Ollama is not reachable. The app can still start, but AI routes will fail until Ollama is available.")

        bootstrap_startup_rag(config)

        handler = resolve_handler(config)

        print(f"  Starting server on {config.base_url}")
        print("  Keep this window open. Press Ctrl+C to stop.")
        print()

        if config.open_browser:
            threading.Thread(target=open_browser_delayed, args=(config,), daemon=True).start()

        with ThreadingHTTPServer((config.host, config.port), handler) as httpd:
            log_event(
                config,
                "SERVER_START",
                {
                    "base_url": config.base_url,
                    "ollama_url": config.ollama_url,
                    "ui_dir": str(config.ui_dir),
                    "data_dir": str(config.data_dir),
                },
            )
            httpd.serve_forever()

    except KeyboardInterrupt:
        print(f"\n  {config.app_name} stopped.")
        log_event(config, "SERVER_STOP", "Stopped by user")
        return 0

    except OSError as exc:
        msg = f"Port {config.port} is unavailable: {exc}"
        log_event(config, "PORT_IN_USE", msg)
        show_error_and_exit(config, msg)

    except Exception as exc:  # pragma: no cover
        tb = traceback.format_exc()
        log_event(config, "UNHANDLED_STARTUP_ERR", f"{type(exc).__name__}: {exc}\n{tb}")
        show_error_and_exit(config, f"{type(exc).__name__}: {exc}")

    return 0


if __name__ == "__main__":
    raise SystemExit(run())
