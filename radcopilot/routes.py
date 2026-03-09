from __future__ import annotations

"""
radcopilot.server.routes

Reusable route registry and dispatch layer for the RadCopilot modular server.

Why this file exists:
- `server/app.py` currently contains inline route matching inside `do_GET()` and
  `do_POST()`.
- the refactor needs a dedicated routing layer so endpoint definitions can be
  declared once, tested independently, and reused by future handler factories.
- this module stays standard-library only and is intentionally compatible with
  the existing `main.py` / `server.app` contract.

What this module provides:
- a small `RouteRegistry` with exact, wildcard, and path-parameter matching
- `RouteContext` for passing request/config/query/body state to handlers
- response helpers for JSON, text, and bytes
- a `build_core_registry()` helper mirroring the current RadCopilot route surface

Design note:
This file does not replace `server/app.py` on its own. It provides the routing
layer that `server/app.py` can adopt in a later pass without changing the public
API surface.
"""

from dataclasses import dataclass, field
import json
from pathlib import Path
import re
import urllib.parse
from typing import Any, Callable, Dict, Iterable, Mapping, MutableMapping, Optional, Protocol


# ---------------------------------------------------------------------------
# Contracts
# ---------------------------------------------------------------------------


class ConfigLike(Protocol):
    """Minimal runtime config expected from the launcher."""

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


class HandlerLike(Protocol):
    """Minimal HTTP handler surface used by this module."""

    path: str
    headers: Mapping[str, str]
    rfile: Any
    wfile: Any

    def send_response(self, code: int) -> None:  # pragma: no cover - protocol only
        ...

    def send_header(self, key: str, value: str) -> None:  # pragma: no cover - protocol only
        ...

    def end_headers(self) -> None:  # pragma: no cover - protocol only
        ...


RouteHandler = Callable[["RouteContext"], None]


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class RouteError(Exception):
    """Base routing error."""


class RouteNotFound(RouteError):
    """Raised when no route matches a request."""


class MethodNotAllowed(RouteError):
    """Raised when the path matches but the method does not."""


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class Route:
    """
    Route definition.

    Supported pattern forms:
    - exact: `/health`
    - wildcard suffix: `/api/*`
    - named param: `/benchmark/{name}`
    - named catch-all: `/static/{path+}`
    """

    method: str
    pattern: str
    handler: RouteHandler
    name: str = ""
    description: str = ""
    tags: tuple[str, ...] = ()


@dataclass(slots=True)
class RouteMatch:
    """Matched route plus extracted path parameters."""

    route: Route
    params: dict[str, str] = field(default_factory=dict)


@dataclass(slots=True)
class RouteContext:
    """Request context passed to route handlers."""

    config: ConfigLike
    request_handler: HandlerLike
    method: str
    raw_path: str
    path: str
    query: dict[str, list[str]]
    params: dict[str, str] = field(default_factory=dict)
    body_cache: bytes | None = None
    json_cache: dict[str, Any] | None = None
    extras: dict[str, Any] = field(default_factory=dict)

    def query_first(self, key: str, default: str = "") -> str:
        return self.query.get(key, [default])[0]

    def read_body(self) -> bytes:
        if self.body_cache is not None:
            return self.body_cache
        length = int(self.request_handler.headers.get("Content-Length", "0") or "0")
        self.body_cache = self.request_handler.rfile.read(length) if length > 0 else b""
        return self.body_cache

    def read_json(self) -> dict[str, Any]:
        if self.json_cache is not None:
            return self.json_cache
        raw = self.read_body()
        if not raw:
            self.json_cache = {}
            return self.json_cache
        parsed = json.loads(raw.decode("utf-8"))
        self.json_cache = dict(parsed) if isinstance(parsed, dict) else {"_root": parsed}
        return self.json_cache


# ---------------------------------------------------------------------------
# Response helpers
# ---------------------------------------------------------------------------


def send_json(handler: HandlerLike, payload: Any, status: int = 200) -> None:
    body = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.send_header("Cache-Control", "no-store")
    handler.end_headers()
    handler.wfile.write(body)



def send_text(handler: HandlerLike, text: str, status: int = 200, content_type: str = "text/plain; charset=utf-8") -> None:
    body = text.encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", content_type)
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)



def send_bytes(handler: HandlerLike, data: bytes, status: int = 200, content_type: str = "application/octet-stream") -> None:
    handler.send_response(status)
    handler.send_header("Content-Type", content_type)
    handler.send_header("Content-Length", str(len(data)))
    handler.end_headers()
    handler.wfile.write(data)


# ---------------------------------------------------------------------------
# Pattern matching
# ---------------------------------------------------------------------------


_PARAM_RE = re.compile(r"\{([a-zA-Z_][a-zA-Z0-9_]*)(\+)?\}")



def _normalize_method(method: str) -> str:
    return method.strip().upper() or "GET"



def _normalize_path(path: str) -> str:
    if not path:
        return "/"
    if not path.startswith("/"):
        path = "/" + path
    if len(path) > 1 and path.endswith("/"):
        path = path[:-1]
    return path



def match_pattern(pattern: str, path: str) -> Optional[dict[str, str]]:
    """Match a route pattern and return extracted params when successful."""
    pattern = _normalize_path(pattern)
    path = _normalize_path(path)

    if pattern == path:
        return {}

    if pattern.endswith("/*"):
        prefix = pattern[:-1]
        if path.startswith(prefix):
            suffix = path[len(prefix):]
            return {"wildcard": suffix}
        return None

    pattern_segments = pattern.strip("/").split("/") if pattern != "/" else []
    path_segments = path.strip("/").split("/") if path != "/" else []

    params: dict[str, str] = {}
    i = 0
    j = 0
    while i < len(pattern_segments) and j < len(path_segments):
        segment = pattern_segments[i]
        match = _PARAM_RE.fullmatch(segment)
        if match:
            name, is_catch_all = match.group(1), bool(match.group(2))
            if is_catch_all:
                params[name] = "/".join(path_segments[j:])
                j = len(path_segments)
                i = len(pattern_segments)
                break
            params[name] = path_segments[j]
            i += 1
            j += 1
            continue
        if segment != path_segments[j]:
            return None
        i += 1
        j += 1

    if i == len(pattern_segments) and j == len(path_segments):
        return params
    if i == len(pattern_segments) - 1:
        last = pattern_segments[i]
        match = _PARAM_RE.fullmatch(last)
        if match and match.group(2):
            params[match.group(1)] = ""
            return params
    return None


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class RouteRegistry:
    """In-memory route registry with request dispatch."""

    def __init__(self) -> None:
        self._routes: list[Route] = []

    def add(
        self,
        method: str,
        pattern: str,
        handler: RouteHandler,
        *,
        name: str = "",
        description: str = "",
        tags: Iterable[str] | None = None,
    ) -> Route:
        route = Route(
            method=_normalize_method(method),
            pattern=_normalize_path(pattern),
            handler=handler,
            name=name,
            description=description,
            tags=tuple(tags or ()),
        )
        self._routes.append(route)
        return route

    def get(self, pattern: str, handler: RouteHandler, **meta: Any) -> Route:
        return self.add("GET", pattern, handler, **meta)

    def post(self, pattern: str, handler: RouteHandler, **meta: Any) -> Route:
        return self.add("POST", pattern, handler, **meta)

    def put(self, pattern: str, handler: RouteHandler, **meta: Any) -> Route:
        return self.add("PUT", pattern, handler, **meta)

    def patch(self, pattern: str, handler: RouteHandler, **meta: Any) -> Route:
        return self.add("PATCH", pattern, handler, **meta)

    def delete(self, pattern: str, handler: RouteHandler, **meta: Any) -> Route:
        return self.add("DELETE", pattern, handler, **meta)

    def any(self, pattern: str, handler: RouteHandler, **meta: Any) -> list[Route]:
        return [self.add(method, pattern, handler, **meta) for method in ("GET", "POST", "PUT", "PATCH", "DELETE")]

    def iter_routes(self) -> Iterable[Route]:
        return tuple(self._routes)

    def match(self, method: str, path: str) -> RouteMatch:
        method = _normalize_method(method)
        path = _normalize_path(path)

        path_exists_for_other_method = False
        for route in self._routes:
            params = match_pattern(route.pattern, path)
            if params is None:
                continue
            if route.method == method:
                return RouteMatch(route=route, params=params)
            path_exists_for_other_method = True

        if path_exists_for_other_method:
            raise MethodNotAllowed(f"Method {method} not allowed for path {path}")
        raise RouteNotFound(f"No route found for {method} {path}")

    def dispatch(
        self,
        *,
        config: ConfigLike,
        request_handler: HandlerLike,
        method: str,
        path: str,
        query: Mapping[str, list[str]] | None = None,
        extras: MutableMapping[str, Any] | None = None,
    ) -> bool:
        match = self.match(method, path)
        ctx = RouteContext(
            config=config,
            request_handler=request_handler,
            method=_normalize_method(method),
            raw_path=request_handler.path,
            path=_normalize_path(path),
            query=dict(query or {}),
            params=match.params,
            extras=dict(extras or {}),
        )
        match.route.handler(ctx)
        return True

    def describe(self) -> list[dict[str, Any]]:
        return [
            {
                "method": route.method,
                "pattern": route.pattern,
                "name": route.name,
                "description": route.description,
                "tags": list(route.tags),
            }
            for route in self._routes
        ]


# ---------------------------------------------------------------------------
# Core route registry builder
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class CoreRouteHandlers:
    """
    Handler bundle for the current RadCopilot route surface.

    These callables let `server/app.py` reuse this routing layer without moving
    all implementation details into this module.
    """

    serve_index: RouteHandler
    serve_static: RouteHandler
    health: RouteHandler
    config_view: RouteHandler
    logs_recent: RouteHandler
    whisper_status: RouteHandler
    rag_status: RouteHandler
    rag_query: RouteHandler
    rag_examples: RouteHandler
    benchmark_datasets: RouteHandler
    benchmark_load_path_get: RouteHandler
    rag_rate: RouteHandler
    rag_train: RouteHandler
    benchmark_load: RouteHandler
    benchmark_load_path_post: RouteHandler
    whisper_transcribe: RouteHandler
    ollama_proxy_get: RouteHandler
    ollama_proxy_post: RouteHandler



def build_core_registry(handlers: CoreRouteHandlers) -> RouteRegistry:
    """Build a registry that mirrors the current route surface in `server.app`."""
    registry = RouteRegistry()

    registry.get("/", handlers.serve_index, name="index-root", tags=("ui",))
    registry.get("/index.html", handlers.serve_index, name="index-html", tags=("ui",))
    registry.get("/static/{path+}", handlers.serve_static, name="static", tags=("ui", "static"))

    registry.get("/health", handlers.health, name="health", tags=("system",))
    registry.get("/config", handlers.config_view, name="config", tags=("system",))
    registry.get("/logs/recent", handlers.logs_recent, name="logs-recent", tags=("logs",))

    registry.get("/whisper/status", handlers.whisper_status, name="whisper-status", tags=("whisper",))
    registry.post("/whisper/transcribe", handlers.whisper_transcribe, name="whisper-transcribe", tags=("whisper",))

    registry.get("/rag/status", handlers.rag_status, name="rag-status", tags=("rag",))
    registry.get("/rag/query", handlers.rag_query, name="rag-query", tags=("rag",))
    registry.get("/rag/examples", handlers.rag_examples, name="rag-examples", tags=("rag",))
    registry.post("/rag/train", handlers.rag_train, name="rag-train", tags=("rag",))
    registry.post("/rag/rate", handlers.rag_rate, name="rag-rate", tags=("rag",))

    registry.get("/benchmark/datasets", handlers.benchmark_datasets, name="benchmark-datasets", tags=("benchmark",))
    registry.get("/benchmark/load-path", handlers.benchmark_load_path_get, name="benchmark-load-path-get", tags=("benchmark",))
    registry.post("/benchmark/load", handlers.benchmark_load, name="benchmark-load", tags=("benchmark",))
    registry.post("/benchmark/load-path", handlers.benchmark_load_path_post, name="benchmark-load-path-post", tags=("benchmark",))

    registry.get("/api/*", handlers.ollama_proxy_get, name="ollama-proxy-get", tags=("ollama", "proxy"))
    registry.post("/api/*", handlers.ollama_proxy_post, name="ollama-proxy-post", tags=("ollama", "proxy"))
    return registry


# ---------------------------------------------------------------------------
# Convenience helpers for app.py integration
# ---------------------------------------------------------------------------



def parse_request_path(raw_path: str) -> tuple[str, dict[str, list[str]]]:
    parsed = urllib.parse.urlparse(raw_path)
    return _normalize_path(parsed.path), urllib.parse.parse_qs(parsed.query)



def dispatch_request(
    *,
    registry: RouteRegistry,
    config: ConfigLike,
    request_handler: HandlerLike,
    method: str,
    extras: MutableMapping[str, Any] | None = None,
) -> bool:
    path, query = parse_request_path(request_handler.path)
    return registry.dispatch(
        config=config,
        request_handler=request_handler,
        method=method,
        path=path,
        query=query,
        extras=extras,
    )


# ---------------------------------------------------------------------------
# Minimal self-test helpers
# ---------------------------------------------------------------------------



def describe_core_routes() -> list[dict[str, Any]]:
    """
    Return the current RadCopilot core route map without needing real handlers.

    This is useful for tests and documentation generation.
    """

    def _noop(_: RouteContext) -> None:
        return None

    registry = build_core_registry(
        CoreRouteHandlers(
            serve_index=_noop,
            serve_static=_noop,
            health=_noop,
            config_view=_noop,
            logs_recent=_noop,
            whisper_status=_noop,
            rag_status=_noop,
            rag_query=_noop,
            rag_examples=_noop,
            benchmark_datasets=_noop,
            benchmark_load_path_get=_noop,
            rag_rate=_noop,
            rag_train=_noop,
            benchmark_load=_noop,
            benchmark_load_path_post=_noop,
            whisper_transcribe=_noop,
            ollama_proxy_get=_noop,
            ollama_proxy_post=_noop,
        )
    )
    return registry.describe()
