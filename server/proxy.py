from __future__ import annotations

"""
radcopilot.server.proxy

Reusable HTTP proxy helpers for forwarding local `/api/*` requests to Ollama.

This module exists to pull proxy behavior out of `server/app.py` so the RadCopilot
refactor can use one forwarding implementation across:
- inline BaseHTTPRequestHandler routes
- future route-registry handlers
- future test harnesses

Design goals:
- standard-library only
- preserve Ollama response status and body
- support GET / POST / PUT / PATCH / DELETE
- preserve streaming responses from Ollama endpoints
- provide helpful JSON errors when upstream forwarding fails
- remain easy to adopt incrementally in the current modular server
"""

from dataclasses import dataclass, field
import json
from typing import Any, Iterable, Mapping, MutableMapping, Protocol
import urllib.error
import urllib.parse
import urllib.request


DEFAULT_TIMEOUT = 300.0
DEFAULT_USER_AGENT = "RadCopilotLocal/0.1"
HOP_BY_HOP_HEADERS = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailers",
    "transfer-encoding",
    "upgrade",
}
DEFAULT_ALLOWED_REQUEST_HEADERS = {
    "accept",
    "accept-encoding",
    "authorization",
    "content-type",
    "x-requested-with",
}
DEFAULT_ALLOWED_RESPONSE_HEADERS = {
    "content-type",
    "cache-control",
    "etag",
    "last-modified",
    "x-request-id",
    "x-response-time",
}


class ConfigLike(Protocol):
    """Minimal runtime config expected by proxy helpers."""

    ollama_url: str


class HandlerLike(Protocol):
    """Subset of BaseHTTPRequestHandler used by the proxy."""

    path: str
    headers: Any
    rfile: Any
    wfile: Any

    def send_response(self, code: int, message: str | None = None) -> None: ...
    def send_header(self, keyword: str, value: str) -> None: ...
    def end_headers(self) -> None: ...


@dataclass(slots=True)
class ProxyRequest:
    """Normalized proxy request metadata."""

    method: str
    source_path: str
    target_url: str
    headers: dict[str, str] = field(default_factory=dict)
    body: bytes | None = None
    timeout: float = DEFAULT_TIMEOUT


@dataclass(slots=True)
class ProxyResponse:
    """Normalized upstream response payload."""

    status: int
    reason: str = ""
    headers: dict[str, str] = field(default_factory=dict)
    body: bytes = b""
    target_url: str = ""

    @property
    def content_type(self) -> str:
        return self.headers.get("Content-Type", "application/json; charset=utf-8")


class ProxyError(RuntimeError):
    """Base proxy error."""


class UpstreamHTTPError(ProxyError):
    """Raised when Ollama returns a non-2xx response."""

    def __init__(self, response: ProxyResponse) -> None:
        message = f"Upstream HTTP {response.status}"
        if response.reason:
            message = f"{message}: {response.reason}"
        super().__init__(message)
        self.response = response


class UpstreamConnectionError(ProxyError):
    """Raised when the upstream service cannot be reached."""


# ---------------------------------------------------------------------------
# URL / request normalization
# ---------------------------------------------------------------------------


def normalize_base_url(base_url: str) -> str:
    text = str(base_url or "").strip()
    if not text:
        return "http://localhost:11434"
    return text.rstrip("/")



def build_target_url(base_url: str, request_path: str) -> str:
    """Map `/api/*` request paths from the local app to the Ollama base URL."""
    parsed = urllib.parse.urlparse(str(request_path or "/"))
    path = parsed.path or "/"
    target = f"{normalize_base_url(base_url)}{path}"
    if parsed.query:
        target = f"{target}?{parsed.query}"
    return target



def infer_method(handler: HandlerLike, default: str = "GET") -> str:
    method = getattr(handler, "command", None) or default
    return str(method).upper()



def should_read_body(method: str) -> bool:
    return str(method).upper() in {"POST", "PUT", "PATCH", "DELETE"}



def read_request_body(handler: HandlerLike) -> bytes | None:
    length_raw = handler.headers.get("Content-Length", "0")
    try:
        length = int(length_raw or "0")
    except (TypeError, ValueError):
        length = 0
    if length <= 0:
        return None
    data = handler.rfile.read(length)
    return bytes(data) if data else None



def collect_forward_headers(
    handler: HandlerLike,
    *,
    allowed: Iterable[str] | None = None,
    extra: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Collect safe request headers for upstream forwarding."""
    allowed_set = {h.lower() for h in (allowed or DEFAULT_ALLOWED_REQUEST_HEADERS)}
    headers: dict[str, str] = {"User-Agent": DEFAULT_USER_AGENT}

    for key, value in handler.headers.items():
        if not value:
            continue
        lowered = key.lower()
        if lowered in HOP_BY_HOP_HEADERS:
            continue
        if lowered in allowed_set:
            headers[key] = str(value)

    for key, value in dict(extra or {}).items():
        if value is None:
            continue
        headers[str(key)] = str(value)

    return headers



def build_proxy_request(
    handler: HandlerLike,
    config: ConfigLike,
    *,
    method: str | None = None,
    timeout: float = DEFAULT_TIMEOUT,
    extra_headers: Mapping[str, str] | None = None,
) -> ProxyRequest:
    final_method = (method or infer_method(handler)).upper()
    body = read_request_body(handler) if should_read_body(final_method) else None
    headers = collect_forward_headers(handler, extra=extra_headers)
    return ProxyRequest(
        method=final_method,
        source_path=str(handler.path or "/"),
        target_url=build_target_url(config.ollama_url, str(handler.path or "/")),
        headers=headers,
        body=body,
        timeout=float(timeout),
    )


# ---------------------------------------------------------------------------
# Upstream forwarding
# ---------------------------------------------------------------------------


def forward_request(request: ProxyRequest) -> ProxyResponse:
    """Execute the upstream Ollama request and normalize the response."""
    req = urllib.request.Request(
        request.target_url,
        data=request.body,
        method=request.method,
        headers=dict(request.headers or {}),
    )
    try:
        with urllib.request.urlopen(req, timeout=float(request.timeout or DEFAULT_TIMEOUT)) as resp:
            payload = resp.read()
            return ProxyResponse(
                status=int(getattr(resp, "status", 200) or 200),
                reason=str(getattr(resp, "reason", "") or ""),
                headers=_select_response_headers(resp.headers),
                body=payload,
                target_url=request.target_url,
            )
    except urllib.error.HTTPError as exc:
        payload = exc.read() if exc.fp else b""
        response = ProxyResponse(
            status=int(exc.code),
            reason=str(exc.reason or "HTTP error"),
            headers=_select_response_headers(exc.headers),
            body=payload,
            target_url=request.target_url,
        )
        raise UpstreamHTTPError(response) from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise UpstreamConnectionError(
            f"Upstream connection failed: {type(exc).__name__}: {exc}"
        ) from exc



def proxy_to_ollama(
    handler: HandlerLike,
    config: ConfigLike,
    *,
    method: str | None = None,
    timeout: float = DEFAULT_TIMEOUT,
    extra_headers: Mapping[str, str] | None = None,
) -> ProxyResponse:
    """Forward the current handler request to Ollama and write the result back."""
    request = build_proxy_request(
        handler,
        config,
        method=method,
        timeout=timeout,
        extra_headers=extra_headers,
    )
    try:
        response = forward_request(request)
        write_proxy_response(handler, response)
        return response
    except UpstreamHTTPError as exc:
        write_proxy_response(handler, exc.response)
        return exc.response
    except ProxyError as exc:
        payload = {
            "ok": False,
            "error": str(exc),
            "path": str(handler.path or ""),
            "target_url": request.target_url,
        }
        send_json_error(handler, payload, status=502)
        return ProxyResponse(
            status=502,
            reason="Bad Gateway",
            headers={"Content-Type": "application/json; charset=utf-8"},
            body=json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8"),
            target_url=request.target_url,
        )


# ---------------------------------------------------------------------------
# Route-registry integration
# ---------------------------------------------------------------------------


def proxy_route_handler(ctx: Any) -> None:
    """
    Registry-friendly route handler.

    Expected ctx fields:
    - request_handler
    - config
    - method
    Optional extras:
    - timeout
    - extra_headers
    """
    handler = getattr(ctx, "request_handler")
    config = getattr(ctx, "config")
    method = getattr(ctx, "method", None)
    extras = getattr(ctx, "extras", {}) or {}
    timeout = float(extras.get("timeout", DEFAULT_TIMEOUT))
    extra_headers = extras.get("extra_headers")
    proxy_to_ollama(handler, config, method=method, timeout=timeout, extra_headers=extra_headers)


# ---------------------------------------------------------------------------
# Response writing helpers
# ---------------------------------------------------------------------------


def write_proxy_response(handler: HandlerLike, response: ProxyResponse) -> None:
    payload = response.body or b""
    handler.send_response(response.status)

    headers = dict(response.headers or {})
    headers.setdefault("Content-Type", response.content_type)
    headers["Content-Length"] = str(len(payload))

    for key, value in headers.items():
        if value is None:
            continue
        handler.send_header(str(key), str(value))
    handler.end_headers()
    if payload:
        handler.wfile.write(payload)



def send_json_error(handler: HandlerLike, payload: Mapping[str, Any], *, status: int = 500) -> None:
    body = json.dumps(dict(payload), ensure_ascii=False, indent=2).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _select_response_headers(source_headers: Any) -> dict[str, str]:
    headers: dict[str, str] = {}
    if not source_headers:
        return headers

    for key, value in source_headers.items():
        lowered = str(key).lower()
        if lowered in HOP_BY_HOP_HEADERS:
            continue
        if lowered == "content-length":
            continue
        if lowered in DEFAULT_ALLOWED_RESPONSE_HEADERS or lowered.startswith("x-"):
            headers[str(key)] = str(value)

    return headers


__all__ = [
    "ConfigLike",
    "HandlerLike",
    "ProxyError",
    "ProxyRequest",
    "ProxyResponse",
    "UpstreamConnectionError",
    "UpstreamHTTPError",
    "build_proxy_request",
    "build_target_url",
    "collect_forward_headers",
    "forward_request",
    "infer_method",
    "normalize_base_url",
    "proxy_route_handler",
    "proxy_to_ollama",
    "read_request_body",
    "send_json_error",
    "should_read_body",
    "write_proxy_response",
]
