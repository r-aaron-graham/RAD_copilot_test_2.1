from __future__ import annotations

"""
radcopilot.server

Import-safe public surface for the RadCopilot server package.

Goals
-----
- expose the most useful server-layer symbols from one place
- avoid heavy startup side effects
- remain tolerant while the refactor is still in progress
- avoid crashing package import if one optional server submodule is incomplete

This file should be safe to import from:
- radcopilot.main
- future tests
- future packaging/CLI entrypoints
- interactive debugging sessions

It intentionally does not start the HTTP server or touch the network.
"""

from typing import Any


# ---------------------------------------------------------------------------
# Package metadata
# ---------------------------------------------------------------------------

__title__ = "radcopilot.server"
__description__ = "Server-layer exports for RadCopilot Local"
__all__: list[str] = []


def _export(name: str, value: Any) -> Any:
    """Register an exported symbol and return it."""
    globals()[name] = value
    if name not in __all__:
        __all__.append(name)
    return value


def get_server_package_info() -> dict[str, Any]:
    """Return lightweight package metadata and component availability."""
    return {
        "name": __title__,
        "description": __description__,
        "components": available_server_components(),
    }


# ---------------------------------------------------------------------------
# app.py
# ---------------------------------------------------------------------------

try:
    from .app import ConfigLike as AppConfigLike
    from .app import create_handler

    _export("AppConfigLike", AppConfigLike)
    _export("create_handler", create_handler)
except Exception:  # pragma: no cover - tolerate partial refactor state
    _export("AppConfigLike", None)
    _export("create_handler", None)


# ---------------------------------------------------------------------------
# routes.py
# ---------------------------------------------------------------------------

try:
    from .routes import ConfigLike as RoutesConfigLike
    from .routes import CoreRouteHandlers
    from .routes import HandlerLike as RoutesHandlerLike
    from .routes import MethodNotAllowed
    from .routes import Route
    from .routes import RouteContext
    from .routes import RouteError
    from .routes import RouteMatch
    from .routes import RouteNotFound
    from .routes import RouteRegistry
    from .routes import build_core_registry
    from .routes import describe_core_routes
    from .routes import dispatch_request
    from .routes import match_pattern
    from .routes import parse_request_path
    from .routes import send_bytes
    from .routes import send_json
    from .routes import send_text

    _export("RoutesConfigLike", RoutesConfigLike)
    _export("CoreRouteHandlers", CoreRouteHandlers)
    _export("RoutesHandlerLike", RoutesHandlerLike)
    _export("MethodNotAllowed", MethodNotAllowed)
    _export("Route", Route)
    _export("RouteContext", RouteContext)
    _export("RouteError", RouteError)
    _export("RouteMatch", RouteMatch)
    _export("RouteNotFound", RouteNotFound)
    _export("RouteRegistry", RouteRegistry)
    _export("build_core_registry", build_core_registry)
    _export("describe_core_routes", describe_core_routes)
    _export("dispatch_request", dispatch_request)
    _export("match_pattern", match_pattern)
    _export("parse_request_path", parse_request_path)
    _export("send_bytes", send_bytes)
    _export("send_json", send_json)
    _export("send_text", send_text)
except Exception:  # pragma: no cover
    _export("RoutesConfigLike", None)
    _export("CoreRouteHandlers", None)
    _export("RoutesHandlerLike", None)
    _export("MethodNotAllowed", None)
    _export("Route", None)
    _export("RouteContext", None)
    _export("RouteError", None)
    _export("RouteMatch", None)
    _export("RouteNotFound", None)
    _export("RouteRegistry", None)
    _export("build_core_registry", None)
    _export("describe_core_routes", None)
    _export("dispatch_request", None)
    _export("match_pattern", None)
    _export("parse_request_path", None)
    _export("send_bytes", None)
    _export("send_json", None)
    _export("send_text", None)


# ---------------------------------------------------------------------------
# proxy.py
# ---------------------------------------------------------------------------

try:
    from .proxy import ConfigLike as ProxyConfigLike
    from .proxy import DEFAULT_ALLOWED_REQUEST_HEADERS
    from .proxy import DEFAULT_ALLOWED_RESPONSE_HEADERS
    from .proxy import DEFAULT_TIMEOUT
    from .proxy import DEFAULT_USER_AGENT
    from .proxy import HOP_BY_HOP_HEADERS
    from .proxy import HandlerLike as ProxyHandlerLike
    from .proxy import ProxyError
    from .proxy import ProxyRequest
    from .proxy import ProxyResponse
    from .proxy import UpstreamConnectionError
    from .proxy import UpstreamHTTPError
    from .proxy import build_proxy_request
    from .proxy import build_target_url
    from .proxy import collect_forward_headers
    from .proxy import forward_request
    from .proxy import infer_method
    from .proxy import normalize_base_url
    from .proxy import proxy_route_handler
    from .proxy import proxy_to_ollama
    from .proxy import read_request_body
    from .proxy import send_json_error
    from .proxy import should_read_body
    from .proxy import write_proxy_response

    _export("ProxyConfigLike", ProxyConfigLike)
    _export("DEFAULT_ALLOWED_REQUEST_HEADERS", DEFAULT_ALLOWED_REQUEST_HEADERS)
    _export("DEFAULT_ALLOWED_RESPONSE_HEADERS", DEFAULT_ALLOWED_RESPONSE_HEADERS)
    _export("DEFAULT_TIMEOUT", DEFAULT_TIMEOUT)
    _export("DEFAULT_USER_AGENT", DEFAULT_USER_AGENT)
    _export("HOP_BY_HOP_HEADERS", HOP_BY_HOP_HEADERS)
    _export("ProxyHandlerLike", ProxyHandlerLike)
    _export("ProxyError", ProxyError)
    _export("ProxyRequest", ProxyRequest)
    _export("ProxyResponse", ProxyResponse)
    _export("UpstreamConnectionError", UpstreamConnectionError)
    _export("UpstreamHTTPError", UpstreamHTTPError)
    _export("build_proxy_request", build_proxy_request)
    _export("build_target_url", build_target_url)
    _export("collect_forward_headers", collect_forward_headers)
    _export("forward_request", forward_request)
    _export("infer_method", infer_method)
    _export("normalize_base_url", normalize_base_url)
    _export("proxy_route_handler", proxy_route_handler)
    _export("proxy_to_ollama", proxy_to_ollama)
    _export("read_request_body", read_request_body)
    _export("send_json_error", send_json_error)
    _export("should_read_body", should_read_body)
    _export("write_proxy_response", write_proxy_response)
except Exception:  # pragma: no cover
    _export("ProxyConfigLike", None)
    _export("DEFAULT_ALLOWED_REQUEST_HEADERS", None)
    _export("DEFAULT_ALLOWED_RESPONSE_HEADERS", None)
    _export("DEFAULT_TIMEOUT", None)
    _export("DEFAULT_USER_AGENT", None)
    _export("HOP_BY_HOP_HEADERS", None)
    _export("ProxyHandlerLike", None)
    _export("ProxyError", None)
    _export("ProxyRequest", None)
    _export("ProxyResponse", None)
    _export("UpstreamConnectionError", None)
    _export("UpstreamHTTPError", None)
    _export("build_proxy_request", None)
    _export("build_target_url", None)
    _export("collect_forward_headers", None)
    _export("forward_request", None)
    _export("infer_method", None)
    _export("normalize_base_url", None)
    _export("proxy_route_handler", None)
    _export("proxy_to_ollama", None)
    _export("read_request_body", None)
    _export("send_json_error", None)
    _export("should_read_body", None)
    _export("write_proxy_response", None)


# ---------------------------------------------------------------------------
# report_routes.py
# ---------------------------------------------------------------------------

try:
    from .report_routes import ConfigLike as ReportRoutesConfigLike
    from .report_routes import build_report_registry
    from .report_routes import describe_report_routes
    from .report_routes import handle_report_generate
    from .report_routes import handle_report_guidelines
    from .report_routes import handle_report_template_detail
    from .report_routes import handle_report_templates
    from .report_routes import handle_report_validate
    from .report_routes import register_report_routes

    _export("ReportRoutesConfigLike", ReportRoutesConfigLike)
    _export("build_report_registry", build_report_registry)
    _export("describe_report_routes", describe_report_routes)
    _export("handle_report_generate", handle_report_generate)
    _export("handle_report_guidelines", handle_report_guidelines)
    _export("handle_report_template_detail", handle_report_template_detail)
    _export("handle_report_templates", handle_report_templates)
    _export("handle_report_validate", handle_report_validate)
    _export("register_report_routes", register_report_routes)
except Exception:  # pragma: no cover
    _export("ReportRoutesConfigLike", None)
    _export("build_report_registry", None)
    _export("describe_report_routes", None)
    _export("handle_report_generate", None)
    _export("handle_report_guidelines", None)
    _export("handle_report_template_detail", None)
    _export("handle_report_templates", None)
    _export("handle_report_validate", None)
    _export("register_report_routes", None)


# ---------------------------------------------------------------------------
# Availability helpers
# ---------------------------------------------------------------------------

def available_server_components() -> dict[str, bool]:
    """
    Report which major server components imported successfully.

    This is useful while the modular refactor is still incomplete.
    """
    return {
        "app": create_handler is not None,
        "routes": RouteRegistry is not None and dispatch_request is not None,
        "proxy": proxy_to_ollama is not None and forward_request is not None,
        "report_routes": build_report_registry is not None and register_report_routes is not None,
    }


def report_routes_available() -> bool:
    """Return True when report route helpers are importable."""
    return bool(build_report_registry is not None and register_report_routes is not None)


def routing_available() -> bool:
    """Return True when the reusable routing layer is importable."""
    return bool(RouteRegistry is not None and dispatch_request is not None)


def proxy_available() -> bool:
    """Return True when the shared proxy layer is importable."""
    return bool(proxy_to_ollama is not None and forward_request is not None)


def app_available() -> bool:
    """Return True when the server handler factory is importable."""
    return bool(create_handler is not None)


_export("available_server_components", available_server_components)
_export("app_available", app_available)
_export("get_server_package_info", get_server_package_info)
_export("proxy_available", proxy_available)
_export("report_routes_available", report_routes_available)
_export("routing_available", routing_available)
