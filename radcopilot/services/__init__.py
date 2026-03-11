from __future__ import annotations

"""
radcopilot.services

Import-safe public surface for RadCopilot service modules.

Goals
-----
- expose service modules from one place
- re-export stable symbols from each service module when available
- avoid hard failures if one service module is incomplete during refactor
- avoid side effects such as model startup or network calls at import time
"""

from importlib import import_module
from types import ModuleType
from typing import Any

__title__ = "radcopilot.services"
__description__ = "Shared service-layer exports for RadCopilot Local"

__all__: list[str] = []


def _export(name: str, value: Any) -> Any:
    """Register one exported symbol."""
    globals()[name] = value
    if name not in __all__:
        __all__.append(name)
    return value


def _safe_import(module_name: str) -> ModuleType | None:
    """
    Import one sibling service module safely.

    Returns None instead of raising if the module is missing or currently broken.
    """
    try:
        return import_module(f".{module_name}", __name__)
    except Exception:
        return None


def _register_module(module_name: str, alias: str | None = None) -> ModuleType | None:
    """
    Import a service module, export the module object, and re-export its
    __all__ symbols if present.
    """
    module = _safe_import(module_name)
    public_name = alias or module_name
    _export(public_name, module)

    if module is None:
        return None

    exported = getattr(module, "__all__", None)
    if isinstance(exported, (list, tuple)):
        for name in exported:
            if not isinstance(name, str) or not name:
                continue
            try:
                value = getattr(module, name)
            except Exception:
                continue
            _export(name, value)

    return module


# ---------------------------------------------------------------------------
# Optional module imports
# ---------------------------------------------------------------------------

logging_service = _register_module("logging_service")
ollama_client = _register_module("ollama_client")
whisper_service = _register_module("whisper_service")
history_service = _register_module("history_service")


# ---------------------------------------------------------------------------
# Availability helpers
# ---------------------------------------------------------------------------

def available_services() -> dict[str, bool]:
    """Return availability status for known service modules."""
    return {
        "logging_service": logging_service is not None,
        "ollama_client": ollama_client is not None,
        "whisper_service": whisper_service is not None,
        "history_service": history_service is not None,
    }


def service_available(name: str) -> bool:
    """Return True if a named service module is currently importable."""
    if not name:
        return False
    value = globals().get(str(name).strip())
    return isinstance(value, ModuleType)


def get_service(name: str) -> ModuleType | None:
    """Return a service module by name if available, otherwise None."""
    if not name:
        return None
    value = globals().get(str(name).strip())
    return value if isinstance(value, ModuleType) else None


def get_services_package_info() -> dict[str, Any]:
    """Return lightweight package metadata and module availability."""
    return {
        "name": __title__,
        "description": __description__,
        "modules": available_services(),
        "exports": sorted(set(__all__)),
    }


# ---------------------------------------------------------------------------
# Convenience flags
# ---------------------------------------------------------------------------

def logging_available() -> bool:
    """Return True when the logging service module is importable."""
    return logging_service is not None


def ollama_available() -> bool:
    """Return True when the Ollama client module is importable."""
    return ollama_client is not None


def whisper_available() -> bool:
    """Return True when the Whisper service module is importable."""
    return whisper_service is not None


def history_available() -> bool:
    """Return True when the history service module is importable."""
    return history_service is not None


_export("available_services", available_services)
_export("service_available", service_available)
_export("get_service", get_service)
_export("get_services_package_info", get_services_package_info)
_export("logging_available", logging_available)
_export("ollama_available", ollama_available)
_export("whisper_available", whisper_available)
_export("history_available", history_available)
