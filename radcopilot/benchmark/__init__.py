from __future__ import annotations

"""
radcopilot.benchmark

Import-safe public surface for RadCopilot benchmark components.

Purpose
-------
This package exposes benchmark loading and scoring helpers from one place while
staying tolerant of partial refactor states.

Design goals
------------
- import-safe
- no startup side effects
- no network access
- standard-library only
- re-export stable symbols from benchmark submodules when available
"""

from importlib import import_module
from types import ModuleType
from typing import Any

__title__ = "radcopilot.benchmark"
__description__ = "Benchmark dataset loading and scoring exports for RadCopilot Local"

__all__: list[str] = []


def _export(name: str, value: Any) -> Any:
    """Register one exported symbol."""
    globals()[name] = value
    if name not in __all__:
        __all__.append(name)
    return value


def _safe_import(module_name: str) -> ModuleType | None:
    """
    Import one sibling benchmark module safely.

    Returns None instead of raising if the module is missing or currently broken.
    """
    try:
        return import_module(f".{module_name}", __name__)
    except Exception:
        return None


def _register_module(module_name: str, alias: str | None = None) -> ModuleType | None:
    """
    Import a benchmark module, export the module object, and re-export its
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

loader = _register_module("loader")
scorer = _register_module("scorer")


# ---------------------------------------------------------------------------
# Availability helpers
# ---------------------------------------------------------------------------

def available_benchmark_modules() -> dict[str, bool]:
    """Return availability status for known benchmark modules."""
    return {
        "loader": loader is not None,
        "scorer": scorer is not None,
    }


def benchmark_module_available(name: str) -> bool:
    """Return True if a named benchmark module is currently importable."""
    if not name:
        return False
    value = globals().get(str(name).strip())
    return isinstance(value, ModuleType)


def get_benchmark_module(name: str) -> ModuleType | None:
    """Return a benchmark module by name if available, otherwise None."""
    if not name:
        return None
    value = globals().get(str(name).strip())
    return value if isinstance(value, ModuleType) else None


def get_benchmark_package_info() -> dict[str, Any]:
    """Return lightweight package metadata and module availability."""
    return {
        "name": __title__,
        "description": __description__,
        "modules": available_benchmark_modules(),
        "exports": sorted(set(__all__)),
    }


# ---------------------------------------------------------------------------
# Convenience flags
# ---------------------------------------------------------------------------

def loader_available() -> bool:
    """Return True when the benchmark loader module is importable."""
    return loader is not None


def scorer_available() -> bool:
    """Return True when the benchmark scorer module is importable."""
    return scorer is not None


_export("available_benchmark_modules", available_benchmark_modules)
_export("benchmark_module_available", benchmark_module_available)
_export("get_benchmark_module", get_benchmark_module)
_export("get_benchmark_package_info", get_benchmark_package_info)
_export("loader_available", loader_available)
_export("scorer_available", scorer_available)
