from __future__ import annotations

"""
radcopilot.report

Import-safe public surface for RadCopilot report-generation components.

Goals
-----
- expose report submodules from one place
- re-export stable symbols from each submodule when available
- avoid import-time side effects
- tolerate partial refactor states where one submodule may be missing or broken
"""

from importlib import import_module
from types import ModuleType
from typing import Any

__title__ = "radcopilot.report"
__description__ = "Report generation, validation, fixing, and guideline exports for RadCopilot Local"

__all__: list[str] = []


def _export(name: str, value: Any) -> Any:
    """Register one exported symbol."""
    globals()[name] = value
    if name not in __all__:
        __all__.append(name)
    return value


def _safe_import(module_name: str) -> ModuleType | None:
    """
    Import one sibling report module safely.

    Returns None instead of raising if the module is missing or currently broken.
    """
    try:
        return import_module(f".{module_name}", __name__)
    except Exception:
        return None


def _register_module(module_name: str, alias: str | None = None) -> ModuleType | None:
    """
    Import a report module, export the module object, and re-export its __all__
    symbols if present.
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

fixer = _register_module("fixer")
generator = _register_module("generator")
guidelines = _register_module("guidelines")
validator = _register_module("validator")


# ---------------------------------------------------------------------------
# Availability helpers
# ---------------------------------------------------------------------------

def available_report_modules() -> dict[str, bool]:
    """Return availability status for known report modules."""
    return {
        "fixer": fixer is not None,
        "generator": generator is not None,
        "guidelines": guidelines is not None,
        "validator": validator is not None,
    }


def report_module_available(name: str) -> bool:
    """Return True if a named report module is currently importable."""
    if not name:
        return False
    value = globals().get(str(name).strip())
    return isinstance(value, ModuleType)


def get_report_module(name: str) -> ModuleType | None:
    """Return a report module by name if available, otherwise None."""
    if not name:
        return None
    value = globals().get(str(name).strip())
    return value if isinstance(value, ModuleType) else None


def get_report_package_info() -> dict[str, Any]:
    """Return lightweight package metadata and module availability."""
    return {
        "name": __title__,
        "description": __description__,
        "modules": available_report_modules(),
        "exports": sorted(set(__all__)),
    }


# ---------------------------------------------------------------------------
# Convenience flags
# ---------------------------------------------------------------------------

def fixer_available() -> bool:
    """Return True when the fixer module is importable."""
    return fixer is not None


def generator_available() -> bool:
    """Return True when the generator module is importable."""
    return generator is not None


def guidelines_available() -> bool:
    """Return True when the guidelines module is importable."""
    return guidelines is not None


def validator_available() -> bool:
    """Return True when the validator module is importable."""
    return validator is not None


_export("available_report_modules", available_report_modules)
_export("report_module_available", report_module_available)
_export("get_report_module", get_report_module)
_export("get_report_package_info", get_report_package_info)
_export("fixer_available", fixer_available)
_export("generator_available", generator_available)
_export("guidelines_available", guidelines_available)
_export("validator_available", validator_available)
