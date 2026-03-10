from __future__ import annotations

"""
radcopilot.rag

Import-safe public surface for RadCopilot retrieval-augmented generation
components.

Goals
-----
- expose the RAG package modules from one place
- re-export stable symbols from each submodule when available
- avoid import-time side effects such as index building or model startup
- tolerate partial refactor states where one submodule may be missing or broken
"""

from importlib import import_module
from types import ModuleType
from typing import Any

__title__ = "radcopilot.rag"
__description__ = "Retrieval-augmented generation exports for RadCopilot Local"

__all__: list[str] = []


def _export(name: str, value: Any) -> Any:
    """Register one exported symbol."""
    globals()[name] = value
    if name not in __all__:
        __all__.append(name)
    return value


def _safe_import(module_name: str) -> ModuleType | None:
    """
    Import one sibling RAG module safely.

    Returns None instead of raising if the module is missing or currently broken.
    """
    try:
        return import_module(f".{module_name}", __name__)
    except Exception:
        return None


def _register_module(module_name: str, alias: str | None = None) -> ModuleType | None:
    """
    Import a RAG module, export the module object, and re-export its __all__
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

library = _register_module("library")
parser = _register_module("parser")
rating = _register_module("rating")
startup_index = _register_module("startup_index")
trainer = _register_module("trainer")


# ---------------------------------------------------------------------------
# Availability helpers
# ---------------------------------------------------------------------------

def available_rag_modules() -> dict[str, bool]:
    """Return availability status for known RAG modules."""
    return {
        "library": library is not None,
        "parser": parser is not None,
        "rating": rating is not None,
        "startup_index": startup_index is not None,
        "trainer": trainer is not None,
    }


def rag_module_available(name: str) -> bool:
    """Return True if a named RAG module is currently importable."""
    if not name:
        return False
    value = globals().get(str(name).strip())
    return isinstance(value, ModuleType)


def get_rag_module(name: str) -> ModuleType | None:
    """Return a RAG module by name if available, otherwise None."""
    if not name:
        return None
    value = globals().get(str(name).strip())
    return value if isinstance(value, ModuleType) else None


def get_rag_package_info() -> dict[str, Any]:
    """Return lightweight package metadata and module availability."""
    return {
        "name": __title__,
        "description": __description__,
        "modules": available_rag_modules(),
        "exports": sorted(set(__all__)),
    }


# ---------------------------------------------------------------------------
# Convenience flags
# ---------------------------------------------------------------------------

def library_available() -> bool:
    """Return True when the retrieval library module is importable."""
    return library is not None


def parser_available() -> bool:
    """Return True when the RAG parser module is importable."""
    return parser is not None


def rating_available() -> bool:
    """Return True when the RAG rating module is importable."""
    return rating is not None


def startup_index_available() -> bool:
    """Return True when the startup indexing module is importable."""
    return startup_index is not None


def trainer_available() -> bool:
    """Return True when the RAG trainer module is importable."""
    return trainer is not None


_export("available_rag_modules", available_rag_modules)
_export("rag_module_available", rag_module_available)
_export("get_rag_module", get_rag_module)
_export("get_rag_package_info", get_rag_package_info)
_export("library_available", library_available)
_export("parser_available", parser_available)
_export("rating_available", rating_available)
_export("startup_index_available", startup_index_available)
_export("trainer_available", trainer_available)
