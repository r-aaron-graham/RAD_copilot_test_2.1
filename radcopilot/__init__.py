from __future__ import annotations

"""
RadCopilot package metadata and lightweight public exports.

This file is intentionally kept import-safe:
- no heavy runtime initialization
- no network/model startup
- no side effects beyond metadata exposure

It supports both direct package imports and tooling that expects a small,
stable package surface.
"""

__title__ = "radcopilot"
__description__ = "Local-first radiology AI workstation"
__url__ = "https://github.com/r-aaron-graham/RAD_copilot_test_2.1"
__author__ = "R. Aaron Graham"
__license__ = "Proprietary"
__version__ = "0.1.0"


def get_version() -> str:
    """Return the package version string."""
    return __version__


def get_package_info() -> dict[str, str]:
    """Return basic package metadata as a JSON-serializable dictionary."""
    return {
        "name": __title__,
        "version": __version__,
        "description": __description__,
        "url": __url__,
        "author": __author__,
        "license": __license__,
    }


# Keep imports narrow and import-safe.
# The current repo surface does not reliably expose UIConfig / ServerConfig /
# ModelConfig / DataConfig from config.py, so do not import symbols that may
# not exist.
try:
    from .config import AppConfig
except Exception:  # pragma: no cover
    AppConfig = None  # type: ignore[assignment]


# main.py exposes run() as the public runtime entrypoint.
# Do not import a nonexistent main() symbol.
try:
    from .main import run
except Exception:  # pragma: no cover
    run = None  # type: ignore[assignment]


__all__ = [
    "__title__",
    "__description__",
    "__url__",
    "__author__",
    "__license__",
    "__version__",
    "get_version",
    "get_package_info",
    "AppConfig",
    "run",
]
