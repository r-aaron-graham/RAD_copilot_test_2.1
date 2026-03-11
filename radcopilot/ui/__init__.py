from __future__ import annotations

"""
radcopilot.ui

Static UI asset helpers for RadCopilot Local.

Purpose
-------
This package currently contains browser assets only:

- index.html
- app.js
- templates.js
- styles.css

This __init__.py provides a lightweight Python surface for the server layer to:
- resolve the UI directory
- resolve asset paths safely
- list known assets
- read UI files
- infer content types for static serving

Design goals
------------
- import-safe
- no side effects
- no network access
- standard-library only
"""

from pathlib import Path
from typing import Any, Iterator
import mimetypes

__title__ = "radcopilot.ui"
__description__ = "Static browser UI assets for RadCopilot Local"

PACKAGE_DIR = Path(__file__).resolve().parent
UI_DIR = PACKAGE_DIR

INDEX_HTML = "index.html"
APP_JS = "app.js"
TEMPLATES_JS = "templates.js"
STYLES_CSS = "styles.css"

KNOWN_ASSETS: tuple[str, ...] = (
    INDEX_HTML,
    APP_JS,
    TEMPLATES_JS,
    STYLES_CSS,
)

# Register useful MIME types explicitly in case platform defaults vary.
mimetypes.add_type("text/javascript", ".js")
mimetypes.add_type("text/css", ".css")
mimetypes.add_type("text/html", ".html")


def get_ui_dir() -> Path:
    """Return the absolute path to the UI asset directory."""
    return UI_DIR


def get_index_path() -> Path:
    """Return the absolute path to index.html."""
    return UI_DIR / INDEX_HTML


def list_ui_assets(existing_only: bool = True) -> list[dict[str, Any]]:
    """
    Return metadata for known UI assets.

    Parameters
    ----------
    existing_only:
        If True, only include files currently present on disk.
    """
    items: list[dict[str, Any]] = []
    for name in KNOWN_ASSETS:
        path = UI_DIR / name
        exists = path.exists() and path.is_file()
        if existing_only and not exists:
            continue

        size_bytes = path.stat().st_size if exists else 0
        items.append(
            {
                "name": name,
                "path": str(path),
                "exists": exists,
                "size_bytes": size_bytes,
                "content_type": get_content_type(name),
            }
        )
    return items


def iter_ui_asset_paths(existing_only: bool = True) -> Iterator[Path]:
    """Yield absolute paths for known UI assets."""
    for name in KNOWN_ASSETS:
        path = UI_DIR / name
        if existing_only and not path.is_file():
            continue
        yield path


def asset_exists(name: str) -> bool:
    """Return True if a named UI asset exists."""
    try:
        return get_asset_path(name, must_exist=True).is_file()
    except FileNotFoundError:
        return False


def get_asset_path(name: str, *, must_exist: bool = False) -> Path:
    """
    Resolve a UI asset name to an absolute path.

    Security behavior
    -----------------
    - rejects path traversal
    - only allows files inside the UI package directory
    """
    normalized = str(name or "").strip().replace("\\", "/")
    if not normalized:
        raise ValueError("Asset name is required.")

    # Prevent path traversal and nested arbitrary access.
    if normalized.startswith("/") or ".." in normalized.split("/"):
        raise ValueError(f"Unsafe asset path: {name}")

    path = (UI_DIR / normalized).resolve()
    if UI_DIR not in (path, *path.parents):
        raise ValueError(f"Asset path escapes UI directory: {name}")

    if must_exist and not path.is_file():
        raise FileNotFoundError(f"UI asset not found: {name}")

    return path


def get_content_type(name_or_path: str | Path) -> str:
    """Return the best-guess content type for a UI asset."""
    value = str(name_or_path)
    content_type, _ = mimetypes.guess_type(value)

    if content_type:
        return content_type

    suffix = Path(value).suffix.lower()
    if suffix == ".html":
        return "text/html; charset=utf-8"
    if suffix == ".css":
        return "text/css; charset=utf-8"
    if suffix == ".js":
        return "text/javascript; charset=utf-8"
    if suffix == ".json":
        return "application/json; charset=utf-8"
    if suffix == ".svg":
        return "image/svg+xml"

    return "application/octet-stream"


def read_asset_text(name: str, *, encoding: str = "utf-8") -> str:
    """Read a UI asset as text."""
    path = get_asset_path(name, must_exist=True)
    return path.read_text(encoding=encoding)


def read_asset_bytes(name: str) -> bytes:
    """Read a UI asset as bytes."""
    path = get_asset_path(name, must_exist=True)
    return path.read_bytes()


def load_index_html(*, encoding: str = "utf-8") -> str:
    """Read and return the main UI HTML document."""
    return get_index_path().read_text(encoding=encoding)


def ui_asset_manifest(existing_only: bool = True) -> dict[str, Any]:
    """Return a JSON-serializable manifest of UI assets."""
    assets = list_ui_assets(existing_only=existing_only)
    return {
        "name": __title__,
        "description": __description__,
        "ui_dir": str(UI_DIR),
        "index_path": str(get_index_path()),
        "asset_count": len(assets),
        "assets": assets,
    }


def get_ui_package_info() -> dict[str, Any]:
    """Return lightweight package metadata for debugging and diagnostics."""
    return {
        "name": __title__,
        "description": __description__,
        "ui_dir": str(UI_DIR),
        "known_assets": list(KNOWN_ASSETS),
        "existing_assets": [item["name"] for item in list_ui_assets(existing_only=True)],
    }


__all__ = [
    "__title__",
    "__description__",
    "PACKAGE_DIR",
    "UI_DIR",
    "INDEX_HTML",
    "APP_JS",
    "TEMPLATES_JS",
    "STYLES_CSS",
    "KNOWN_ASSETS",
    "asset_exists",
    "get_asset_path",
    "get_content_type",
    "get_index_path",
    "get_ui_dir",
    "get_ui_package_info",
    "iter_ui_asset_paths",
    "list_ui_assets",
    "load_index_html",
    "read_asset_bytes",
    "read_asset_text",
    "ui_asset_manifest",
]
