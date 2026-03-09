"""
Shared configuration for the modular RadCopilot refactor.

This module centralizes runtime defaults, path resolution, and environment
overrides so that the launcher, server, RAG, logging, and service layers all
use one source of truth.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on", "y"}


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value.strip())
    except Exception:
        return default


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return float(value.strip())
    except Exception:
        return default


def _env_list(name: str, default: Iterable[str]) -> List[str]:
    value = os.getenv(name)
    if value is None or not value.strip():
        return list(default)
    return [part.strip() for part in value.split(",") if part.strip()]


def _env_path(name: str, default: Path) -> Path:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    return Path(value).expanduser().resolve()


def _project_root() -> Path:
    return Path(__file__).resolve().parent


@dataclass(slots=True)
class AppPaths:
    project_root: Path
    ui_dir: Path
    data_dir: Path
    logs_dir: Path
    rag_dir: Path
    temp_dir: Path
    error_log_file: Path
    rag_store_file: Path
    rag_rating_file: Path
    benchmark_dir: Path
    dataset_search_roots: List[Path] = field(default_factory=list)

    def ensure_dirs(self) -> None:
        for path in {
            self.ui_dir,
            self.data_dir,
            self.logs_dir,
            self.rag_dir,
            self.temp_dir,
            self.benchmark_dir,
        }:
            path.mkdir(parents=True, exist_ok=True)

    def as_dict(self) -> Dict[str, str]:
        payload = asdict(self)
        payload["dataset_search_roots"] = [str(p) for p in self.dataset_search_roots]
        return {
            key: str(value) if isinstance(value, Path) else value
            for key, value in payload.items()
        }


@dataclass(slots=True)
class OllamaSettings:
    base_url: str = "http://localhost:11434"
    default_model: str = "llama3.1:8b"
    generate_endpoint: str = "/api/generate"
    chat_endpoint: str = "/api/chat"
    tags_endpoint: str = "/api/tags"
    embeddings_endpoint: str = "/api/embeddings"
    timeout_seconds: float = 120.0
    stream_timeout_seconds: float = 180.0
    connect_timeout_seconds: float = 10.0
    retries: int = 2
    auto_start: bool = True
    serve_command: List[str] = field(default_factory=lambda: ["ollama", "serve"])
    extra_env: Dict[str, str] = field(default_factory=lambda: {"OLLAMA_ORIGINS": "*"})

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class RagSettings:
    enabled: bool = True
    build_startup_index: bool = True
    max_examples: int = 3
    similarity_threshold: float = 0.12
    max_records: int = 100000
    use_sklearn_if_available: bool = True
    allowed_extensions: List[str] = field(
        default_factory=lambda: [".txt", ".csv", ".xml", ".json", ".jsonl", ".tgz", ".gz", ".tar"]
    )

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class WhisperSettings:
    enabled: bool = True
    model_name: str = "base"
    auto_load: bool = False
    temp_suffix: str = ".webm"
    keep_temp_files: bool = False

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ServerSettings:
    host: str = "127.0.0.1"
    port: int = 7432
    open_browser: bool = True
    browser_open_delay_seconds: float = 1.0
    allow_reuse_address: bool = True
    daemon_threads: bool = True
    static_index: str = "index.html"

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class LoggingSettings:
    enabled: bool = True
    max_recent_records: int = 200
    max_detail_chars: int = 8000
    max_traceback_chars: int = 16000
    json_indent: Optional[int] = None

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class UISettings:
    default_mode: str = "report"
    default_template_id: str = "ct-chest"
    save_history: bool = True
    max_history_items: int = 20
    enable_phi_scrub_notice: bool = True

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class AppConfig:
    app_name: str
    version: str
    debug: bool
    environment: str
    paths: AppPaths
    server: ServerSettings
    ollama: OllamaSettings
    rag: RagSettings
    whisper: WhisperSettings
    logging: LoggingSettings
    ui: UISettings

    def ensure_runtime_dirs(self) -> None:
        self.paths.ensure_dirs()

    def to_public_dict(self) -> Dict[str, Any]:
        return {
            "app_name": self.app_name,
            "version": self.version,
            "debug": self.debug,
            "environment": self.environment,
            "paths": self.paths.as_dict(),
            "server": self.server.as_dict(),
            "ollama": self.ollama.as_dict(),
            "rag": self.rag.as_dict(),
            "whisper": self.whisper.as_dict(),
            "logging": self.logging.as_dict(),
            "ui": self.ui.as_dict(),
        }


def _default_dataset_roots(project_root: Path) -> List[Path]:
    home = Path.home()
    downloads = home / "Downloads"
    return [
        project_root / "data" / "datasets",
        downloads / "radcopilot_datasets",
        downloads,
    ]


def build_paths(project_root: Optional[Path] = None) -> AppPaths:
    root = (project_root or _project_root()).resolve()
    data_dir = _env_path("RADCOPILOT_DATA_DIR", root / "data")
    logs_dir = _env_path("RADCOPILOT_LOGS_DIR", data_dir / "logs")
    rag_dir = _env_path("RADCOPILOT_RAG_DIR", data_dir / "rag")
    temp_dir = _env_path("RADCOPILOT_TEMP_DIR", data_dir / "tmp")
    benchmark_dir = _env_path("RADCOPILOT_BENCHMARK_DIR", data_dir / "benchmarks")
    ui_dir = _env_path("RADCOPILOT_UI_DIR", root / "ui")
    search_roots = [
        Path(p).expanduser().resolve()
        for p in _env_list(
            "RADCOPILOT_DATASET_ROOTS",
            [str(p) for p in _default_dataset_roots(root)],
        )
    ]

    return AppPaths(
        project_root=root,
        ui_dir=ui_dir,
        data_dir=data_dir,
        logs_dir=logs_dir,
        rag_dir=rag_dir,
        temp_dir=temp_dir,
        error_log_file=_env_path("RADCOPILOT_ERROR_LOG_FILE", logs_dir / "radcopilot_errors.jsonl"),
        rag_store_file=_env_path("RADCOPILOT_RAG_STORE_FILE", rag_dir / "rag_library.json"),
        rag_rating_file=_env_path("RADCOPILOT_RAG_RATING_FILE", rag_dir / "rag_ratings.jsonl"),
        benchmark_dir=benchmark_dir,
        dataset_search_roots=search_roots,
    )


def load_config(project_root: Optional[Path] = None) -> AppConfig:
    paths = build_paths(project_root)

    config = AppConfig(
        app_name=os.getenv("RADCOPILOT_APP_NAME", "RadCopilot Local"),
        version=os.getenv("RADCOPILOT_VERSION", "0.1.0"),
        debug=_env_bool("RADCOPILOT_DEBUG", False),
        environment=os.getenv("RADCOPILOT_ENV", "local"),
        paths=paths,
        server=ServerSettings(
            host=os.getenv("RADCOPILOT_HOST", "127.0.0.1"),
            port=_env_int("RADCOPILOT_PORT", 7432),
            open_browser=_env_bool("RADCOPILOT_OPEN_BROWSER", True),
            browser_open_delay_seconds=_env_float("RADCOPILOT_BROWSER_DELAY", 1.0),
            allow_reuse_address=_env_bool("RADCOPILOT_ALLOW_REUSE_ADDRESS", True),
            daemon_threads=_env_bool("RADCOPILOT_DAEMON_THREADS", True),
            static_index=os.getenv("RADCOPILOT_STATIC_INDEX", "index.html"),
        ),
        ollama=OllamaSettings(
            base_url=os.getenv("RADCOPILOT_OLLAMA_URL", "http://localhost:11434"),
            default_model=os.getenv("RADCOPILOT_OLLAMA_MODEL", "llama3.1:8b"),
            generate_endpoint=os.getenv("RADCOPILOT_OLLAMA_GENERATE_ENDPOINT", "/api/generate"),
            chat_endpoint=os.getenv("RADCOPILOT_OLLAMA_CHAT_ENDPOINT", "/api/chat"),
            tags_endpoint=os.getenv("RADCOPILOT_OLLAMA_TAGS_ENDPOINT", "/api/tags"),
            embeddings_endpoint=os.getenv("RADCOPILOT_OLLAMA_EMBEDDINGS_ENDPOINT", "/api/embeddings"),
            timeout_seconds=_env_float("RADCOPILOT_OLLAMA_TIMEOUT", 120.0),
            stream_timeout_seconds=_env_float("RADCOPILOT_OLLAMA_STREAM_TIMEOUT", 180.0),
            connect_timeout_seconds=_env_float("RADCOPILOT_OLLAMA_CONNECT_TIMEOUT", 10.0),
            retries=_env_int("RADCOPILOT_OLLAMA_RETRIES", 2),
            auto_start=_env_bool("RADCOPILOT_OLLAMA_AUTOSTART", True),
            serve_command=_env_list("RADCOPILOT_OLLAMA_SERVE_COMMAND", ["ollama", "serve"]),
            extra_env={"OLLAMA_ORIGINS": os.getenv("RADCOPILOT_OLLAMA_ORIGINS", "*")},
        ),
        rag=RagSettings(
            enabled=_env_bool("RADCOPILOT_RAG_ENABLED", True),
            build_startup_index=_env_bool("RADCOPILOT_RAG_BUILD_STARTUP", True),
            max_examples=_env_int("RADCOPILOT_RAG_MAX_EXAMPLES", 3),
            similarity_threshold=_env_float("RADCOPILOT_RAG_SIMILARITY_THRESHOLD", 0.12),
            max_records=_env_int("RADCOPILOT_RAG_MAX_RECORDS", 100000),
            use_sklearn_if_available=_env_bool("RADCOPILOT_RAG_USE_SKLEARN", True),
            allowed_extensions=_env_list(
                "RADCOPILOT_RAG_ALLOWED_EXTENSIONS",
                [".txt", ".csv", ".xml", ".json", ".jsonl", ".tgz", ".gz", ".tar"],
            ),
        ),
        whisper=WhisperSettings(
            enabled=_env_bool("RADCOPILOT_WHISPER_ENABLED", True),
            model_name=os.getenv("RADCOPILOT_WHISPER_MODEL", "base"),
            auto_load=_env_bool("RADCOPILOT_WHISPER_AUTOLOAD", False),
            temp_suffix=os.getenv("RADCOPILOT_WHISPER_TEMP_SUFFIX", ".webm"),
            keep_temp_files=_env_bool("RADCOPILOT_WHISPER_KEEP_TEMP_FILES", False),
        ),
        logging=LoggingSettings(
            enabled=_env_bool("RADCOPILOT_LOGGING_ENABLED", True),
            max_recent_records=_env_int("RADCOPILOT_LOG_MAX_RECENT", 200),
            max_detail_chars=_env_int("RADCOPILOT_LOG_MAX_DETAIL_CHARS", 8000),
            max_traceback_chars=_env_int("RADCOPILOT_LOG_MAX_TRACEBACK_CHARS", 16000),
            json_indent=(
                _env_int("RADCOPILOT_LOG_JSON_INDENT", 2)
                if os.getenv("RADCOPILOT_LOG_JSON_INDENT")
                else None
            ),
        ),
        ui=UISettings(
            default_mode=os.getenv("RADCOPILOT_DEFAULT_MODE", "report"),
            default_template_id=os.getenv("RADCOPILOT_DEFAULT_TEMPLATE", "ct-chest"),
            save_history=_env_bool("RADCOPILOT_SAVE_HISTORY", True),
            max_history_items=_env_int("RADCOPILOT_MAX_HISTORY", 20),
            enable_phi_scrub_notice=_env_bool("RADCOPILOT_ENABLE_PHI_NOTICE", True),
        ),
    )

    config.ensure_runtime_dirs()
    return config


DEFAULT_CONFIG = load_config()


__all__ = [
    "AppConfig",
    "AppPaths",
    "DEFAULT_CONFIG",
    "LoggingSettings",
    "OllamaSettings",
    "RagSettings",
    "ServerSettings",
    "UISettings",
    "WhisperSettings",
    "build_paths",
    "load_config",
]
