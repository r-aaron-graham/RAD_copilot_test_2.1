from __future__ import annotations

"""
radcopilot.services.whisper_service

Optional local Whisper speech-to-text service for RadCopilot Local.

Goals:
- keep Whisper integration isolated from the HTTP server layer
- lazy-load the Whisper model only when needed
- support raw body, JSON-base64, and multipart uploads
- return normalized JSON-friendly transcription payloads
- fail gracefully when Whisper is not installed

This module intentionally uses only the Python standard library plus the
optional `whisper` dependency when it is available.
"""

from dataclasses import dataclass, field
import base64
import cgi
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
import threading
from typing import Any, BinaryIO, Protocol

from radcopilot.services.logging_service import log_event, log_exception

DEFAULT_MODEL_NAME = "base"
DEFAULT_TASK = "transcribe"
MAX_UPLOAD_BYTES = 100 * 1024 * 1024
_AUDIO_FIELD_CANDIDATES = ("file", "audio", "blob", "recording", "upload")


class ConfigLike(Protocol):
    """Minimal runtime config contract used by this module."""

    base_dir: Path
    log_file: Path


@dataclass(slots=True)
class WhisperStatus:
    """Normalized capability snapshot for the local Whisper service."""

    available: bool
    loaded: bool
    model_name: str = DEFAULT_MODEL_NAME
    backend: str = "whisper"
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "available": self.available,
            "loaded": self.loaded,
            "model_name": self.model_name,
            "backend": self.backend,
        }
        if self.detail:
            payload["detail"] = self.detail
        return payload


@dataclass(slots=True)
class TranscriptionResult:
    """Canonical transcription result returned by this service."""

    ok: bool
    text: str = ""
    model_name: str = DEFAULT_MODEL_NAME
    task: str = DEFAULT_TASK
    language: str = ""
    duration_seconds: float | None = None
    source_bytes: int = 0
    source_name: str = ""
    source_sha256: str = ""
    segments: list[dict[str, Any]] = field(default_factory=list)
    error: str = ""
    available: bool = True

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "ok": self.ok,
            "available": self.available,
            "model_name": self.model_name,
            "task": self.task,
            "text": self.text,
            "source_bytes": self.source_bytes,
            "segments": list(self.segments),
        }
        if self.language:
            payload["language"] = self.language
        if self.duration_seconds is not None:
            payload["duration_seconds"] = round(float(self.duration_seconds), 3)
        if self.source_name:
            payload["source_name"] = self.source_name
        if self.source_sha256:
            payload["source_sha256"] = self.source_sha256
        if self.error:
            payload["error"] = self.error
        return payload


_MODEL_LOCK = threading.RLock()
_LOADED_MODEL: Any | None = None
_LOADED_MODEL_NAME: str = ""
_LAST_IMPORT_ERROR: str = ""


def whisper_available() -> bool:
    """Return True when the `whisper` package can be imported."""
    global _LAST_IMPORT_ERROR
    try:
        import whisper  # noqa: F401

        _LAST_IMPORT_ERROR = ""
        return True
    except Exception as exc:
        _LAST_IMPORT_ERROR = f"{type(exc).__name__}: {exc}"
        return False


def get_status(model_name: str = DEFAULT_MODEL_NAME) -> dict[str, Any]:
    """Return a normalized availability / loaded-state payload."""
    available = whisper_available()
    status = WhisperStatus(
        available=available,
        loaded=bool(_LOADED_MODEL is not None and _LOADED_MODEL_NAME == model_name),
        model_name=model_name,
        detail="" if available else _LAST_IMPORT_ERROR,
    )
    return status.to_dict()


def load_model(model_name: str = DEFAULT_MODEL_NAME) -> Any:
    """Lazy-load and cache a Whisper model instance."""
    global _LOADED_MODEL, _LOADED_MODEL_NAME

    with _MODEL_LOCK:
        if _LOADED_MODEL is not None and _LOADED_MODEL_NAME == model_name:
            return _LOADED_MODEL

        try:
            import whisper  # type: ignore
        except Exception as exc:  # pragma: no cover - dependency controlled by runtime
            raise RuntimeError(f"Whisper not available: {type(exc).__name__}: {exc}") from exc

        model = whisper.load_model(model_name)
        _LOADED_MODEL = model
        _LOADED_MODEL_NAME = model_name
        return model


def unload_model() -> None:
    """Clear the cached Whisper model reference."""
    global _LOADED_MODEL, _LOADED_MODEL_NAME
    with _MODEL_LOCK:
        _LOADED_MODEL = None
        _LOADED_MODEL_NAME = ""


def transcribe_file(
    file_path: str | Path,
    *,
    model_name: str = DEFAULT_MODEL_NAME,
    task: str = DEFAULT_TASK,
    language: str | None = None,
    temperature: float = 0.0,
) -> dict[str, Any]:
    """Transcribe a local audio file path with Whisper."""
    path = Path(file_path).expanduser().resolve()
    if not path.exists() or not path.is_file():
        return TranscriptionResult(
            ok=False,
            available=whisper_available(),
            model_name=model_name,
            task=task,
            error=f"Audio file does not exist: {path}",
        ).to_dict()

    try:
        model = load_model(model_name)
    except Exception as exc:
        return TranscriptionResult(
            ok=False,
            available=False,
            model_name=model_name,
            task=task,
            error=str(exc),
        ).to_dict()

    kwargs: dict[str, Any] = {
        "task": task or DEFAULT_TASK,
        "temperature": float(temperature),
        "verbose": False,
    }
    if language:
        kwargs["language"] = language

    result = model.transcribe(str(path), **kwargs)
    text = normalize_text(str(result.get("text") or ""))
    segments = normalize_segments(result.get("segments"))
    language_out = str(result.get("language") or language or "")

    return TranscriptionResult(
        ok=True,
        text=text,
        model_name=model_name,
        task=kwargs["task"],
        language=language_out,
        source_bytes=int(path.stat().st_size),
        source_name=path.name,
        source_sha256=sha256_file(path),
        segments=segments,
    ).to_dict()


def transcribe_bytes(
    audio_bytes: bytes,
    *,
    filename: str = "audio.webm",
    model_name: str = DEFAULT_MODEL_NAME,
    task: str = DEFAULT_TASK,
    language: str | None = None,
    temperature: float = 0.0,
) -> dict[str, Any]:
    """Transcribe raw uploaded audio bytes by writing them to a temporary file."""
    audio_bytes = audio_bytes or b""
    if not audio_bytes:
        return TranscriptionResult(
            ok=False,
            available=whisper_available(),
            model_name=model_name,
            task=task,
            error="Empty audio payload",
        ).to_dict()
    if len(audio_bytes) > MAX_UPLOAD_BYTES:
        return TranscriptionResult(
            ok=False,
            available=whisper_available(),
            model_name=model_name,
            task=task,
            source_bytes=len(audio_bytes),
            error=f"Audio payload exceeds max size of {MAX_UPLOAD_BYTES} bytes",
        ).to_dict()

    suffix = safe_suffix(filename)
    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(prefix="radcopilot_whisper_", suffix=suffix, delete=False) as fh:
            fh.write(audio_bytes)
            tmp_path = Path(fh.name)

        result = transcribe_file(
            tmp_path,
            model_name=model_name,
            task=task,
            language=language,
            temperature=temperature,
        )
        result["source_name"] = filename or result.get("source_name", "")
        result["source_bytes"] = len(audio_bytes)
        result["source_sha256"] = sha256_bytes(audio_bytes)
        return result
    finally:
        if tmp_path is not None:
            try:
                tmp_path.unlink(missing_ok=True)
            except Exception:
                pass


def transcribe_request(config: ConfigLike, handler: Any) -> dict[str, Any]:
    """
    Parse an incoming HTTP request and transcribe its audio payload.

    Supported request shapes:
    - multipart/form-data with field names like file/audio/blob
    - application/json with `audio_b64` / `audioBase64` / `audio`
    - raw body upload with audio/* or application/octet-stream
    """
    try:
        content_type = str(handler.headers.get("Content-Type") or "").strip()
        length = int(handler.headers.get("Content-Length", "0") or "0")
        if length <= 0:
            return TranscriptionResult(
                ok=False,
                available=whisper_available(),
                error="Missing or empty request body",
            ).to_dict()
        if length > MAX_UPLOAD_BYTES:
            return TranscriptionResult(
                ok=False,
                available=whisper_available(),
                error=f"Request body exceeds max size of {MAX_UPLOAD_BYTES} bytes",
                source_bytes=length,
            ).to_dict()

        model_name = str(handler.headers.get("X-Whisper-Model") or DEFAULT_MODEL_NAME).strip() or DEFAULT_MODEL_NAME
        language = str(handler.headers.get("X-Whisper-Language") or "").strip() or None
        task = str(handler.headers.get("X-Whisper-Task") or DEFAULT_TASK).strip() or DEFAULT_TASK

        audio_bytes: bytes = b""
        filename = str(handler.headers.get("X-Filename") or "audio.webm").strip() or "audio.webm"
        temperature = 0.0

        if content_type.startswith("multipart/form-data"):
            parsed = parse_multipart(handler=handler, content_type=content_type, content_length=length)
            if not parsed["ok"]:
                return parsed
            audio_bytes = parsed.get("audio_bytes", b"")
            filename = str(parsed.get("filename") or filename)
            model_name = str(parsed.get("model_name") or model_name)
            task = str(parsed.get("task") or task)
            language = str(parsed.get("language") or language or "").strip() or None
            temperature = float(parsed.get("temperature") or 0.0)
        else:
            raw = handler.rfile.read(length)
            if content_type.startswith("application/json"):
                parsed = parse_json_audio_request(raw)
                if not parsed["ok"]:
                    return parsed
                audio_bytes = parsed.get("audio_bytes", b"")
                filename = str(parsed.get("filename") or filename)
                model_name = str(parsed.get("model_name") or model_name)
                task = str(parsed.get("task") or task)
                language = str(parsed.get("language") or language or "").strip() or None
                temperature = float(parsed.get("temperature") or 0.0)
            else:
                audio_bytes = raw

        result = transcribe_bytes(
            audio_bytes,
            filename=filename,
            model_name=model_name,
            task=task,
            language=language,
            temperature=temperature,
        )

        if result.get("ok"):
            log_event(
                "WHISPER_TRANSCRIBE",
                f"Transcribed audio bytes={result.get('source_bytes', 0)} model={result.get('model_name', model_name)}",
                config=config,
                source="whisper_service",
                route_path=getattr(handler, "path", ""),
                context={
                    "filename": result.get("source_name", ""),
                    "task": result.get("task", task),
                    "language": result.get("language", ""),
                    "segments": len(result.get("segments", [])),
                },
            )
        else:
            log_event(
                "WHISPER_TRANSCRIBE_FAILED",
                str(result.get("error") or "Whisper transcription failed"),
                config=config,
                source="whisper_service",
                level="WARNING",
                route_path=getattr(handler, "path", ""),
                context={
                    "filename": filename,
                    "bytes": len(audio_bytes),
                    "model_name": model_name,
                },
            )
        return result
    except Exception as exc:  # pragma: no cover - defensive integration boundary
        log_exception(
            exc,
            type="WHISPER_EXCEPTION",
            detail="Failed to transcribe request",
            config=config,
            source="whisper_service",
            route_path=getattr(handler, "path", ""),
        )
        return TranscriptionResult(
            ok=False,
            available=whisper_available(),
            error=f"{type(exc).__name__}: {exc}",
        ).to_dict()


def parse_json_audio_request(raw: bytes) -> dict[str, Any]:
    """Parse a JSON request body that carries a base64 audio payload."""
    try:
        payload = json.loads(raw.decode("utf-8"))
    except Exception as exc:
        return {"ok": False, "error": f"Invalid JSON body: {type(exc).__name__}: {exc}", "available": whisper_available()}

    if not isinstance(payload, dict):
        return {"ok": False, "error": "JSON body must be an object", "available": whisper_available()}

    audio_b64 = first_non_empty(
        payload.get("audio_b64"),
        payload.get("audioBase64"),
        payload.get("base64"),
        payload.get("audio"),
    )
    if not audio_b64:
        return {"ok": False, "error": "Missing audio base64 field", "available": whisper_available()}

    try:
        audio_bytes = decode_base64_audio(str(audio_b64))
    except Exception as exc:
        return {"ok": False, "error": f"Invalid base64 audio payload: {type(exc).__name__}: {exc}", "available": whisper_available()}

    return {
        "ok": True,
        "audio_bytes": audio_bytes,
        "filename": str(payload.get("filename") or payload.get("name") or "audio.webm"),
        "model_name": str(payload.get("model_name") or payload.get("model") or DEFAULT_MODEL_NAME),
        "task": str(payload.get("task") or DEFAULT_TASK),
        "language": str(payload.get("language") or "").strip(),
        "temperature": payload.get("temperature") or 0.0,
    }


def parse_multipart(*, handler: Any, content_type: str, content_length: int) -> dict[str, Any]:
    """Parse a multipart upload and extract the most likely audio file field."""
    environ = {
        "REQUEST_METHOD": "POST",
        "CONTENT_TYPE": content_type,
        "CONTENT_LENGTH": str(content_length),
    }
    headers = {"content-type": content_type}
    fs = cgi.FieldStorage(  # noqa: UP031 - stdlib multipart parser remains pragmatic here
        fp=handler.rfile,
        headers=headers,
        environ=environ,
        keep_blank_values=True,
    )

    file_field = None
    for candidate in _AUDIO_FIELD_CANDIDATES:
        if candidate in fs:
            file_field = fs[candidate]
            break
    if file_field is None:
        for key in fs.keys():
            field = fs[key]
            if getattr(field, "file", None) is not None:
                file_field = field
                break
    if file_field is None or getattr(file_field, "file", None) is None:
        return {"ok": False, "error": "Multipart request did not contain an audio file field", "available": whisper_available()}

    file_obj = file_field.file
    audio_bytes = read_binary_stream(file_obj)
    filename = str(getattr(file_field, "filename", "") or "audio.webm")

    def _field_text(name: str, default: str = "") -> str:
        if name not in fs:
            return default
        value = fs.getvalue(name)
        return str(value).strip() if value is not None else default

    return {
        "ok": True,
        "audio_bytes": audio_bytes,
        "filename": filename,
        "model_name": _field_text("model_name", _field_text("model", DEFAULT_MODEL_NAME)),
        "task": _field_text("task", DEFAULT_TASK),
        "language": _field_text("language", ""),
        "temperature": _coerce_float(_field_text("temperature", "0"), 0.0),
    }


def normalize_segments(raw_segments: Any) -> list[dict[str, Any]]:
    """Normalize Whisper segments into a JSON-friendly, compact list."""
    segments: list[dict[str, Any]] = []
    if not isinstance(raw_segments, list):
        return segments
    for item in raw_segments:
        if not isinstance(item, dict):
            continue
        text = normalize_text(str(item.get("text") or ""))
        if not text:
            continue
        seg: dict[str, Any] = {"text": text}
        for key in ("id", "start", "end", "avg_logprob", "no_speech_prob"):
            if key in item:
                seg[key] = item[key]
        segments.append(seg)
    return segments


def normalize_text(text: str) -> str:
    """Collapse whitespace and normalize transcription text for storage."""
    return " ".join(str(text or "").replace("\x00", " ").split()).strip()


def read_binary_stream(stream: BinaryIO, chunk_size: int = 1024 * 1024) -> bytes:
    """Read a binary stream safely into memory."""
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = stream.read(chunk_size)
        if not chunk:
            break
        if isinstance(chunk, str):
            chunk = chunk.encode("utf-8")
        total += len(chunk)
        if total > MAX_UPLOAD_BYTES:
            raise ValueError(f"Audio payload exceeds max size of {MAX_UPLOAD_BYTES} bytes")
        chunks.append(chunk)
    return b"".join(chunks)


def decode_base64_audio(value: str) -> bytes:
    """Decode plain base64 or data-URL-style base64 audio content."""
    value = str(value or "").strip()
    if "," in value and value.lower().startswith("data:"):
        value = value.split(",", 1)[1]
    return base64.b64decode(value, validate=False)


def first_non_empty(*values: Any) -> str:
    """Return the first non-empty string-like value."""
    for value in values:
        text = str(value).strip() if value is not None else ""
        if text:
            return text
    return ""


def safe_suffix(filename: str) -> str:
    """Return a sanitized file suffix for temporary audio files."""
    suffix = Path(filename or "audio.webm").suffix.lower()
    if not suffix or len(suffix) > 12:
        return ".webm"
    if not suffix.startswith("."):
        suffix = f".{suffix}"
    return suffix


def sha256_bytes(data: bytes) -> str:
    """Return a SHA-256 digest for raw bytes."""
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    """Return a SHA-256 digest for a file path."""
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        while True:
            chunk = fh.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def utc_now() -> str:
    """Return an ISO-8601 UTC timestamp."""
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _coerce_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return default
