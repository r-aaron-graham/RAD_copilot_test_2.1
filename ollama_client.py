from __future__ import annotations

"""
radcopilot.services.ollama_client

Reusable Ollama client for RadCopilot Local.

Design goals:
- standard-library only
- stable API for future report / guideline / differential modules
- explicit retries and timeouts
- support for health checks, model listing, chat, generate, and streaming
- safe JSON parsing and helpful exception types
- easy integration with the current modular local architecture

Typical usage:

    from radcopilot.services.ollama_client import OllamaClient

    client = OllamaClient(base_url="http://localhost:11434", model="llama3.1:8b")
    ok = client.health_check()
    models = client.list_models()
    reply = client.chat_text("Summarize these findings")
"""

from dataclasses import dataclass, field
import json
import time
from typing import Any, Iterator, Iterable, Mapping
import urllib.error
import urllib.parse
import urllib.request


DEFAULT_BASE_URL = "http://localhost:11434"
DEFAULT_TIMEOUT = 120.0
DEFAULT_CONNECT_TIMEOUT = 5.0
DEFAULT_RETRIES = 2
DEFAULT_KEEP_ALIVE = "10m"
DEFAULT_MODEL = "llama3.1:8b"
USER_AGENT = "RadCopilotLocal/0.1"


class OllamaClientError(RuntimeError):
    """Base exception for Ollama client failures."""


class OllamaHTTPError(OllamaClientError):
    """Raised when Ollama returns a non-2xx HTTP response."""

    def __init__(self, status: int, message: str, body: str = "") -> None:
        super().__init__(f"HTTP {status}: {message}")
        self.status = status
        self.message = message
        self.body = body


class OllamaDecodeError(OllamaClientError):
    """Raised when a response cannot be decoded as expected JSON/text."""


@dataclass(slots=True)
class OllamaConfig:
    """Configuration for the Ollama client."""

    base_url: str = DEFAULT_BASE_URL
    model: str = DEFAULT_MODEL
    timeout: float = DEFAULT_TIMEOUT
    connect_timeout: float = DEFAULT_CONNECT_TIMEOUT
    retries: int = DEFAULT_RETRIES
    keep_alive: str = DEFAULT_KEEP_ALIVE
    headers: dict[str, str] = field(default_factory=dict)

    def normalized_base_url(self) -> str:
        return normalize_base_url(self.base_url)


@dataclass(slots=True)
class ChatMessage:
    """Typed chat message used by `/api/chat`."""

    role: str
    content: str

    def to_dict(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


@dataclass(slots=True)
class ChatResult:
    """Normalized result from a non-streaming chat request."""

    model: str
    content: str
    raw: dict[str, Any]
    done: bool = True
    total_duration: int | None = None
    load_duration: int | None = None
    prompt_eval_count: int | None = None
    eval_count: int | None = None


@dataclass(slots=True)
class GenerateResult:
    """Normalized result from a non-streaming generate request."""

    model: str
    response: str
    raw: dict[str, Any]
    done: bool = True
    total_duration: int | None = None
    load_duration: int | None = None
    prompt_eval_count: int | None = None
    eval_count: int | None = None


class OllamaClient:
    """Reusable standard-library Ollama API client."""

    def __init__(
        self,
        *,
        base_url: str = DEFAULT_BASE_URL,
        model: str = DEFAULT_MODEL,
        timeout: float = DEFAULT_TIMEOUT,
        connect_timeout: float = DEFAULT_CONNECT_TIMEOUT,
        retries: int = DEFAULT_RETRIES,
        keep_alive: str = DEFAULT_KEEP_ALIVE,
        headers: Mapping[str, str] | None = None,
    ) -> None:
        self.config = OllamaConfig(
            base_url=base_url,
            model=model,
            timeout=timeout,
            connect_timeout=connect_timeout,
            retries=retries,
            keep_alive=keep_alive,
            headers=dict(headers or {}),
        )

    # ------------------------------------------------------------------
    # Health / metadata
    # ------------------------------------------------------------------
    @property
    def base_url(self) -> str:
        return self.config.normalized_base_url()

    @property
    def model(self) -> str:
        return self.config.model

    def health_check(self, *, timeout: float | None = None) -> bool:
        """Return True when Ollama responds successfully to /api/tags."""
        try:
            self._request_json("GET", "/api/tags", timeout=timeout or self.config.connect_timeout)
            return True
        except OllamaClientError:
            return False

    def list_models(self) -> list[dict[str, Any]]:
        """Return installed local models from /api/tags."""
        payload = self._request_json("GET", "/api/tags")
        models = payload.get("models", [])
        return models if isinstance(models, list) else []

    def show_model(self, model: str | None = None) -> dict[str, Any]:
        """Return metadata for a single model via /api/show."""
        payload = {"name": model or self.model}
        return self._request_json("POST", "/api/show", data=payload)

    def pull_model(
        self,
        model: str,
        *,
        insecure: bool = False,
        stream: bool = False,
    ) -> dict[str, Any] | Iterator[dict[str, Any]]:
        """Pull a model from Ollama registry."""
        payload = {"name": model, "insecure": bool(insecure), "stream": bool(stream)}
        if stream:
            return self._stream_json("POST", "/api/pull", data=payload)
        return self._request_json("POST", "/api/pull", data=payload)

    # ------------------------------------------------------------------
    # Chat API
    # ------------------------------------------------------------------
    def chat(
        self,
        messages: Iterable[ChatMessage | Mapping[str, Any] | dict[str, Any]],
        *,
        model: str | None = None,
        system: str | None = None,
        options: Mapping[str, Any] | None = None,
        stream: bool = False,
        keep_alive: str | None = None,
        format: str | Mapping[str, Any] | None = None,
    ) -> ChatResult | Iterator[dict[str, Any]]:
        """Call /api/chat with structured messages."""
        payload = self._build_chat_payload(
            messages=messages,
            model=model,
            system=system,
            options=options,
            stream=stream,
            keep_alive=keep_alive,
            format=format,
        )
        if stream:
            return self._stream_json("POST", "/api/chat", data=payload)

        raw = self._request_json("POST", "/api/chat", data=payload)
        content = extract_chat_content(raw)
        return ChatResult(
            model=str(raw.get("model") or payload["model"]),
            content=content,
            raw=raw,
            done=bool(raw.get("done", True)),
            total_duration=_maybe_int(raw.get("total_duration")),
            load_duration=_maybe_int(raw.get("load_duration")),
            prompt_eval_count=_maybe_int(raw.get("prompt_eval_count")),
            eval_count=_maybe_int(raw.get("eval_count")),
        )

    def chat_text(
        self,
        prompt: str,
        *,
        model: str | None = None,
        system: str | None = None,
        options: Mapping[str, Any] | None = None,
        keep_alive: str | None = None,
        format: str | Mapping[str, Any] | None = None,
    ) -> str:
        """Convenience wrapper for single-turn user prompt -> assistant text."""
        result = self.chat(
            [ChatMessage(role="user", content=str(prompt))],
            model=model,
            system=system,
            options=options,
            stream=False,
            keep_alive=keep_alive,
            format=format,
        )
        assert isinstance(result, ChatResult)
        return result.content

    def stream_chat_text(
        self,
        prompt: str,
        *,
        model: str | None = None,
        system: str | None = None,
        options: Mapping[str, Any] | None = None,
        keep_alive: str | None = None,
        format: str | Mapping[str, Any] | None = None,
    ) -> Iterator[str]:
        """Yield incremental content strings from a streaming chat call."""
        stream = self.chat(
            [ChatMessage(role="user", content=str(prompt))],
            model=model,
            system=system,
            options=options,
            stream=True,
            keep_alive=keep_alive,
            format=format,
        )
        assert not isinstance(stream, ChatResult)
        for chunk in stream:
            content = extract_chat_content(chunk)
            if content:
                yield content

    # ------------------------------------------------------------------
    # Generate API
    # ------------------------------------------------------------------
    def generate(
        self,
        prompt: str,
        *,
        model: str | None = None,
        system: str | None = None,
        template: str | None = None,
        context: list[int] | None = None,
        options: Mapping[str, Any] | None = None,
        stream: bool = False,
        keep_alive: str | None = None,
        suffix: str | None = None,
        raw_mode: bool = False,
        format: str | Mapping[str, Any] | None = None,
    ) -> GenerateResult | Iterator[dict[str, Any]]:
        """Call /api/generate for single-prompt generation."""
        payload: dict[str, Any] = {
            "model": model or self.model,
            "prompt": str(prompt),
            "stream": bool(stream),
            "keep_alive": keep_alive or self.config.keep_alive,
        }
        if system:
            payload["system"] = str(system)
        if template:
            payload["template"] = str(template)
        if context is not None:
            payload["context"] = list(context)
        if options:
            payload["options"] = dict(options)
        if suffix:
            payload["suffix"] = str(suffix)
        if raw_mode:
            payload["raw"] = True
        if format is not None:
            payload["format"] = format

        if stream:
            return self._stream_json("POST", "/api/generate", data=payload)

        raw = self._request_json("POST", "/api/generate", data=payload)
        response = str(raw.get("response") or "")
        return GenerateResult(
            model=str(raw.get("model") or payload["model"]),
            response=response,
            raw=raw,
            done=bool(raw.get("done", True)),
            total_duration=_maybe_int(raw.get("total_duration")),
            load_duration=_maybe_int(raw.get("load_duration")),
            prompt_eval_count=_maybe_int(raw.get("prompt_eval_count")),
            eval_count=_maybe_int(raw.get("eval_count")),
        )

    def generate_text(
        self,
        prompt: str,
        *,
        model: str | None = None,
        system: str | None = None,
        template: str | None = None,
        context: list[int] | None = None,
        options: Mapping[str, Any] | None = None,
        keep_alive: str | None = None,
        suffix: str | None = None,
        raw_mode: bool = False,
        format: str | Mapping[str, Any] | None = None,
    ) -> str:
        """Convenience wrapper for generate -> response text."""
        result = self.generate(
            prompt,
            model=model,
            system=system,
            template=template,
            context=context,
            options=options,
            stream=False,
            keep_alive=keep_alive,
            suffix=suffix,
            raw_mode=raw_mode,
            format=format,
        )
        assert isinstance(result, GenerateResult)
        return result.response

    def embeddings(
        self,
        text: str | list[str],
        *,
        model: str | None = None,
        truncate: bool | None = None,
        options: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Return embeddings from /api/embed using the supplied model."""
        payload: dict[str, Any] = {
            "model": model or self.model,
            "input": text,
        }
        if truncate is not None:
            payload["truncate"] = bool(truncate)
        if options:
            payload["options"] = dict(options)
        return self._request_json("POST", "/api/embed", data=payload)

    # ------------------------------------------------------------------
    # Internal request plumbing
    # ------------------------------------------------------------------
    def _build_chat_payload(
        self,
        *,
        messages: Iterable[ChatMessage | Mapping[str, Any] | dict[str, Any]],
        model: str | None,
        system: str | None,
        options: Mapping[str, Any] | None,
        stream: bool,
        keep_alive: str | None,
        format: str | Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        normalized_messages: list[dict[str, Any]] = []
        if system:
            normalized_messages.append({"role": "system", "content": str(system)})
        for item in messages:
            if isinstance(item, ChatMessage):
                normalized_messages.append(item.to_dict())
            elif isinstance(item, Mapping):
                role = str(item.get("role") or "user")
                content = str(item.get("content") or "")
                message: dict[str, Any] = {"role": role, "content": content}
                if "images" in item:
                    message["images"] = item["images"]
                if "tool_calls" in item:
                    message["tool_calls"] = item["tool_calls"]
                normalized_messages.append(message)
            else:
                raise TypeError(f"Unsupported message type: {type(item)!r}")

        return {
            "model": model or self.model,
            "messages": normalized_messages,
            "stream": bool(stream),
            "keep_alive": keep_alive or self.config.keep_alive,
            **({"options": dict(options)} if options else {}),
            **({"format": format} if format is not None else {}),
        }

    def _request_json(
        self,
        method: str,
        path: str,
        *,
        data: Mapping[str, Any] | None = None,
        timeout: float | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> dict[str, Any]:
        """Issue a JSON request with retry handling and return parsed JSON."""
        payload = None if data is None else json.dumps(data).encode("utf-8")
        final_headers = {
            "Accept": "application/json",
            "User-Agent": USER_AGENT,
            **self.config.headers,
            **dict(headers or {}),
        }
        if payload is not None:
            final_headers.setdefault("Content-Type", "application/json")

        url = build_url(self.base_url, path)
        attempts = max(1, int(self.config.retries) + 1)
        request_timeout = float(timeout or self.config.timeout)
        last_exc: Exception | None = None

        for attempt in range(1, attempts + 1):
            request = urllib.request.Request(url, data=payload, method=method.upper(), headers=final_headers)
            try:
                with urllib.request.urlopen(request, timeout=request_timeout) as response:
                    raw_body = response.read().decode("utf-8", errors="replace")
                    if not raw_body.strip():
                        return {}
                    try:
                        parsed = json.loads(raw_body)
                    except json.JSONDecodeError as exc:
                        raise OllamaDecodeError(f"Invalid JSON from {path}: {exc}") from exc
                    if isinstance(parsed, dict):
                        return parsed
                    raise OllamaDecodeError(f"Expected JSON object from {path}, got {type(parsed).__name__}")
            except urllib.error.HTTPError as exc:
                body = exc.read().decode("utf-8", errors="replace") if exc.fp else ""
                raise OllamaHTTPError(exc.code, exc.reason or "HTTP error", body) from exc
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                last_exc = exc
                if attempt >= attempts:
                    break
                time.sleep(_retry_delay(attempt))

        raise OllamaClientError(f"Request failed for {path}: {type(last_exc).__name__}: {last_exc}")

    def _stream_json(
        self,
        method: str,
        path: str,
        *,
        data: Mapping[str, Any] | None = None,
        timeout: float | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> Iterator[dict[str, Any]]:
        """Yield JSON objects from line-delimited Ollama streaming endpoints."""
        payload = None if data is None else json.dumps(data).encode("utf-8")
        final_headers = {
            "Accept": "application/json",
            "User-Agent": USER_AGENT,
            **self.config.headers,
            **dict(headers or {}),
        }
        if payload is not None:
            final_headers.setdefault("Content-Type", "application/json")

        url = build_url(self.base_url, path)
        request = urllib.request.Request(url, data=payload, method=method.upper(), headers=final_headers)

        try:
            with urllib.request.urlopen(request, timeout=float(timeout or self.config.timeout)) as response:
                for raw_line in response:
                    line = raw_line.decode("utf-8", errors="replace").strip()
                    if not line:
                        continue
                    try:
                        item = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise OllamaDecodeError(f"Invalid streaming JSON from {path}: {exc}") from exc
                    if isinstance(item, dict):
                        yield item
                    else:
                        raise OllamaDecodeError(
                            f"Expected JSON object in stream from {path}, got {type(item).__name__}"
                        )
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace") if exc.fp else ""
            raise OllamaHTTPError(exc.code, exc.reason or "HTTP error", body) from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise OllamaClientError(f"Streaming request failed for {path}: {type(exc).__name__}: {exc}") from exc


# ----------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------

def normalize_base_url(base_url: str) -> str:
    value = str(base_url or DEFAULT_BASE_URL).strip()
    if not value:
        value = DEFAULT_BASE_URL
    if not value.startswith(("http://", "https://")):
        value = f"http://{value}"
    return value.rstrip("/")



def build_url(base_url: str, path: str) -> str:
    base = normalize_base_url(base_url)
    if not path.startswith("/"):
        path = "/" + path
    return urllib.parse.urljoin(base + "/", path.lstrip("/"))



def extract_chat_content(payload: Mapping[str, Any] | None) -> str:
    """Extract assistant text from Ollama /api/chat responses or chunks."""
    if not payload:
        return ""
    message = payload.get("message")
    if isinstance(message, Mapping):
        content = message.get("content")
        if content is not None:
            return str(content)
    response = payload.get("response")
    if response is not None:
        return str(response)
    return ""



def coerce_messages(
    prompt: str,
    *,
    system: str | None = None,
    role: str = "user",
) -> list[ChatMessage]:
    """Create a standard message list from a single prompt."""
    result: list[ChatMessage] = []
    if system:
        result.append(ChatMessage(role="system", content=str(system)))
    result.append(ChatMessage(role=role, content=str(prompt)))
    return result



def summarize_models(models: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Return a compact model summary list for UI or diagnostics."""
    items: list[dict[str, Any]] = []
    for model in models:
        items.append(
            {
                "name": str(model.get("name") or ""),
                "size": _maybe_int(model.get("size")),
                "modified_at": model.get("modified_at"),
                "digest": model.get("digest"),
            }
        )
    return items



def _retry_delay(attempt: int) -> float:
    # mild exponential backoff with a low ceiling for local usage
    return min(0.35 * (2 ** max(0, attempt - 1)), 2.0)



def _maybe_int(value: Any) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except Exception:
        return None


__all__ = [
    "DEFAULT_BASE_URL",
    "DEFAULT_MODEL",
    "DEFAULT_TIMEOUT",
    "ChatMessage",
    "ChatResult",
    "GenerateResult",
    "OllamaClient",
    "OllamaClientError",
    "OllamaConfig",
    "OllamaDecodeError",
    "OllamaHTTPError",
    "build_url",
    "coerce_messages",
    "extract_chat_content",
    "normalize_base_url",
    "summarize_models",
]
