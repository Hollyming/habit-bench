"""Rate-limited OpenAI-compatible gateway for external HABIT-Bench runs.

The gateway is intentionally model agnostic: callers retain the exact model
name stored in each experiment plan.  It owns the upstream credential, applies
one shared RPM/TPM budget across all local workers, disables provider thinking
through the PJLab-compatible chat-template extension, and absorbs transient
429/5xx responses without exposing prompts or credentials in its logs.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import signal
import threading
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import httpx


RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _log(event: str, **fields: Any) -> None:
    print(
        json.dumps(
            {"time": _utc_now(), "event": event, **fields},
            ensure_ascii=False,
            sort_keys=True,
        ),
        flush=True,
    )


def load_credentials(path: Path) -> tuple[str, str]:
    """Read a shell-style KEY=VALUE file without evaluating shell code."""

    values: dict[str, str] = {}
    for line_number, raw_line in enumerate(
        path.expanduser().read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise ValueError(f"Invalid credential line {line_number} in {path}")
        name, value = line.split("=", 1)
        name = name.strip()
        if not name or not value:
            raise ValueError(f"Empty credential field on line {line_number} in {path}")
        values[name] = value
    api_key = values.get("OPENAI_API_KEY", "")
    base_url = values.get("HABITBENCH_EXTERNAL_API_BASE_URL", "")
    if not api_key or not base_url:
        raise ValueError(
            "Credential file must define OPENAI_API_KEY and "
            "HABITBENCH_EXTERNAL_API_BASE_URL"
        )
    parsed = urlsplit(base_url)
    if parsed.scheme != "https" or not parsed.netloc:
        raise ValueError("External API base URL must be an absolute HTTPS URL")
    return api_key, base_url.rstrip("/")


def estimate_request_tokens(payload: dict[str, Any]) -> int:
    """Return a deliberately conservative TPM reservation.

    One serialized Unicode character is charged as one token, then the full
    requested completion budget is added.  This is substantially more
    conservative than the observed tokenizers while remaining far below the
    configured 50M TPM allowance at 60 RPM.
    """

    serialized = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    completion = payload.get("max_completion_tokens", payload.get("max_tokens", 0))
    try:
        completion_tokens = max(0, int(completion or 0))
    except (TypeError, ValueError):
        completion_tokens = 0
    return max(1, len(serialized)) + completion_tokens


class SlidingWindowLimiter:
    """Thread-safe rolling one-minute request/token limiter with cooldown."""

    def __init__(self, rpm: int, tpm: int, *, window_sec: float = 60.0) -> None:
        if rpm <= 0 or tpm <= 0 or window_sec <= 0:
            raise ValueError("rpm, tpm and window_sec must be positive")
        self.rpm = rpm
        self.tpm = tpm
        self.window_sec = window_sec
        self._requests: deque[float] = deque()
        self._tokens: deque[tuple[float, int]] = deque()
        self._token_total = 0
        self._not_before = 0.0
        self._condition = threading.Condition()

    def _prune(self, now: float) -> None:
        cutoff = now - self.window_sec
        while self._requests and self._requests[0] <= cutoff:
            self._requests.popleft()
        while self._tokens and self._tokens[0][0] <= cutoff:
            _, tokens = self._tokens.popleft()
            self._token_total -= tokens

    def defer(self, seconds: float) -> None:
        if seconds <= 0:
            return
        with self._condition:
            self._not_before = max(self._not_before, time.monotonic() + seconds)
            self._condition.notify_all()

    def acquire(self, estimated_tokens: int) -> float:
        if estimated_tokens <= 0:
            raise ValueError("estimated_tokens must be positive")
        if estimated_tokens > self.tpm:
            raise ValueError(
                f"One request reserves {estimated_tokens} tokens, exceeding TPM={self.tpm}"
            )
        started = time.monotonic()
        with self._condition:
            while True:
                now = time.monotonic()
                self._prune(now)
                waits: list[float] = []
                if now < self._not_before:
                    waits.append(self._not_before - now)
                if len(self._requests) >= self.rpm:
                    waits.append(self._requests[0] + self.window_sec - now)
                if self._token_total + estimated_tokens > self.tpm and self._tokens:
                    running = self._token_total
                    for timestamp, tokens in self._tokens:
                        running -= tokens
                        if running + estimated_tokens <= self.tpm:
                            waits.append(timestamp + self.window_sec - now)
                            break
                if not waits:
                    self._requests.append(now)
                    self._tokens.append((now, estimated_tokens))
                    self._token_total += estimated_tokens
                    return time.monotonic() - started
                self._condition.wait(timeout=max(0.01, min(waits)))

    def snapshot(self) -> dict[str, int | float]:
        with self._condition:
            now = time.monotonic()
            self._prune(now)
            return {
                "rpm_limit": self.rpm,
                "tpm_limit": self.tpm,
                "window_sec": self.window_sec,
                "requests_in_window": len(self._requests),
                "reserved_tokens_in_window": self._token_total,
                "cooldown_remaining_sec": round(max(0.0, self._not_before - now), 3),
            }


def _retry_after(response: httpx.Response, attempt: int) -> float:
    header = response.headers.get("retry-after")
    if header:
        try:
            return max(0.1, float(header))
        except ValueError:
            pass
    try:
        payload = response.json()
    except (ValueError, json.JSONDecodeError):
        payload = None
    if isinstance(payload, dict):
        error = payload.get("error")
        candidates = [payload]
        if isinstance(error, dict):
            candidates.insert(0, error)
        for candidate in candidates:
            details = candidate.get("details")
            if isinstance(details, dict) and details.get("retry_after") is not None:
                try:
                    return max(0.1, float(details["retry_after"]))
                except (TypeError, ValueError):
                    pass
    return min(60.0, 2.0 ** min(attempt, 5))


@dataclass(frozen=True)
class GatewayConfig:
    upstream_base_url: str
    api_key: str
    max_upstream_retries: int
    metrics_path: Path | None
    metrics_every: int


class GatewayState:
    def __init__(self, config: GatewayConfig, limiter: SlidingWindowLimiter) -> None:
        self.config = config
        self.limiter = limiter
        self.client = httpx.Client(
            timeout=httpx.Timeout(connect=30.0, read=1800.0, write=120.0, pool=120.0),
            limits=httpx.Limits(max_connections=128, max_keepalive_connections=64),
            follow_redirects=False,
            # Cluster login shells commonly export HTTP(S)_PROXY. The token
            # endpoint is reached directly from the RJob, and inheriting a
            # proxy can also misroute localhost gateway tests/health traffic.
            trust_env=False,
        )
        self._lock = threading.Lock()
        self._requests = 0
        self._upstream_attempts = 0
        self._retries = 0
        self._status_counts: dict[str, int] = {}
        self._model_counts: dict[str, int] = {}
        self._usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        self.started_at = _utc_now()

    def close(self) -> None:
        self.client.close()
        self.write_metrics(force=True)

    def record(
        self,
        *,
        model: str | None,
        status_code: int,
        upstream_attempts: int,
        usage: dict[str, Any] | None,
    ) -> int:
        with self._lock:
            self._requests += 1
            request_number = self._requests
            self._upstream_attempts += upstream_attempts
            self._retries += max(0, upstream_attempts - 1)
            status = str(status_code)
            self._status_counts[status] = self._status_counts.get(status, 0) + 1
            if model:
                self._model_counts[model] = self._model_counts.get(model, 0) + 1
            if isinstance(usage, dict):
                for name in self._usage:
                    try:
                        self._usage[name] += int(usage.get(name) or 0)
                    except (TypeError, ValueError):
                        pass
        if request_number % self.config.metrics_every == 0:
            self.write_metrics()
        return request_number

    def metrics(self) -> dict[str, Any]:
        with self._lock:
            values = {
                "contract_version": "habitbench.external_api_gateway.v1",
                "started_at": self.started_at,
                "observed_at": _utc_now(),
                "requests": self._requests,
                "upstream_attempts": self._upstream_attempts,
                "retries": self._retries,
                "status_counts": dict(sorted(self._status_counts.items())),
                "model_counts": dict(sorted(self._model_counts.items())),
                "usage": dict(self._usage),
            }
        values["limiter"] = self.limiter.snapshot()
        values["upstream_origin"] = urlsplit(self.config.upstream_base_url).netloc
        return values

    def write_metrics(self, *, force: bool = False) -> None:
        path = self.config.metrics_path
        if path is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = self.metrics()
        temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
        if force:
            _log("gateway_metrics_flushed", path=str(path), requests=payload["requests"])

    def upstream_url(self, incoming_path: str) -> str:
        path = incoming_path.split("?", 1)[0]
        if path == "/v1":
            suffix = ""
        elif path.startswith("/v1/"):
            suffix = path[3:]
        else:
            suffix = path
        query = ""
        if "?" in incoming_path:
            query = "?" + incoming_path.split("?", 1)[1]
        return self.config.upstream_base_url + suffix + query

    def forward(
        self,
        method: str,
        path: str,
        body: bytes,
        content_type: str,
    ) -> tuple[httpx.Response, int, str | None, int]:
        payload: dict[str, Any] | None = None
        model: str | None = None
        if body and "json" in content_type.lower():
            parsed = json.loads(body.decode("utf-8"))
            if not isinstance(parsed, dict):
                raise ValueError("OpenAI-compatible request body must be a JSON object")
            payload = parsed
            model_value = payload.get("model")
            model = str(model_value) if model_value is not None else None
            if path.split("?", 1)[0].endswith("/chat/completions"):
                if payload.get("stream") is True:
                    raise ValueError("Streaming is disabled for reproducible HABIT evaluation")
                chat_kwargs = payload.get("chat_template_kwargs")
                if not isinstance(chat_kwargs, dict):
                    chat_kwargs = {}
                chat_kwargs["enable_thinking"] = False
                payload["chat_template_kwargs"] = chat_kwargs
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        estimate = estimate_request_tokens(payload) if payload is not None else 1
        response: httpx.Response | None = None
        attempts = 0
        total_wait = 0.0
        for attempt in range(self.config.max_upstream_retries + 1):
            attempts += 1
            waited = self.limiter.acquire(estimate)
            total_wait += waited
            try:
                response = self.client.request(
                    method,
                    self.upstream_url(path),
                    content=body or None,
                    headers={
                        "Authorization": f"Bearer {self.config.api_key}",
                        "Content-Type": content_type or "application/json",
                        "Accept": "application/json",
                    },
                )
            except httpx.TransportError as exc:
                if attempt >= self.config.max_upstream_retries:
                    raise RuntimeError(
                        "External API transport failed after "
                        f"{attempts} attempts ({type(exc).__name__})"
                    ) from exc
                delay = min(60.0, 2.0 ** min(attempt, 5)) + random.uniform(
                    0.05, 0.35
                )
                self.limiter.defer(delay)
                _log(
                    "upstream_transport_retry",
                    model=model,
                    error_type=type(exc).__name__,
                    attempt=attempt + 1,
                    retry_after_sec=round(delay, 3),
                )
                continue
            if response.status_code not in RETRYABLE_STATUS_CODES:
                break
            if attempt >= self.config.max_upstream_retries:
                break
            delay = _retry_after(response, attempt) + random.uniform(0.05, 0.35)
            self.limiter.defer(delay)
            _log(
                "upstream_retry",
                model=model,
                status=response.status_code,
                attempt=attempt + 1,
                retry_after_sec=round(delay, 3),
                request_id=response.headers.get("x-request-id"),
            )
        assert response is not None
        return response, attempts, model, math.ceil(total_wait * 1000)


class GatewayHandler(BaseHTTPRequestHandler):
    server: "GatewayServer"
    protocol_version = "HTTP/1.1"

    def log_message(self, format: str, *args: Any) -> None:
        return

    def _write(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(body)

    def _handle(self) -> None:
        if self.path.split("?", 1)[0] == "/healthz":
            body = json.dumps(
                {"status": "ok", **self.server.state.metrics()},
                ensure_ascii=False,
            ).encode("utf-8")
            self._write(200, body, "application/json")
            return
        try:
            content_length = int(self.headers.get("Content-Length", "0"))
            if content_length < 0 or content_length > 64 * 1024 * 1024:
                raise ValueError("Invalid or oversized request body")
            body = self.rfile.read(content_length) if content_length else b""
            started = time.monotonic()
            response, attempts, model, waited_ms = self.server.state.forward(
                self.command,
                self.path,
                body,
                self.headers.get("Content-Type", "application/json"),
            )
            response_body = response.content
            content_type = response.headers.get("content-type", "application/json")
            usage = None
            try:
                response_payload = response.json()
                if isinstance(response_payload, dict):
                    usage = response_payload.get("usage")
            except (ValueError, json.JSONDecodeError):
                pass
            request_number = self.server.state.record(
                model=model,
                status_code=response.status_code,
                upstream_attempts=attempts,
                usage=usage if isinstance(usage, dict) else None,
            )
            _log(
                "request_complete",
                request=request_number,
                model=model,
                status=response.status_code,
                attempts=attempts,
                limiter_wait_ms=waited_ms,
                elapsed_sec=round(time.monotonic() - started, 3),
                request_id=response.headers.get("x-request-id"),
            )
            self._write(response.status_code, response_body, content_type)
        except (BrokenPipeError, ConnectionResetError):
            _log("client_disconnected", path=self.path)
        except Exception as exc:
            _log("gateway_error", error_type=type(exc).__name__, error=str(exc), path=self.path)
            payload = json.dumps(
                {
                    "error": {
                        "type": "habitbench_gateway_error",
                        "message": str(exc),
                    }
                }
            ).encode("utf-8")
            try:
                self._write(502, payload, "application/json")
            except (BrokenPipeError, ConnectionResetError):
                pass

    def do_GET(self) -> None:  # noqa: N802
        self._handle()

    def do_POST(self) -> None:  # noqa: N802
        self._handle()


class GatewayServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address: tuple[str, int], state: GatewayState):
        super().__init__(address, GatewayHandler)
        self.state = state


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--credential-file", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8090)
    parser.add_argument("--rpm", type=int, default=60)
    parser.add_argument("--tpm", type=int, default=50_000_000)
    parser.add_argument("--max-upstream-retries", type=int, default=120)
    parser.add_argument("--metrics-path", type=Path)
    parser.add_argument("--metrics-every", type=int, default=25)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0 < args.port < 65536:
        raise ValueError("port must be between 1 and 65535")
    if args.max_upstream_retries < 0 or args.metrics_every <= 0:
        raise ValueError("retry and metrics settings are invalid")
    api_key, upstream_base_url = load_credentials(args.credential_file)
    state = GatewayState(
        GatewayConfig(
            upstream_base_url=upstream_base_url,
            api_key=api_key,
            max_upstream_retries=args.max_upstream_retries,
            metrics_path=args.metrics_path,
            metrics_every=args.metrics_every,
        ),
        SlidingWindowLimiter(args.rpm, args.tpm),
    )
    server = GatewayServer((args.host, args.port), state)

    def stop_server(signum: int, frame: Any) -> None:
        _log("gateway_signal", signal=signum)
        threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGTERM, stop_server)
    signal.signal(signal.SIGINT, stop_server)
    _log(
        "gateway_ready",
        host=args.host,
        port=args.port,
        rpm=args.rpm,
        tpm=args.tpm,
        upstream_origin=urlsplit(upstream_base_url).netloc,
    )
    try:
        server.serve_forever(poll_interval=0.5)
    finally:
        server.server_close()
        state.close()
        _log("gateway_stopped")


if __name__ == "__main__":
    main()
