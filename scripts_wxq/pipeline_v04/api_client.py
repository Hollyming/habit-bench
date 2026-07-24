#!/usr/bin/env python3
"""Minimal OpenAI-compatible JSON client shared by the v0.4 pipeline."""

from __future__ import annotations

import json
import os
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Iterable, Sequence


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    """Atomically write JSON Lines without exposing a partially written file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    os.replace(temporary, path)


def write_json(path: Path, row: dict[str, Any]) -> None:
    """Atomically write one formatted JSON object."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(row, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def post_chat(
    *,
    base_url: str,
    api_key: str,
    model: str,
    messages: Sequence[dict[str, str]],
    max_tokens: int,
    timeout: int,
    retries: int,
    transport: str = "curl",
    reasoning_effort: str | None = None,
    temperature: float = 0.7,
) -> dict[str, Any]:
    """Request a strict JSON response from an OpenAI-compatible chat endpoint."""
    payload: dict[str, Any] = {
        "model": model,
        "messages": list(messages),
        "temperature": temperature,
        "max_tokens": max_tokens,
        "response_format": {"type": "json_object"},
    }
    reasoning_effort = reasoning_effort or os.getenv("HABITBENCH_REASONING_EFFORT")
    if reasoning_effort:
        payload["reasoning_effort"] = reasoning_effort
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")

    if transport in {"curl", "curl_stream"}:
        url = base_url.rstrip("/") + "/chat/completions"
        last_error = None
        for attempt in range(retries + 1):
            proc = subprocess.run(
                [
                    "curl", "-sS", "-N", "--http1.1",
                    "--connect-timeout", str(min(timeout, 30)),
                    "--max-time", str(timeout),
                    "-H", f"Authorization: Bearer {api_key}",
                    "-H", "Content-Type: application/json",
                    "--data-binary", "@-", url,
                ],
                input=data.decode("utf-8"),
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            if proc.returncode == 0:
                try:
                    response = json.loads(proc.stdout)
                    if "error" not in response:
                        content = response["choices"][0]["message"]["content"]
                        return {
                            "json": json.loads(content),
                            "raw": content,
                            "usage": response.get("usage", {}),
                        }
                    last_error = json.dumps(response["error"], ensure_ascii=False)[:800]
                except Exception as exc:
                    last_error = f"bad_json_response:{type(exc).__name__}:{proc.stdout[:800]}"
            else:
                last_error = f"curl_exit_{proc.returncode}:{proc.stderr[:800]}:{proc.stdout[:300]}"
            if attempt < retries:
                time.sleep(min(2 * (attempt + 1), 12))
        raise RuntimeError(last_error or "unknown_curl_error")

    last_error = None
    for attempt in range(retries + 1):
        try:
            request = urllib.request.Request(
                base_url.rstrip("/") + "/chat/completions",
                data=data,
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(request, timeout=timeout) as response:
                body = json.loads(response.read().decode("utf-8"))
            content = body["choices"][0]["message"]["content"]
            return {"json": json.loads(content), "raw": content, "usage": body.get("usage", {})}
        except (urllib.error.URLError, TimeoutError, KeyError, json.JSONDecodeError) as exc:
            last_error = repr(exc)
            if attempt < retries:
                time.sleep(min(2**attempt, 10))
    raise RuntimeError(f"chat_request_failed:{last_error}")
