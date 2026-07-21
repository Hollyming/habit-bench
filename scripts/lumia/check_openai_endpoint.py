#!/usr/bin/env python
"""Check a local OpenAI-compatible endpoint before Lumia full runs."""

from __future__ import annotations

import argparse
import json
import os
import urllib.error
import urllib.request
from typing import Any, Dict


def request_json(url: str, payload: Dict[str, Any] | None, api_key: str, timeout: int) -> Dict[str, Any]:
    headers = {"Authorization": f"Bearer {api_key}"}
    data = None
    if payload is not None:
        headers["Content-Type"] = "application/json"
        data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method="POST" if payload else "GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} from {url}: {body[:1000]}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Could not reach {url}: {exc}") from exc
    return json.loads(body)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default=os.getenv("OPENAI_BASE_URL", "http://127.0.0.1:8000/v1"))
    parser.add_argument("--api-key", default=os.getenv("OPENAI_API_KEY", "dummy"))
    parser.add_argument("--model", default=os.getenv("HABITBENCH_SERVED_MODEL", "habitbench-qwen3-8b"))
    parser.add_argument(
        "--structured-output-mode",
        choices=["json_schema", "json_object"],
        default=os.getenv("HABITBENCH_STRUCTURED_OUTPUT_MODE", "json_schema"),
    )
    parser.add_argument("--timeout-sec", type=int, default=60)
    parser.add_argument("--skip-chat", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base = args.base_url.rstrip("/")
    checks = []
    models = request_json(f"{base}/models", None, args.api_key, args.timeout_sec)
    model_ids = [row.get("id") for row in models.get("data", []) if isinstance(row, dict)]
    if args.model not in model_ids:
        checks.append(
            {
                "status": "warning",
                "check": "models",
                "expected_model": args.model,
                "available_models": model_ids,
            }
        )
    else:
        checks.append({"status": "pass", "check": "models", "model": args.model})

    if args.skip_chat:
        status = "warning" if any(check["status"] == "warning" for check in checks) else "pass"
        print(json.dumps({"status": status, "checks": checks}, indent=2, sort_keys=True))
        return

    if args.structured_output_mode == "json_schema":
        response_format: Dict[str, Any] = {
            "type": "json_schema",
            "json_schema": {
                "name": "habitbench_endpoint_check",
                "schema": {
                    "type": "object",
                    "properties": {
                        "ok": {"type": "boolean"},
                        "message": {"type": "string"},
                    },
                    "required": ["ok", "message"],
                },
            },
        }
    else:
        response_format = {"type": "json_object"}

    completion = request_json(
        f"{base}/chat/completions",
        {
            "model": args.model,
            "messages": [
                {"role": "system", "content": "Return only valid JSON."},
                {"role": "user", "content": 'Return {"ok": true, "message": "habitbench-ready"}.'},
            ],
            "temperature": 0,
            "max_tokens": 64,
            "response_format": response_format,
        },
        args.api_key,
        args.timeout_sec,
    )
    content = completion["choices"][0]["message"].get("content", "")
    parsed = json.loads(content)
    if parsed.get("ok") is not True:
        raise RuntimeError(f"Endpoint returned JSON but ok was not true: {parsed}")
    checks.append(
        {
            "status": "pass",
            "check": "chat_completions",
            "structured_output_mode": args.structured_output_mode,
            "parsed": parsed,
        }
    )
    status = "warning" if any(check["status"] == "warning" for check in checks) else "pass"
    print(json.dumps({"status": status, "checks": checks}, indent=2, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(json.dumps({"status": "fail", "error": str(exc)}, indent=2, sort_keys=True))
        raise SystemExit(1) from exc
