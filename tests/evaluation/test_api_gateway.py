from __future__ import annotations

import json
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from eval.api_gateway import (
    GatewayConfig,
    GatewayState,
    SlidingWindowLimiter,
    estimate_request_tokens,
    load_credentials,
)
from scripts.run_multigpu_plan import _external_worker


class _UpstreamHandler(BaseHTTPRequestHandler):
    calls: list[dict] = []

    def log_message(self, format: str, *args: object) -> None:
        return

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length", "0"))
        payload = json.loads(self.rfile.read(length))
        type(self).calls.append(
            {
                "path": self.path,
                "payload": payload,
                "authorization": self.headers.get("Authorization"),
            }
        )
        if len(type(self).calls) == 1:
            body = json.dumps(
                {
                    "error": {
                        "type": "rate_limit_error",
                        "details": {"retry_after": 0.01},
                    }
                }
            ).encode()
            self.send_response(429)
        else:
            body = json.dumps(
                {
                    "choices": [
                        {"message": {"content": '{"choice_id":"A"}'}}
                    ],
                    "usage": {
                        "prompt_tokens": 10,
                        "completion_tokens": 4,
                        "total_tokens": 14,
                    },
                }
            ).encode()
            self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class ApiGatewayTests(unittest.TestCase):
    def test_load_credentials_without_shell_evaluation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "secret.env"
            path.write_text(
                "OPENAI_API_KEY=test-key\n"
                "HABITBENCH_EXTERNAL_API_BASE_URL=https://example.test/v1\n",
                encoding="utf-8",
            )
            key, base_url = load_credentials(path)
        self.assertEqual(key, "test-key")
        self.assertEqual(base_url, "https://example.test/v1")

    def test_token_estimate_reserves_requested_completion(self) -> None:
        small = estimate_request_tokens({"messages": [{"content": "hello"}]})
        large = estimate_request_tokens(
            {"messages": [{"content": "hello"}], "max_tokens": 512}
        )
        self.assertGreaterEqual(large - small, 512)

    def test_gateway_retries_and_injects_no_thinking(self) -> None:
        _UpstreamHandler.calls = []
        server = ThreadingHTTPServer(("127.0.0.1", 0), _UpstreamHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        state = GatewayState(
            GatewayConfig(
                upstream_base_url=f"http://127.0.0.1:{server.server_port}/v1",
                api_key="private-test-key",
                max_upstream_retries=2,
                metrics_path=None,
                metrics_every=25,
            ),
            SlidingWindowLimiter(1000, 1_000_000, window_sec=0.1),
        )
        try:
            response, attempts, model, _ = state.forward(
                "POST",
                "/v1/chat/completions",
                json.dumps(
                    {
                        "model": "test-model",
                        "messages": [{"role": "user", "content": "hello"}],
                        "max_tokens": 8,
                    }
                ).encode(),
                "application/json",
            )
        finally:
            state.close()
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(attempts, 2)
        self.assertEqual(model, "test-model")
        self.assertEqual(len(_UpstreamHandler.calls), 2)
        call = _UpstreamHandler.calls[-1]
        self.assertEqual(call["path"], "/v1/chat/completions")
        self.assertEqual(call["authorization"], "Bearer private-test-key")
        self.assertEqual(
            call["payload"]["chat_template_kwargs"],
            {"enable_thinking": False},
        )

    def test_external_worker_has_no_local_server_process(self) -> None:
        worker = _external_worker(2, "7", "http://127.0.0.1:8090/v1/")
        self.assertEqual(worker["worker_index"], 2)
        self.assertEqual(worker["gpu"], "7")
        self.assertIsNone(worker["port"])
        self.assertEqual(worker["base_url"], "http://127.0.0.1:8090/v1")
        self.assertEqual(
            worker["throughput_gate"]["status"],
            "not_applicable_external_api",
        )


if __name__ == "__main__":
    unittest.main()
