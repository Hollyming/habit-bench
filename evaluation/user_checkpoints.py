"""Persist completed, independent memory users without restoring partial DB writes."""
from pathlib import Path
import json

from evaluation.checkpoints import atomic_json, digest


class UserResultCheckpoints:
    def __init__(self, root: Path, identity: dict):
        self.root = root
        self.identity_sha256 = digest(identity)
        root.mkdir(parents=True, exist_ok=True)
        path = root / "manifest.json"
        contract = {"version": "habitbench.memory_user_checkpoint.v1",
                    "identity_sha256": self.identity_sha256}
        if path.exists() and json.loads(path.read_text()) != contract:
            raise ValueError("Memory-user checkpoint input/config/source mismatch; use a new output directory")
        if not path.exists():
            atomic_json(path, contract)

    def _path(self, job: dict) -> Path:
        return self.root / (digest(job) + ".json")

    def get(self, job: dict):
        path = self._path(job)
        if not path.exists():
            return None
        payload = json.loads(path.read_text())
        result = payload["result"]
        if (payload.get("identity_sha256") != self.identity_sha256
                or payload.get("result_sha256") != digest(result)
                or result.get("user_id") != job["user_id"]
                or result.get("context_id") != job["context_id"]
                or set(result.get("predictions", {})) != {p["probe_id"] for p in job["probes"]}):
            raise ValueError("Memory-user checkpoint content mismatch")
        return result

    def put(self, job: dict, result: dict):
        if (result.get("user_id") != job["user_id"]
                or result.get("context_id") != job["context_id"]
                or set(result.get("predictions", {})) != {p["probe_id"] for p in job["probes"]}):
            raise ValueError("Cannot checkpoint an incomplete memory user")
        atomic_json(self._path(job), {"identity_sha256": self.identity_sha256,
                    "result_sha256": digest(result), "result": result})
