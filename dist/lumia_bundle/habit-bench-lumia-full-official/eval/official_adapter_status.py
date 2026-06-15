#!/usr/bin/env python
"""Check official-baseline integration readiness for HABIT-Bench.

This script does not run method-inspired proxies. It reports whether official
packages/repos needed for method integrations are available in the current
environment and writes a machine-readable manifest.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPO_ROOT = PROJECT_ROOT / "third_party" / "official-baselines" / "repos"


BASELINES = [
    {
        "name": "Mem0",
        "kind": "pip_package",
        "package": "mem0ai",
        "import_names": ["mem0"],
        "install_hint": "pip install mem0ai",
        "official_source": "https://github.com/mem0ai/mem0",
        "adapter_status": "official_api_retrieval_adapter_ran",
        "notes": "HABIT-Bench adapter uses official Memory.add(..., infer=False) and Memory.search with local HuggingFace embeddings and Qdrant. Full LLM fact extraction/update is not reproduced yet.",
    },
    {
        "name": "Zep/Graphiti",
        "kind": "pip_package_plus_backend",
        "package": "graphiti-core",
        "import_names": ["graphiti_core"],
        "install_hint": "pip install graphiti-core",
        "official_source": "https://github.com/getzep/graphiti",
        "adapter_status": "official_code_graph_storage_search_adapter_ran",
        "notes": "HABIT-Bench adapter uses official Kuzu driver, EntityNode/EntityEdge writes, and Graphiti search_ with edge cosine. Full Graphiti requires LLM episode extraction/KG resolution and a production graph backend; local Kuzu BM25 full-text index was unavailable.",
    },
    {
        "name": "A-MEM",
        "kind": "repo_or_package",
        "package": None,
        "import_names": ["a_mem", "agentic_memory"],
        "env_repo_var": "AMEM_REPO",
        "default_repo_path": "a-mem",
        "install_hint": "clone https://github.com/agiresearch/a-mem or https://github.com/WujiangXu/A-mem-sys and expose its Python path",
        "official_source": "https://github.com/agiresearch/a-mem",
        "adapter_status": "official_code_retrieval_adapter_ran",
        "notes": "HABIT-Bench adapter uses official AgenticMemorySystem.add_note and search_agentic. LLM-based process_memory/evolution is disabled pending a live backend.",
    },
    {
        "name": "SeCom",
        "kind": "paper_method",
        "package": None,
        "import_names": [],
        "env_repo_var": "SECOM_REPO",
        "default_repo_path": "SeCom",
        "install_hint": "provide official repo path through SECOM_REPO if released",
        "official_source": "https://github.com/microsoft/SeCom",
        "adapter_status": "official_code_retrieval_adapter_ran",
        "notes": "HABIT-Bench adapter uses official SeCom.retrieve_external_memory with session-level BM25. Full paper settings require LLM segmentation and compression dependencies.",
    },
    {
        "name": "RMM",
        "kind": "paper_method",
        "package": None,
        "import_names": [],
        "env_repo_var": "RMM_REPO",
        "install_hint": "provide official repo path through RMM_REPO if released",
        "official_source": "https://aclanthology.org/2025.acl-long.413/",
        "adapter_status": "no_public_official_code_found",
        "notes": "No local or public official package/repository was found; current evaluator includes only a method-inspired proxy unless authors release code or RMM_REPO is provided.",
    },
    {
        "name": "O-Mem",
        "kind": "paper_method",
        "package": None,
        "import_names": [],
        "env_repo_var": "OMEM_REPO",
        "default_repo_path": "O-Mem",
        "install_hint": "provide official repo path through OMEM_REPO if released",
        "official_source": "https://github.com/OPPO-PersonalAI/O-Mem",
        "adapter_status": "official_code_retrieval_adapter_ran",
        "notes": "HABIT-Bench adapter uses official SimpleMemory/MemoryChain/MemoryManager.retrieve_from_memory_soft_segmentation with injected visible sessions. Full O-Mem active profiling and generation are not reproduced.",
    },
]


def module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def repo_available(env_var: str | None, default_repo_path: str | None = None) -> Dict[str, Any]:
    if not env_var and not default_repo_path:
        return {"available": False, "reason": "no_repo_env_var_or_default_path_defined"}
    value = os.environ.get(env_var)
    source = "env_var"
    if value:
        path = Path(value)
    elif default_repo_path:
        path = DEFAULT_REPO_ROOT / default_repo_path
        source = "default_repo_path"
    else:
        return {"available": False, "env_var": env_var, "reason": "env_var_not_set"}
    return {
        "available": path.exists(),
        "env_var": env_var,
        "path": str(path),
        "source": source,
        "reason": "ok" if path.exists() else "path_not_found",
    }


def check() -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    for baseline in BASELINES:
        imports = {
            name: module_available(name)
            for name in baseline.get("import_names", [])
        }
        package_available = any(imports.values()) if imports else False
        repo = repo_available(baseline.get("env_repo_var"), baseline.get("default_repo_path"))
        runnable = package_available or repo["available"]
        rows.append(
            {
                **baseline,
                "import_available": imports,
                "repo_available": repo,
                "runnable_in_current_env": runnable,
                "current_status": "ready_to_wire" if runnable else "missing_dependency_or_repo",
            }
        )
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "official_adapter_scope": "status_check_only",
        "baselines": rows,
    }


def write_markdown(path: Path, manifest: Dict[str, Any]) -> None:
    lines = [
        "# Official Baseline Integration Status",
        "",
        "This file distinguishes official-method readiness from lightweight proxy results.",
        "",
        "| method | runnable now | adapter status | missing / next action | source |",
        "| --- | ---: | --- | --- | --- |",
    ]
    for row in manifest["baselines"]:
        imports = row.get("import_available", {})
        missing = row["install_hint"]
        if row["runnable_in_current_env"]:
            missing = "dependency/repo found; implement or run adapter"
        lines.append(
            f"| {row['name']} | {str(row['runnable_in_current_env']).lower()} | {row['adapter_status']} | {missing} | {row['official_source']} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("./runs/official_adapter_status"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest = check()
    (args.out_dir / "official_adapter_status.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    write_markdown(args.out_dir / "official_adapter_status.md", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
