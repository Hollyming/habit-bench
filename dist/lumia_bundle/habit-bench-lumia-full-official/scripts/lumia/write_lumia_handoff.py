#!/usr/bin/env python
"""Write a concise Lumia handoff for the HABIT-Bench full official run."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict


ROOT = Path(".")
DEFAULT_DATASET = ROOT / "runs/habit_bench_balanced_v0_3_official_subset_90"
DEFAULT_BUNDLE = ROOT / "dist/lumia_bundle/habit-bench-lumia-full-official.tar.gz"


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def format_map(values: Dict[str, Any]) -> str:
    return ", ".join(f"{key}: {values[key]}" for key in sorted(values))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--host-placeholder", default="USER@LUMIA")
    parser.add_argument("--remote-dir", default="/path/to/habitbench-lumia")
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "dist/lumia_bundle/habit-bench-lumia-full-official.handoff.md",
    )
    args = parser.parse_args()

    subset = read_json(args.dataset_dir / "reports/official_subset_manifest.json")
    provenance = read_json(args.dataset_dir / "reports/domain_provenance_summary.json")
    bundle_manifest = read_json(args.bundle.with_suffix(args.bundle.suffix + ".manifest.json"))
    goal_status = read_json(args.root / "runs/goal_status.json")

    counts = subset["counts"]
    tarball = bundle_manifest["tarball"]
    host = args.host_placeholder
    remote_dir = args.remote_dir
    launcher = "python ./scripts/lumia/launch_lumia_remote.py"
    guarded_cycle = "python ./scripts/lumia/run_lumia_guarded_full_cycle.py"
    scp_hint = '--scp "scp -O"'
    python_hint = "--remote-env PYTHON_BIN=/home/jmzhang/miniconda3/bin/python"
    model_hints = [
        "--remote-env HABITBENCH_LLM_MODEL=/home/jmzhang/models/Qwen2.5-14B-Instruct",
        "--remote-env HABITBENCH_EMBED_MODEL=/home/jmzhang/models/e5-base-v2",
        "--remote-env HABITBENCH_EMBED_DIMS=768",
        "--remote-env HABITBENCH_MAX_MODEL_LEN=16384",
        "--remote-env HABITBENCH_MEMORY_LLM_MAX_TOKENS=256",
        "--remote-env HABITBENCH_OFFICIAL_TIMEOUT_SEC=21600",
        "--remote-env HABITBENCH_PROGRESS_EVERY=100",
    ]
    slurm_srun_hint = '--remote-run-prefix "srun --partition=L40S --gres=gpu:1 --time=06:00:00 bash -lc"'
    slurm_batch_block = "\n".join(
        [
            "  --slurm-detached \\",
            "  --slurm-partition L40S \\",
            "  --slurm-gres gpu:1 \\",
            "  --slurm-time 06:00:00 \\",
            "  --slurm-job-name habitbench \\",
        ]
    )
    all_env_hints = [python_hint, *model_hints]
    env_block = " \\\n".join(f"  {hint}" for hint in all_env_hints)

    lines = [
        "# Lumia Full Official Handoff",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "## Current Artifact",
        "",
        f"- Bundle: `{tarball['path']}`",
        f"- SHA256: `{tarball['sha256']}`",
        f"- Bytes: {tarball['bytes']}",
        f"- Bundle files: {bundle_manifest['file_count']}",
        "",
        "## Dataset",
        "",
        f"- Dataset: `{args.dataset_dir}`",
        f"- Probes: {counts['probes']}",
        f"- Users: {counts['users']}",
        f"- Sessions: {counts['sessions']}",
        f"- Family counts: {format_map(subset['by_habit_family'])}",
        f"- Probe types: {format_map(subset['by_probe_type'])}",
        f"- Stress variants: {format_map(subset['by_stress_variant'])}",
        f"- Domain provenance: `{provenance['status']}`",
        f"- Family-domain contract: `{provenance['source_contract']['family_domain_contract']}`",
        "",
        "## Launch",
        "",
        "Preferred single-command guarded cycle:",
        "",
        "```bash",
        f"{guarded_cycle} \\",
        f"  --host {host} \\",
        f"  --remote-dir {remote_dir} \\",
        f"  {scp_hint} \\",
        f"{env_block} \\",
        f"{slurm_batch_block}",
        "  --wait-timeout-sec 86400 \\",
        "  --wait-poll-sec 300 \\",
        "  --execute",
        "```",
        "",
        "Run it without `--execute` first to inspect both child launcher plans.",
        "",
        "Dry-run first:",
        "",
        "```bash",
        f"{launcher} \\",
        f"  --host {host} \\",
        f"  --remote-dir {remote_dir} \\",
        f"  {scp_hint} \\",
        f"{env_block} \\",
        f"  {slurm_srun_hint}",
        "```",
        "",
        "For the currently configured `lumia` SSH alias, `scp -O` is required",
        "because the JumpServer SFTP-mode scp cannot see the same home path;",
        "`PYTHON_BIN=/home/jmzhang/miniconda3/bin/python` avoids the login",
        "node's missing `python3-venv`; `--remote-run-prefix` can route short",
        "preflight/download runs through Slurm GPU nodes; and `--slurm-detached`",
        "submits the long full run with `sbatch` so Slurm owns the process",
        "lifetime. The current Lumia offline path uses local",
        "`/home/jmzhang/models/Qwen2.5-14B-Instruct` and",
        "`/home/jmzhang/models/e5-base-v2`; the latter requires",
        "`HABITBENCH_EMBED_DIMS=768`.",
        "",
        "Run remote preflight only before starting the long job:",
        "",
        "```bash",
        f"{launcher} \\",
        f"  --host {host} \\",
        f"  --remote-dir {remote_dir} \\",
        f"  {scp_hint} \\",
        f"{env_block} \\",
        f"  {slurm_srun_hint} \\",
        "  --preflight-only \\",
        "  --execute",
        "```",
        "",
        "This fetches the returned preflight manifests and writes local",
        "`runs/lumia_preflight_import_audit.json` and `.md`; require JSON",
        "`status=pass`",
        "before launching the detached full run.",
        "",
        "Download/cache open models only and audit the returned manifest:",
        "",
        "```bash",
        f"{launcher} \\",
        f"  --host {host} \\",
        f"  --remote-dir {remote_dir} \\",
        f"  {scp_hint} \\",
        f"{env_block} \\",
        f"  {slurm_srun_hint} \\",
        "  --download-models-only \\",
        "  --execute",
        "```",
        "",
        "Preferred one-command option: preflight, download/cache open models,",
        "fetch/audit both returned manifest sets locally, then start the detached",
        "full run only if both audits pass:",
        "",
        "```bash",
        f"{launcher} \\",
        f"  --host {host} \\",
        f"  --remote-dir {remote_dir} \\",
        f"  {scp_hint} \\",
        f"{env_block} \\",
        f"{slurm_batch_block}",
        "  --preflight-download-then-detached \\",
        "  --execute",
        "```",
        "",
        "After those audits pass, this mode reuses the same remote workspace for",
        "the detached full run and sets `HABITBENCH_SKIP_MODEL_DOWNLOAD=1`, so the",
        "audited `model_download_manifest.json` is preserved for final import.",
        "",
        "Preflight-only one-command option for hosts where model download has",
        "already been audited separately:",
        "",
        "```bash",
        f"{launcher} \\",
        f"  --host {host} \\",
        f"  --remote-dir {remote_dir} \\",
        f"  {scp_hint} \\",
        f"{env_block} \\",
        f"{slurm_batch_block}",
        "  --preflight-then-detached \\",
        "  --execute",
        "```",
        "",
        "Start the real detached Lumia run with an already-audited model/cache",
        "manifest and an existing remote workspace:",
        "",
        "```bash",
        f"{launcher} \\",
        f"  --host {host} \\",
        f"  --remote-dir {remote_dir} \\",
        f"  {scp_hint} \\",
        f"{env_block} \\",
        f"{slurm_batch_block}",
        "  --reuse-remote-workspace \\",
        "  --detached \\",
        "  --execute",
        "```",
        "",
        "Check status and log tail:",
        "",
        "```bash",
        f"{launcher} \\",
        f"  --host {host} \\",
        f"  --remote-dir {remote_dir} \\",
        f"  {scp_hint} \\",
        "  --status-only \\",
        "  --execute",
        "```",
        "",
        "Preferred collector after the detached job starts: wait for",
        "`habitbench_remote_e2e.exitcode=0`, then fetch and import/audit locally:",
        "",
        "```bash",
        f"{launcher} \\",
        f"  --host {host} \\",
        f"  --remote-dir {remote_dir} \\",
        f"  {scp_hint} \\",
        "  --wait-and-fetch \\",
        "  --wait-timeout-sec 86400 \\",
        "  --wait-poll-sec 300 \\",
        "  --execute",
        "```",
        "",
        "This mode does not import partial results when the detached job is still",
        "running, missing an exit-code file, or exits nonzero.",
        "",
        "Fetch and audit after the job stops and `exit_code=0` is present:",
        "",
        "```bash",
        f"{launcher} \\",
        f"  --host {host} \\",
        f"  --remote-dir {remote_dir} \\",
        f"  {scp_hint} \\",
        "  --fetch-only \\",
        "  --execute",
        "```",
        "",
        "## Current Lumia Evidence",
        "",
        "- SSH alias `lumia` resolves to login/storage host `storage-hdd`; GPU",
        "  work must be routed through Slurm.",
        "- Verified Slurm GPU smoke: `srun --partition=RTX4090 --gres=gpu:1`",
        "  reached `gpu-4090-2` with one RTX 4090. The recommended full-run",
        "  partition is L40S because the detected local LLM is a 14B model.",
        "- Remote readiness and GPU/dataset/import preflight pass on the RTX4090",
        "  node when using the conda Python bootstrap above.",
        "- Hugging Face downloads on Lumia require proxy to be disabled first:",
        "  run `proxy_off` from the account bashrc before model download; use",
        "  `proxy_on` only when another network path explicitly needs it.",
        "- Current full-run submission path uses `--slurm-detached`, which writes",
        "  `habitbench_remote_e2e.sbatch`, records `habitbench_remote_e2e.jobid`,",
        "  and lets Slurm own the long job lifecycle.",
        "",
        "## Completion Gate",
        "",
        "Do not claim completion until all of the following are true:",
        "",
        "- `habitbench_remote_e2e.exitcode` exists on Lumia and contains `0`.",
        "- `full_official_results/run_manifests/<run_id>/lumia_preflight_manifest.json` has `status=pass`.",
        "- `full_official_results/run_manifests/<run_id>/suite_end_manifest.json` has `extra.exit_code=0`.",
        "- If present, `full_official_results/run_manifests/<run_id>/e2e_end_manifest.json` has `extra.exit_code=0`.",
        "- `runs/lumia_manifests/model_download_manifest.json` exists with top-level `status=pass` and `dry_run=false`.",
        "- Every model row in `model_download_manifest.json` has `status=pass` and a non-empty `cache_path`.",
        "- `full_official_results/mem0_full_llm_openai/` exists and contains raw/scored predictions plus config/runtime/report files.",
        "- `full_official_results/graphiti_full_llm_episode_kuzu/` exists and contains raw/scored predictions plus config/runtime/report files.",
        "- `full_official_results/collected/official_results_collected.csv` exists.",
        "- `full_official_results/audit/full_official_audit.json` has `status=pass`.",
        "",
        "## Current Missing Items",
        "",
    ]
    missing = goal_status["items"][-1].get("missing", [])
    lines.extend(f"- {item}" for item in missing)
    lines.extend(
        [
            "",
            "## Notes",
            "",
        "- The benchmark uses `allenai/WildChat` as the single real prompt seed source.",
        "- The official subset is real-prompt-seeded, domain-grounded, and synthetic-longitudinal.",
        "- The nine habit families use unique representative domains; they are not nine separate external datasets.",
        "- This handoff is a sidecar document next to the bundle; it is intentionally not included inside the tarball because it records the tarball hash.",
    ]
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(args.out), "sha256": tarball["sha256"]}, indent=2))


if __name__ == "__main__":
    main()
