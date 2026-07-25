#!/usr/bin/env python3
"""
Personalization Lab - Memory Evaluation System
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import ConfigLoader, PROJECT_ROOT
from src.evaluator import create_evaluator
from src.agent import list_available_methods


def parse_persona_ids(value: str) -> list[int]:
    """Parse comma-separated IDs and inclusive ranges (for example 1-4,9)."""
    ids: list[int] = []
    for raw_part in value.split(","):
        part = raw_part.strip()
        if not part:
            continue
        if "-" in part:
            start_text, end_text = part.split("-", 1)
            start, end = int(start_text), int(end_text)
            if start > end:
                raise argparse.ArgumentTypeError(f"Invalid persona range: {part}")
            ids.extend(range(start, end + 1))
        else:
            ids.append(int(part))
    if not ids or any(persona_id <= 0 for persona_id in ids):
        raise argparse.ArgumentTypeError("Persona IDs must be positive integers")
    return list(dict.fromkeys(ids))


def parse_sample_ids(value: str) -> list[str]:
    """Parse comma-separated dataset sample IDs while preserving order."""
    ids = [part.strip() for part in value.split(",") if part.strip()]
    if not ids:
        raise argparse.ArgumentTypeError("Sample IDs must not be empty")
    return list(dict.fromkeys(ids))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Personalization Lab - Memory Evaluation System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("-m", "--method", type=str, help="Method config name")
    parser.add_argument("-d", "--dataset", type=str, help="Dataset name")
    parser.add_argument("-o", "--output-dir", type=str, default="./outputs", help="Output directory")
    parser.add_argument("--dry-run", action="store_true", help="Dry run mode")
    parser.add_argument("-q", "--quiet", action="store_true", help="Quiet mode")
    parser.add_argument("--list-methods", action="store_true", help="List available methods")
    parser.add_argument("--list-datasets", action="store_true", help="List available datasets")
    parser.add_argument("--list-agents", action="store_true", help="List available agent types")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument(
        "--persona-ids",
        type=parse_persona_ids,
        help="Override dataset personas, e.g. 1-4,9 (enables isolated shards)",
    )
    parser.add_argument(
        "--sample-ids",
        type=parse_sample_ids,
        help="Override dataset samples, e.g. conv-26,conv-30 (LoCoMo shards)",
    )

    return parser.parse_args()


def list_methods(config_loader: ConfigLoader) -> None:
    print("\nAvailable methods:")
    print("=" * 60)

    configs = config_loader.list_method_configs()
    if not configs:
        print("  (none)")
    else:
        for name in sorted(configs):
            try:
                cfg = config_loader.load_method_config(name)
                print(f"  {name}")
                print(f"    type: {cfg.method_type}, model: {cfg.model.name}")
            except Exception as e:
                print(f"  {name} (load failed: {e})")

    print("=" * 60)


def list_datasets(config_loader: ConfigLoader) -> None:
    print("\nAvailable datasets:")
    print("=" * 60)

    configs = config_loader.list_dataset_configs()
    if not configs:
        print("  (none)")
    else:
        for name in sorted(configs):
            try:
                cfg = config_loader.load_dataset_config(name)
                print(f"  {name} ({cfg.language})")
            except Exception as e:
                print(f"  {name} (load failed: {e})")

    print("=" * 60)


def list_agents_info() -> None:
    print("\nAvailable agent types:")
    print("=" * 60)
    for method in list_available_methods():
        print(f"  - {method}")
    print("=" * 60)


def main() -> int:
    args = parse_args()
    config_loader = ConfigLoader()

    if args.list_methods:
        list_methods(config_loader)
        return 0

    if args.list_datasets:
        list_datasets(config_loader)
        return 0

    if args.list_agents:
        list_agents_info()
        return 0

    if not args.method:
        print("Error: please specify method (-m/--method)")
        return 1

    if not args.dataset:
        print("Error: please specify dataset (-d/--dataset)")
        return 1

    try:
        dataset_overrides = {}
        if args.persona_ids is not None:
            dataset_overrides["persona_ids"] = args.persona_ids
        if args.sample_ids is not None:
            dataset_overrides["sample_ids"] = args.sample_ids

        evaluator = create_evaluator(
            method_config_name=args.method,
            dataset_name=args.dataset,
            config_loader=config_loader,
            output_dir=Path(args.output_dir),
            dry_run=args.dry_run,
            verbose=not args.quiet,
            resume=args.resume,
            dataset_overrides=dataset_overrides or None,
        )
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return 1
    except Exception as e:
        print(f"Init failed: {e}")
        return 1

    print(f"\nMethod: {args.method}, Dataset: {args.dataset}, Dry Run: {args.dry_run}\n")

    try:
        report = evaluator.run()
        summary = report.summary
        print(f"\nResults: {summary.get('correct', 0)}/{summary.get('total', 0)} "
              f"({summary.get('overall_accuracy', 0):.2%})")
        print(f"Output: {args.output_dir}")
        return 0
    except KeyboardInterrupt:
        print("\nInterrupted")
        return 130
    except Exception as e:
        print(f"\nFailed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
