from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.run_supplementary_analysis import (
    DEFAULT_DATASETS,
    _load_suite_datasets,
    _parse_domain_suite_roots,
)


class SupplementaryRunnerTest(unittest.TestCase):
    def test_manifest_selects_exact_four_domain_dataset_versions(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            suite_root = Path(temporary)
            datasets = {
                "food": {
                    "dataset_dir": "/datasets/food-v5",
                    "domain_filter": None,
                },
                "finance": {
                    "dataset_dir": "/datasets/finance-software-v1.4",
                    "domain_filter": "finance",
                },
                "software": {
                    "dataset_dir": "/datasets/finance-software-v1.4",
                    "domain_filter": "software",
                },
                "travel": {
                    "dataset_dir": "/datasets/travel-v16",
                    "domain_filter": None,
                },
            }
            (suite_root / "shard_plan.manifest.json").write_text(
                json.dumps({"datasets": datasets}), encoding="utf-8"
            )

            resolved = _load_suite_datasets(suite_root)

            self.assertEqual(set(resolved), set(datasets))
            self.assertEqual(resolved["travel"], (Path("/datasets/travel-v16"), None))
            self.assertEqual(
                resolved["finance"],
                (Path("/datasets/finance-software-v1.4"), "finance"),
            )

    def test_suite_without_manifest_uses_current_four_domain_mapping(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            self.assertEqual(
                _load_suite_datasets(Path(temporary)), DEFAULT_DATASETS
            )

    def test_domain_suite_override_resolves_newer_travel_root(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            overrides = _parse_domain_suite_roots(
                [f"travel={root}"]
            )
            self.assertEqual(overrides, {"travel": root.resolve()})


if __name__ == "__main__":
    unittest.main()
