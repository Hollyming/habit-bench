#!/usr/bin/env python
"""Deprecated alias for the current four-domain shard-plan generator.

The former implementation hard-coded Food v4, Finance/Software v1.3, three
ClusterX A800 nodes, and human-audit preparation.  Those assumptions are no
longer valid.  This compatibility name now exposes exactly the generic
``create_shard_plan`` CLI, whose defaults are Food v5, Finance/Software v1.4,
and Travel v16.  New automation should call ``scripts/create_shard_plan.py``.
"""

from __future__ import annotations

import sys

from scripts.create_shard_plan import main


if __name__ == "__main__":
    print(
        "warning: create_v3_experiment_plans.py is deprecated; using the "
        "current four-domain create_shard_plan interface",
        file=sys.stderr,
    )
    main()
