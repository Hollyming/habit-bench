import importlib.util
import sys
import types
from pathlib import Path

import numpy as np


def test_cosine_distance_supports_bge_512_dimensions() -> None:
    source = (
        Path(__file__).resolve().parents[1]
        / "methods" / "MIRIX" / "mirix" / "orm" / "sqlite_functions.py"
    )
    old_mirix = sys.modules.get("mirix")
    old_constants = sys.modules.get("mirix.constants")
    fake_mirix = types.ModuleType("mirix")
    fake_mirix.__path__ = []
    fake_constants = types.ModuleType("mirix.constants")
    fake_constants.MAX_EMBEDDING_DIM = 4096
    sys.modules["mirix"] = fake_mirix
    sys.modules["mirix.constants"] = fake_constants
    try:
        spec = importlib.util.spec_from_file_location("_mirix_sqlite_functions_test", source)
        assert spec and spec.loader
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        first = np.zeros(512, dtype=np.float32)
        first[0] = 1.0
        second = np.zeros(512, dtype=np.float32)
        second[1] = 1.0
        assert module.cosine_distance(module.adapt_array(first), module.adapt_array(first)) == 0.0
        assert module.cosine_distance(module.adapt_array(first), module.adapt_array(second)) == 1.0
    finally:
        if old_mirix is None:
            sys.modules.pop("mirix", None)
        else:
            sys.modules["mirix"] = old_mirix
        if old_constants is None:
            sys.modules.pop("mirix.constants", None)
        else:
            sys.modules["mirix.constants"] = old_constants
