"""Shared training engine for new records.

The canonical engine is the upstream baseline in ``train_gpt.py`` at the repo
root. New record folders must import this facade instead of copying that file
into ``records/``. A record that needs different behavior can load the engine
and patch its classes or functions before calling ``main()``:

    from shared.train_core import load_engine

    engine = load_engine()
    engine.Hyperparameters.matrix_lr = 0.02
    engine.main()

Environment overrides work through ``run()`` because the baseline reads its
hyperparameters from the environment when the module is loaded.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from types import ModuleType

REPO_ROOT = Path(__file__).resolve().parents[1]
ENGINE_PATH = REPO_ROOT / "train_gpt.py"


def load_engine(path: Path = ENGINE_PATH) -> ModuleType:
    """Load the baseline engine as a module without executing its ``main()``."""
    spec = importlib.util.spec_from_file_location("parameter_golf_engine", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load training engine from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run(overrides: dict[str, object] | None = None) -> None:
    """Run the shared engine after applying environment overrides."""
    for key, value in (overrides or {}).items():
        os.environ[str(key)] = str(value)
    load_engine().main()
