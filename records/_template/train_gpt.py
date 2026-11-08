"""Template entry point for a new record.

Copy this folder to ``records/<track>/<YYYY-MM-DD>_<Name>/`` and edit CONFIG.
The training engine lives in ``shared/train_core.py``; do not copy the engine
into the record folder. Record-specific code belongs in sibling modules and may
load the engine itself:

    from shared.train_core import load_engine
    engine = load_engine()
    engine.Hyperparameters.matrix_lr = 0.02
    engine.main()
"""

from __future__ import annotations

import sys
from pathlib import Path


def _repo_root(start: Path) -> Path:
    for parent in start.parents:
        if (parent / "shared" / "train_core.py").is_file():
            return parent
    raise RuntimeError("shared/train_core.py not found: run from a parameter-golf checkout")


sys.path.insert(0, str(_repo_root(Path(__file__).resolve())))

from shared.train_core import run  # noqa: E402

CONFIG: dict[str, object] = {
    # "RUN_ID": "my-record",
    # "NUM_LAYERS": 9,
    # "DATA_PATH": "./data/datasets/fineweb10B_sp8192",
    # "TOKENIZER_PATH": "./data/tokenizers/fineweb_8192_bpe.model",
}

if __name__ == "__main__":
    run(CONFIG)
