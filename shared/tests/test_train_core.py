"""Behavior tests for the shared engine facade."""

from __future__ import annotations

import importlib.util
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from shared import train_core

REPO_ROOT = Path(__file__).resolve().parents[2]


class FakeEngine:
    def __init__(self) -> None:
        self.ran = False

    def main(self) -> None:
        self.ran = True


class TrainCoreTest(unittest.TestCase):
    def test_load_engine_does_not_run_main(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "engine.py"
            path.write_text("RAN = False\n\ndef main():\n    global RAN\n    RAN = True\n", encoding="utf-8")
            module = train_core.load_engine(path)
            self.assertFalse(module.RAN)
            module.main()
            self.assertTrue(module.RAN)

    def test_run_applies_overrides_before_loading_engine(self) -> None:
        fake = FakeEngine()
        seen: dict[str, str] = {}

        def load() -> FakeEngine:
            seen["RUN_ID"] = os.environ["RUN_ID"]
            return fake

        with mock.patch.object(train_core, "load_engine", side_effect=load):
            train_core.run({"RUN_ID": "test-run"})
        self.assertEqual(seen["RUN_ID"], "test-run")
        self.assertTrue(fake.ran)
        os.environ.pop("RUN_ID", None)

    def test_template_entry_finds_repo_root(self) -> None:
        template = REPO_ROOT / "records" / "_template" / "train_gpt.py"
        spec = importlib.util.spec_from_file_location("record_template_entry", template)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.assertEqual(module._repo_root(template.resolve()), REPO_ROOT)

    def test_template_entry_rejects_path_outside_checkout(self) -> None:
        template = REPO_ROOT / "records" / "_template" / "train_gpt.py"
        spec = importlib.util.spec_from_file_location("record_template_entry_missing", template)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(RuntimeError):
                module._repo_root(Path(tmp) / "train_gpt.py")


if __name__ == "__main__":
    unittest.main()
