"""Behavior tests for the new-record guard (written before the guard)."""

from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from shared import check_records

REPO_ROOT = Path(__file__).resolve().parents[2]
OLD_ENTRY = "track_10min_16mb/2026-01-01_Old"
OLD_SCRIPT = "ENGINE = 'old shared baseline'\n\n\ndef main():\n    return ENGINE\n"

THIN_ENTRY = """\
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from shared.train_core import run  # noqa: E402

CONFIG = {"RUN_ID": "new-record"}

if __name__ == "__main__":
    run(CONFIG)
"""


class Fixture:
    def __init__(self, root: Path) -> None:
        self.root = root
        (root / "shared").mkdir(parents=True)
        (root / "shared" / "train_core.py").write_text("", encoding="utf-8")
        self.write(OLD_ENTRY, "train_gpt.py", OLD_SCRIPT)
        self.write(OLD_ENTRY, "README.md", "old record\n")
        self.write("track_10min_16mb/2026-01-02_Other", "train_gpt.py", OLD_SCRIPT)
        self.manifest = check_records.build_manifest(root)
        (root / "shared" / "historical_records.json").write_text(
            json.dumps(self.manifest), encoding="utf-8"
        )

    def write(self, entry: str, name: str, text: str) -> Path:
        path = self.root / "records" / entry / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return path

    def check(self) -> list[str]:
        return check_records.check_repository(self.root, self.manifest)


class GuardTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.repo = Path(self._tmp.name)
        self.fx = Fixture(self.repo)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def assert_guard_error(self, fragment: str) -> None:
        errors = self.fx.check()
        self.assertTrue(any(fragment in error for error in errors), errors)

    def test_untouched_archive_passes(self) -> None:
        self.assertEqual(self.fx.check(), [])

    def test_modified_historical_script_fails(self) -> None:
        self.fx.write(OLD_ENTRY, "train_gpt.py", OLD_SCRIPT + "# edited\n")
        self.assert_guard_error("historical record changed")

    def test_extra_file_in_historical_record_fails(self) -> None:
        self.fx.write(OLD_ENTRY, "notes.txt", "new\n")
        self.assert_guard_error("was added")

    def test_deleted_historical_record_fails(self) -> None:
        (self.repo / "records" / OLD_ENTRY / "train_gpt.py").unlink()
        self.assert_guard_error("is missing")

    def test_thin_new_record_importing_shared_passes(self) -> None:
        self.fx.write("track_10min_16mb/2026-11-08_New", "train_gpt.py", THIN_ENTRY)
        self.assertEqual(self.fx.check(), [])

    def test_new_record_without_shared_import_fails(self) -> None:
        self.fx.write("track_10min_16mb/2026-11-08_New", "train_gpt.py", "def main():\n    pass\n")
        self.assert_guard_error("must import from shared.train_core")

    def test_new_record_with_large_script_fails(self) -> None:
        body = "\n".join(f"VALUE_{i} = {i}" for i in range(check_records.MAX_ENTRY_LINES + 10))
        self.fx.write("track_10min_16mb/2026-11-08_New", "train_gpt.py", body + "\n")
        self.assert_guard_error("import shared.train_core instead of copying")

    def test_new_record_with_identical_copy_fails(self) -> None:
        self.fx.write("track_10min_16mb/2026-11-08_New", "engine_copy.py", OLD_SCRIPT)
        self.fx.write("track_10min_16mb/2026-11-08_New", "train_gpt.py", THIN_ENTRY)
        self.assert_guard_error("byte-identical copy")

    def test_template_entry_is_compliant(self) -> None:
        errors = check_records.check_new_entry(REPO_ROOT / "records" / "_template", set())
        self.assertEqual(errors, [])


class CommandTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.repo = Path(self._tmp.name)
        self.fx = Fixture(self.repo)
        self.manifest = self.repo / "shared" / "historical_records.json"

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def run_main(self, *args: str) -> tuple[int, str]:
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = check_records.main(["--root", str(self.repo), *args])
        return code, out.getvalue() + err.getvalue()

    def test_main_passes_on_clean_repo(self) -> None:
        code, output = self.run_main()
        self.assertEqual(code, 0)
        self.assertIn("historical records unchanged", output)

    def test_main_fails_on_changed_historical_record(self) -> None:
        self.fx.write(OLD_ENTRY, "train_gpt.py", OLD_SCRIPT + "# edited\n")
        code, output = self.run_main()
        self.assertEqual(code, 1)
        self.assertIn("historical record changed", output)

    def test_main_requires_manifest(self) -> None:
        self.manifest.unlink()
        code, output = self.run_main()
        self.assertEqual(code, 2)
        self.assertIn("missing manifest", output)

    def test_write_manifest_snapshots_entries(self) -> None:
        self.manifest.unlink()
        code, output = self.run_main("--manifest", str(self.manifest), "--write-manifest")
        self.assertEqual(code, 0)
        self.assertTrue(self.manifest.is_file())
        self.assertIn("historical records", output)


if __name__ == "__main__":
    unittest.main()
