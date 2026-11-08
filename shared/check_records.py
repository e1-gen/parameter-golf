"""Guard against new copied training scripts under ``records/``.

Two rules:

1. Historical competition entries are frozen. Their file hashes are stored in
   ``shared/historical_records.json``; any change, addition or deletion fails.
2. New entries must be thin: ``train_gpt.py`` imports ``shared.train_core``,
   stays under the size limit, and no Python file may be a byte-identical copy
   of an archived script.

Run ``python3 shared/check_records.py`` from the repo root. Exit code 0 means
the archive is intact and every new entry uses the shared engine.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RECORDS_DIR = "records"
ENTRY_SCRIPT = "train_gpt.py"
MANIFEST_NAME = "historical_records.json"
MAX_ENTRY_BYTES = 8192
MAX_ENTRY_LINES = 200
SHARED_IMPORT_RE = re.compile(
    r"(?m)^\s*(?:from\s+shared\.train_core\s+import|import\s+shared\.train_core)\b"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, object]:
    return {"sha256": sha256(path), "size": path.stat().st_size}


def find_entries(records_root: Path) -> list[Path]:
    entries: list[Path] = []
    for track in sorted(records_root.iterdir()):
        if not track.is_dir() or track.name.startswith("_"):
            continue
        entries.extend(
            sorted(entry for entry in track.iterdir() if entry.is_dir() and not entry.name.startswith("_"))
        )
    return entries


def entry_files(entry: Path) -> list[Path]:
    return sorted(path for path in entry.rglob("*") if path.is_file())


def build_manifest(root: Path = REPO_ROOT) -> dict[str, object]:
    records_root = root / RECORDS_DIR
    records: dict[str, object] = {}
    for entry in find_entries(records_root):
        relative = entry.relative_to(records_root).as_posix()
        records[relative] = {
            "files": {path.relative_to(entry).as_posix(): file_record(path) for path in entry_files(entry)}
        }
    return {"generated_at": datetime.now().astimezone().isoformat(timespec="seconds"), "records": records}


def check_historical(root: Path, manifest: dict[str, object]) -> list[str]:
    errors: list[str] = []
    records_root = root / RECORDS_DIR
    current = {entry.relative_to(records_root).as_posix(): entry for entry in find_entries(records_root)}
    for relative in sorted(manifest["records"]):
        expected = manifest["records"][relative]["files"]
        entry = current.get(relative)
        if entry is None:
            errors.append(f"historical record changed: records/{relative} is missing")
            continue
        actual = {path.relative_to(entry).as_posix(): file_record(path) for path in entry_files(entry)}
        for name in sorted(set(expected) - set(actual)):
            errors.append(f"historical record changed: records/{relative}/{name} is missing")
        for name in sorted(set(actual) - set(expected)):
            errors.append(f"historical record changed: records/{relative}/{name} was added")
        for name in sorted(set(expected) & set(actual)):
            if actual[name] != expected[name]:
                errors.append(f"historical record changed: records/{relative}/{name} differs from the snapshot")
    return errors


def check_new_entry(entry: Path, historical_hashes: set[str]) -> list[str]:
    errors: list[str] = []
    script = entry / ENTRY_SCRIPT
    if not script.is_file():
        return [f"new record {entry.name}: missing {ENTRY_SCRIPT}"]
    text = script.read_text(encoding="utf-8", errors="replace")
    size = len(text.encode("utf-8"))
    lines = len(text.splitlines())
    if size > MAX_ENTRY_BYTES or lines > MAX_ENTRY_LINES:
        errors.append(
            f"new record {entry.name}: {ENTRY_SCRIPT} is {size} bytes/{lines} lines; "
            "import shared.train_core instead of copying the engine"
        )
    if not SHARED_IMPORT_RE.search(text):
        errors.append(f"new record {entry.name}: {ENTRY_SCRIPT} must import from shared.train_core")
    for path in entry.rglob("*.py"):
        if sha256(path) in historical_hashes:
            errors.append(
                f"new record {entry.name}: {path.relative_to(entry)} is a byte-identical copy "
                "of an archived script"
            )
    return errors


def check_repository(root: Path, manifest: dict[str, object]) -> list[str]:
    errors = check_historical(root, manifest)
    records_root = root / RECORDS_DIR
    historical_hashes = {
        file["sha256"] for record in manifest["records"].values() for file in record["files"].values()
    }
    engine = root / "train_gpt.py"
    if engine.is_file():
        historical_hashes.add(sha256(engine))
    for entry in find_entries(records_root):
        relative = entry.relative_to(records_root).as_posix()
        if relative not in manifest["records"]:
            errors.extend(check_new_entry(entry, historical_hashes))
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPO_ROOT, help="repository root")
    parser.add_argument("--manifest", type=Path, help="frozen historical snapshot")
    parser.add_argument(
        "--write-manifest",
        action="store_true",
        help="write the initial snapshot; only for the one-time archive freeze",
    )
    args = parser.parse_args(argv)
    root = args.root.resolve()
    manifest_path = args.manifest or root / "shared" / MANIFEST_NAME
    if args.write_manifest:
        manifest = build_manifest(root)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"wrote {manifest_path} with {len(manifest['records'])} historical records")
        return 0
    if not manifest_path.is_file():
        print(f"missing manifest: {manifest_path}", file=sys.stderr)
        return 2
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    errors = check_repository(root, manifest)
    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        print(f"FAIL: {len(errors)} problem(s)", file=sys.stderr)
        return 1
    new_count = sum(
        1
        for entry in find_entries(root / RECORDS_DIR)
        if entry.relative_to(root / RECORDS_DIR).as_posix() not in manifest["records"]
    )
    print(f"OK: {len(manifest['records'])} historical records unchanged; {new_count} new record(s) use shared.train_core")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
