# Shared train engine for new records

`records/` is a competition archive: every historical entry is an immutable
copy of a training script. **Do not deduplicate or edit historical entries.**
The guard freezes them by SHA-256 in `historical_records.json`.

New records must not add another copy. Use the shared facade:

- `shared/train_core.py` — loads the canonical engine (`train_gpt.py` at the
  repo root, the upstream baseline) and exposes `run(overrides)` and
  `load_engine()`. Hyperparameters come from the environment, so an entry only
  needs a `CONFIG` dict; classes/functions can be patched through
  `load_engine()` when an experiment needs different behavior.
- `records/_template/` — copy this folder to `records/<track>/<date>_<name>/`
  and edit `CONFIG`.

## Guard

```bash
python3 shared/check_records.py
```

It checks that:

1. every historical file still matches the frozen snapshot (no edits, additions
   or deletions in existing record folders);
2. every new record has a thin `train_gpt.py` (≤ 200 lines / 8 KB) that imports
   `shared.train_core`;
3. no Python file in a new record is byte-identical to an archived script.

Exit code 0 is a pass. `--write-manifest` exists only for the one-time archive
freeze; do not use it to bless a newly copied script.

## Tests

```bash
python3 -m unittest discover -s shared/tests -v
```

## Upstream submission caveat

The competition counts code bytes in `train_gpt.py` and requires the script to
be self-contained. Thin entries are for the fork's development workflow. Before
opening an upstream PR, the shared engine and record-specific modules must be
packaged into a single self-contained `train_gpt.py` inside the record folder.
The existing JEPA submission remains on the `jepa-hybrid-submission` branch.
