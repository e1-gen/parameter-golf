# New record template

Do not edit this folder. Copy it to a new folder named
`records/<track>/<YYYY-MM-DD>_<Name>/`, then:

1. Fill `CONFIG` in `train_gpt.py` (environment overrides for the shared engine).
2. Put record-specific code in sibling modules; keep `train_gpt.py` thin and
   importing `shared.train_core`.
3. Add `README.md`, `submission.json` and run logs for the competition entry.
4. Run `python3 shared/check_records.py` from the repo root. It must pass.

The guard freezes all existing competition entries and rejects a new entry that
copies a historical script or does not import the shared engine.

Note: upstream counts all submission code bytes inside `train_gpt.py` and
requires a self-contained script. Thin entries are the fork's development
layout; a future upstream PR must package the engine into the record's
`train_gpt.py` first (see `shared/README.md`).
