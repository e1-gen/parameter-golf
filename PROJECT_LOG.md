# parameter-golf — project log

## 2026-11-08 16:14 — DISCOVERY-11 — shared train module for new records + archive guard (opencode)

**Why:** UB audit E1GEN-28 found 7,903 duplicate pairs; every `records/*/*/train_gpt.py` is a maintained copy. Historical entries are a competition archive and must not be deduplicated.

**What changed:**
- `shared/train_core.py` — shared facade: loads the canonical upstream baseline `train_gpt.py` and exposes `run(overrides)` / `load_engine()`. Records configure via environment and may patch engine classes before `main()`.
- `shared/check_records.py` + `shared/historical_records.json` — guard: freezes 37 historical entries / 216 files by SHA-256. New entries must have a thin `train_gpt.py` (≤200 lines / 8 KB) importing `shared.train_core`; no Python file may byte-match an archived script.
- `records/_template/` — copyable thin entry for the next record.
- `shared/README.md`, `shared/tests/` — usage and behavior tests.
- `.gitignore` — `target/` (coverage output from crapper).

**Not done:** historical records untouched, no dedup. Thin entries are the fork development layout; an upstream PR still needs a self-contained `train_gpt.py` because the competition counts code bytes (documented in `shared/README.md`).

**Checks:**
- `python3 -m unittest discover -s .` → 17 tests OK.
- `python3 shared/check_records.py` → `OK: 37 historical records unchanged; 0 new record(s) use shared.train_core` (exit 0).
- `crapper` on changed files → max CRAP 7.1 (all < 10, coverage 85–100%).
- `dryer` on changed files → no duplicate candidates.
