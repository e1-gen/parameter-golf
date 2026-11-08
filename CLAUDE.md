# parameter-golf (fork for submissions)

Fork of **openai/parameter-golf** — OpenAI's Model Craft Challenge to train the best byte-level LM fitting in a **16MB artifact** in **≤10 minutes** on **8×H100 SXM**, scored on FineWeb validation bits-per-byte. This fork holds our submission branches and tracks upstream `main`. Research/experimentation happens in the sister repo **butbutt42/Parameter_Golf-** (Frankenstein pipeline).

## Status (updated 2026-04-18)
- **Shipped:** JEPA Hybrid submission as **PR #1685 OPEN** at openai/parameter-golf
  - Branch: `jepa-hybrid-submission` (commit `fd8ae23` — "Non-record: JEPA Hybrid — first latent-prediction LM in Parameter Golf")
  - Path: `records/track_non_record_16mb/2026-04-17_JEPA_Hybrid_7.5MB/` (README.md + submission.json + train_gpt.py + train.log)
  - **BPB:** 1.7579 pre-quant / **1.7622** int8+zlib roundtrip
  - **Artifact:** 7,488,519 bytes (**7.5MB** / 16MB cap)
  - **Training:** 5,908 steps in 600s on 8×H100 SXM, FLAT 0.2GB memory, peak 15.76GB
  - **First JEPA in the competition** — OpenAI explicitly requested JEPA in the README wishlist
  - 6 crashes debugged before success — root-cause analysis documented in the submission README
- **Track:** `track_non_record_16mb` (non-record because we haven't beaten SOTA 0.8265, but JEPA is a novel architecture direction OpenAI asked for)
- `main` tracks upstream openai/parameter-golf at `75700cb` (Merge PR #1511 — April leaderboard update)
- **Competition deadline:** 2026-04-30

## Competition constraints (hard limits — DO NOT violate)
From upstream README:
- **16,000,000 bytes (decimal 16MB, not 16MiB)** — total of code bytes (in `train_gpt.py`) + compressed model weights
- **10 minutes wall-clock** on 8×H100 SXM for training (record track)
- **10 minutes wall-clock** for evaluation (in addition to the training 10 min)
- No external downloads, training dataset access, or network calls during eval — artifact must be self-contained
- Validation = fixed first-50k-document FineWeb subset; **cannot TTT on un-evaluated validation tokens**
- Non-record track accepts unique/creative submissions that don't hit SOTA (where our JEPA lives)
- Submissions must beat existing SOTA by ≥0.005 nats at p<0.01 to become new record

## Tech Stack (our JEPA submission)
- **Architecture:** Byte-level decoder-only LM with **alternating JEPA-latent-MSE + AR cross-entropy** training (parity step = JEPA, odd step = AR — avoids dual forward memory overhead)
- **Config:** 4 effective layers with 2 unique blocks reused (paired weight sharing), d_model=768, grouped-query attention (12 Q heads + 4 KV heads), top-1 MoE with 2 experts, tied embedding + LM head
- **Training:** Muon optimizer for matrix weights + AdamW for rest, ternary STE enabled after warmup, packed ternary export for matrix tensors
- **Tokenizer:** sp8192 (`sproos/parameter-golf-tokenizers` HF repo — SP8192 is community-supplied, sp1024 is the upstream default)
- **Critical hardening (learned from 6 crashes):**
  - `torch.no_grad()` instead of `torch.inference_mode()` (inference_mode tensors poison torch.compile cache → backward crash)
  - `torch.compile` **DISABLED** for JEPA path (recompile after val eval triggered crash)
  - `val_loss_every=99999` (skip periodic val, eval only at end — every val recompile was a crash trigger)
  - `torch.cuda.empty_cache()` every 100 steps
  - GPU memory logging every 200 steps (`alloc/reserved GB` — flat reading confirms no leak)
  - DDP: **do NOT pass `device_id=device` to `dist.init_process_group()`** — call `torch.cuda.set_device(device)` separately

## Hardware architecture
**Training (ad-hoc rental, pay-per-run):**
- **8×H100 SXM** on RunPod ($23.92/hr grant account, ~$20/hr community) — used for actual submission
- **1×H100 SXM** on RunPod ($2.69-$2.99/hr) — smoke tests (~$0.50 per 10-min run)
- **2×H100 SXM** — required DDP debug step before 8×H100 (skipping this cost $5.33 on Crash #1)
- **Successful submission pod:** `ebsrx9g59fa255` @ `103.207.149.68:12890`, personal account, 50GB container disk
- **Template:** OpenAI-provided RunPod template `y5cejece4j` (pre-installed FlashAttention-3, correct torch) — generic pytorch images have FA3 version mismatches

**Research/analytics (persistent servers, NOT in this fork — lives in Parameter_Golf-):**
- VPS / hh-compute-2 (`83.138.53.68`) — Frankenstein pipeline driver
- GPU3 — Qdrant (`arxiv_robotics` 2.66M papers, `parameter_golf_submissions` 1,596 PRs, `parameter_golf_discord` 6,741 msgs, `parameter_golf_special` curated)
- GPU1 — bge-m3 embed server
- SSH tunnels on VPS: `localhost:26333 → GPU3:6333` (Qdrant), `localhost:28003 → GPU1:8003` (bge)
- Research pipeline generated the JEPA hypothesis, verified it passes compatibility/size/time critique, then produced `wishlist_jepa/train_gpt.py`

**Spend so far:** ~$25 OpenAI grant + ~$30 personal on RunPod, plus ~$5 on OpenRouter research calls. $500 grant extension applied.

## Key Files
- `README.md` — upstream OpenAI competition spec (leaderboard, rules, getting-started, FAQ)
- `train_gpt.py` — upstream baseline CUDA training script (~48KB, 1.2 BPB baseline on sp1024)
- `train_gpt_mlx.py` — upstream Apple Silicon MLX variant for local iteration
- `data/cached_challenge_fineweb.py` — dataset downloader (sp1024 variant default, our runs use sp8192)
- `data/tokenizer_specs.json` — defines sp1024/sp8192/byte variants
- `records/track_10min_16mb/` — 27 SOTA record submissions (current leader: 0.8265 BPB)
- `records/track_non_record_16mb/` — non-record / unlimited-compute submissions (our JEPA goes here)
- `THIRD_PARTY_NOTICES.md` — adapted from `modded-nanogpt`
- **Our submission on branch `jepa-hybrid-submission`:** `records/track_non_record_16mb/2026-04-17_JEPA_Hybrid_7.5MB/` (README.md, submission.json, train_gpt.py, train.log)

## Deploy / Run

```bash
# Keep fork main in sync with upstream openai/parameter-golf:
git remote add upstream https://github.com/openai/parameter-golf.git  # one-time
git fetch upstream && git checkout main && git merge upstream/main && git push origin main

# Reproduce JEPA Hybrid submission (on fresh RunPod 8×H100, OpenAI template y5cejece4j):
git clone https://github.com/butbutt42/parameter-golf.git
cd parameter-golf && git checkout jepa-hybrid-submission
python3 -c "from huggingface_hub import snapshot_download; \
  snapshot_download('sproos/parameter-golf-tokenizers', \
  allow_patterns=['datasets/fineweb10B_sp8192/*','tokenizers/fineweb_8192*'], \
  local_dir='./data')"
cd records/track_non_record_16mb/2026-04-17_JEPA_Hybrid_7.5MB
torchrun --standalone --nproc_per_node=8 train_gpt.py

# After run: IMMEDIATELY scp artifact + logs back, THEN delete pod (billing clock keeps running on stopped pods!)
scp -P <port> root@<ip>:/workspace/.../submission.int8.ptz ./
scp -P <port> root@<ip>:/workspace/.../train.log ./
```

## Current state / TODO
- PR #1685 awaiting review at openai/parameter-golf
- `main` is clean relative to upstream — no custom changes to baseline `train_gpt.py`
- **Pending grant extension** ($500 requested) — if approved, 4 more wishlist submissions ready in sister repo (Diffusion BPB 2.79, Megakernel crash, H-net, Adapters BPB 2.89)
- **Upstream wishlist items still unclaimed:** Text Diffusion, H-net tokenization, Megakernels, Learning adapters on random linear maps (our scripts for all 4 exist in Parameter_Golf-/runs/wishlist_*)
- Record-track entry not yet attempted — SOTA 0.8265 BPB on 10-min track needs further hypothesis runs (Plan A: SP1024+SLOT-48+L-BFGS+TTT stack, expected 0.20-0.24 BPB)

## Critical rules for AI agents
- **New records never copy `train_gpt.py`** (2026-11-08): start from `records/_template/` and import `shared.train_core`; historical entries are frozen by `shared/check_records.py`. Run `python3 shared/check_records.py` before committing a record.
- **ALWAYS pass `allow_patterns` to `snapshot_download`** — 2026-04-17 lost $3 because the 20GB container disk filled when `allow_patterns` was missing (sp1024 + full repo pulled alongside sp8192)
- **Minimum 50GB container disk** on every RunPod pod — smaller will OOM the disk on dataset download
- **10-minute wall-clock is hard** — L-BFGS SLOT and full N-gram Cache hypotheses were rejected because eval timed out (+250s) or artifact overflowed (+1.25MB). Verify size/time at compatibility-critique stage BEFORE spending compute
- **30-60s smoke tests DO NOT catch memory leaks or torch.compile×inference_mode crashes** — crashes #1-4 all manifested at steps 2000-4000. Run ≥10 min on 1×H100 and 2×H100 before 8×H100
- **Personal vs grant accounts on RunPod are SEPARATE** — grant credits cannot cover personal pods and vice versa. Check which account a pod is on before launch
- **NEVER use `kill -9 $(pgrep python3)`** in run scripts — it kills the bash wrapper and stops the remaining runs. Use concrete PID or `pkill -f train_gpt` and let processes exit naturally (`runpod_manual.sh` does this right)
- **DDP ladder: 1 → 2 → 8 GPUs.** First `torchrun --nproc_per_node=1` passes (wrapper OK), then `--nproc_per_node=2` passes (communication OK), THEN `--nproc_per_node=8`. Crash #1 skipped step 2 and burned $5.33
- **Real `model.parameters()` count, not formulas** — Adapters miscalc showed 11M, real was 22.8M (12.2MB artifact, 46MB for the buggy 32L config)
- **Use OpenAI RunPod template `y5cejece4j`** — FA3 version matches. Generic pytorch images have FA3 mismatches and waste setup time
- **DELETE pods after extracting artifacts** — stopped pods still bill for storage. Stop ≠ delete. Lost $2 on Crash #1 to idle billing
- **Community H100 pools are often empty** — use `cloudType: SECURE` for reliable availability at $2.99/hr premium
- **Do NOT commit to `main`** for submissions — always use per-submission feature branches (e.g. `jepa-hybrid-submission`) so upstream merges stay clean

## Related
- **Upstream:** https://github.com/openai/parameter-golf (fetched from `upstream` or re-merged into `main`)
- **Sister research repo:** https://github.com/butbutt42/Parameter_Golf- (Frankenstein pipeline, hypothesis cards, WORKLOG — `CLAUDE.md` there has full research context)
- **PICO sister project:** hypothesis gen for finance with €5 budget constraint — same Frankenstein engine reused
- **OpenAI compute grant form:** https://openai.com/index/parameter-golf/#credit-form
- **Community Discord:** #parameter-golf-discussions, #parameter-golf-announcements (OpenAI Discord server)
- **SP8192 tokenizer + dataset source:** `sproos/parameter-golf-tokenizers` on HuggingFace (community supplied; sp1024 only in upstream)
- **Master context:** `/home/dev/CLAUDE.md`, `/home/dev/.claude/projects/-home-dev/memory/MEMORY.md`
- **Our PR #1685:** https://github.com/openai/parameter-golf/pull/1685
