# sl_ — Full Training Step Optimization for PeakSetSIGReg

You are an autonomous training-step performance researcher. Your target is the **full training
step** of a PeakSetSIGReg model (small config): forward, backward, grad clipping, optimizer,
scheduler, and EMA — all as one self-contained benchmark target.
**There is no termination condition — you keep finding and applying optimizations forever.**

---

## Environment

- **Working directory**: `~/muonadamw` — all commands run from here.
- **Python**: `~/muonadamw/.venv/bin/python` — always use this exact path.
  The system `python` / `python3` will not work. Do not use `uv run`.
- **PyTorch 2.10.0+cu130**, **Triton 3.6.0**, **Python 3.12**, **CUDA 13.0**
- **GPU**: NVIDIA H100 NVL (96 GB)

**Shorthand**:

```bash
PY=~/muonadamw/.venv/bin/python
```

---

## What This Benchmark Measures

A complete PeakSetSIGReg training step (small config: dim=512, 12 layers, 8 heads):

1. `model.advance_sigreg_lambda_schedule()`
2. Forward: `model.forward_augmented(batch)` — 3 encoder calls (context + 2 target blocks) + teacher encoder + predictor + loss
3. `loss.backward()`
4. `clip_grad_norm_` with max_norm=1.0
5. `optimizer.step()` (MuonAdamW)
6. `optimizer.zero_grad(set_to_none=True)`
7. `model.update_teacher()` (EMA update every 10 steps)

Batch: 256 samples × 64 peaks, bf16 precision.

---

## What You Can Modify

**TWO files** — both are optimization targets:

1. **`sl_model_opt.py`** — The model architecture, forward pass, loss computation, EMA update.
   Optimize with custom Triton kernels, fused ops, batched encoder calls, etc.

2. **`sl_train_step.py`** — Training step orchestration, optimizer integration.
   Optimize with torch.compile, CUDA graphs, compute/memory overlap, etc.

## What You MUST NOT Modify

- `sl_constants.py` — hyperparameters (read-only)
- `sl_model.py` — reference model + param grouping + batch gen (read-only)
- `sl_prepare.py` — baseline generation (read-only)
- `sl_bench.py` — benchmark harness (read-only)
- `sl_program.md` — this file (read-only)
- `sl_data/` — reference artifacts (read-only)
- `optimizer.py` — MuonAdamW optimizer (used as-is)
- `constants.py`, `models.py`, `bench.py`, etc. — other project files (read-only)

---

## Optimization Loop

```
FOREVER:
  1. Identify current bottleneck (profile if needed)
  2. Hypothesize: what optimization to try and which file to modify
  3. Implement: modify sl_model_opt.py and/or sl_train_step.py
  4. Commit: git add sl_model_opt.py sl_train_step.py && git commit -m "sl exp N: <hypothesis>"
  5. Benchmark: $PY sl_bench.py --level 2
  6. Verify: correctness PASS, max_diff < 0.05
  7. Keep or revert based on results
  8. If improved: update baseline
  9. GOTO 1
```

---

## Current Direction: Custom Triton Kernels

The attention pattern in this model is unique — non-causal, symmetric visibility-mask
attention (AND of two boolean vectors) with very short sequences (64 tokens) and
head dim (64). This makes it a poor fit for general-purpose flex_attention and a great
candidate for hand-written Triton kernels.

**Primary focus: write custom Triton kernels rather than relying on torch.compile tuning.**
torch.compile gains have plateaued (~26.5 ms). The next level of performance requires
custom GPU code.

### Priority: Custom Triton Attention (forward + backward)
- The entire seq_len=64 × head_dim=32 tile fits in SRAM — no multi-block tiling needed.
- Fuse the visibility mask (simple bool AND) directly into the kernel instead of
  materializing a BlockMask.
- Fuse RoPE application into the attention kernel (compute sin/cos inline or load from
  a small buffer, apply before QK dot product).
- Write both the forward and backward (dQ, dK, dV) kernels so autograd uses them
  end-to-end — don't rely on torch.compile to generate the backward.
- Consider a fused QKV-proj → RoPE → attention → output-proj megakernel if register
  pressure allows.

### Priority: Megakernel / Fused Transformer Block
- Consider fusing LayerNorm + QKV projection + RoPE + attention + output projection
  into a single Triton kernel (forward + backward).
- Consider fusing FFN norm + SiLU-gated FFN (w1, w2) into a single kernel.
- The small dimensions (dim=512, hidden=1366, heads=8, head_dim=64) mean most of these
  fit comfortably in shared memory / registers — exploit this.
- Training runs under bf16 autocast — all Triton kernels should operate in bfloat16
  (inputs, outputs, and accumulation where precision allows). Use fp32 accumulation
  only where numerically necessary (e.g., softmax reduction, loss computation).

### Priority: Sparse-Aware GEMMs
- The visibility mask means many tokens are invalid (padding) — effective sequence
  lengths are 32–64 out of 64 slots. The QKV projections and FFN matmuls waste FLOPs
  on padded positions.
- Consider gathering only valid tokens before GEMMs and scattering back after, or
  writing Triton GEMM kernels that skip padded rows entirely.
- With batched encoder calls (B*(K+1) views), the padding waste is amplified — a
  sparse/compressed layout could yield significant savings.
- Alternatively, a block-sparse GEMM approach where blocks corresponding to invalid
  tokens are skipped at the tile level.

### Secondary
- Fused L2 loss + masking kernel
- Triton EMA update kernel
- Fused predictor path (same attention pattern, 4 layers)

### Step-level (sl_train_step.py)
- Already well-optimized via torch.compile reduce-overhead.
- Consider CUDA graphs for the compiled region if Triton kernels enable it.
- Memory/compute overlap (optimizer with teacher forward, EMA with next batch).

### Pending: Sync full-view teacher encoding (upstream commit 2a24523)
- Upstream spectra-learning changed teacher target computation: instead of K separate
  forward passes with per-target-block `visible_mask=target_masks` (via `repeat_interleave`),
  the teacher now sees the **full valid spectrum once** (`visible_mask=peak_valid_mask`) and
  the result is expanded to K views via `unsqueeze(1).expand(-1, K, -1, -1)`.
- The reference model (imported from spectra-learning) already has this change — `sl_prepare.py`
  regenerated reference artifacts reflect the new semantics.
- `sl_model_opt.py` still uses the old per-target-block approach in both `compute_teacher_targets`
  and the `elif self.teacher_encoder` / `else` branches of `forward_augmented` (~lines 1008-1083).
- **Must update**: `compute_teacher_targets`, `forward_augmented` teacher branch, and
  `forward_augmented` else branch to match upstream full-view encoding.
- This is also a performance win: 1 forward pass instead of K (=2), no `repeat_interleave`.
- **Caveat**: `compute_teacher_targets` is called via `_CUDAGraphRunner` in `sl_train_step.py`.
  The output shape changes from materialized `[B, K, N, D]` (reshape) to an expanded view —
  ensure CUDA graph compatibility and/or torch compile compatibility (may need `.contiguous()`).
- **Opportunity**: With the full-view change eliminating `repeat_interleave` and reducing teacher
  to a single forward pass, a full `torch.compile` of the entire step (teacher + student + loss)
  with `max-autotune` may now discover better fusion opportunities that were previously blocked
  by the K-way expansion. Worth re-evaluating full-step compile after syncing.
- **Config synced**: `sl_constants.py` now matches upstream `gems_a_masked_latent_index_small.py`
  (model_dim=512, teacher_ema_decay=1.0, teacher_ema_decay_start=0.996,
  teacher_ema_decay_warmup_steps=400_000). Reference artifacts regenerated.

### What has already been tried and plateaued
- torch.compile mode variations (reduce-overhead, max-autotune, max-autotune-no-cudagraphs)
- compile options (coordinate_descent_tuning, aggressive_fusion, max_autotune_gemm)
- fullgraph=True on various subgraphs
- SDPA as a flex_attention replacement (failed correctness or regressed)
- Caching/hoisting minor Python-level overhead (negligible at this point)

---

## Experiment Protocol

**Working tree must be clean between experiments.**

For each experiment:

1. **Pre-flight**: Confirm `git status --porcelain=v2 -- sl_model_opt.py sl_train_step.py` is clean.
2. **Hypothesize**: Write 1-2 sentences about what you expect and why.
3. **Implement**: Make one focused change to `sl_model_opt.py` and/or `sl_train_step.py`.
4. **Commit**: `git add sl_model_opt.py sl_train_step.py && git commit -m "sl exp N: <hypothesis>"`
5. **Run**: `$PY sl_bench.py --level 2 > sl_run.log 2>&1`
6. **Check**:
   ```bash
   grep "correctness\|ours_ms\|speedup\|max_diff\|teacher_max_diff" sl_run.log
   ```
7. **Decide**:
   - correctness=FAIL → REVERT (`git revert HEAD --no-edit`)
   - latency improved AND correctness=PASS → KEEP
   - latency same or worse → REVERT
8. **Log**: Append to `sl_results.tsv`:
   ```
   experiment	hypothesis	ours_ms	ref_ms	speedup_vs_ref	correctness	max_diff	status
   ```

---

## Correctness Thresholds

| Metric | Level 2 (1 step) | Level 3 (20 steps) |
|--------|------------------|--------------------|
| Trainable params max_diff | < 0.05 | < 0.1 |
| Teacher params max_diff | < 0.01 | < 0.05 |
| Loss relative diff | < 1e-3 | N/A |
| Params actually changed | yes | yes |

---

## Decision Framework

| Situation | Strategy |
|-----------|----------|
| 5 consecutive reverts on same target | Move to next optimization target |
| Triton kernel slower than reference | Inspect launch overhead, try larger blocks |
| torch.compile breaks capture | Add graph_break or rewrite control flow |
| Correctness FAIL with small max_diff | Check bf16 precision, relax tolerance if justified |
| Model change breaks state_dict compat | Ensure same parameter names/shapes as reference |

---

## Constraints

1. **Never modify read-only files**.
2. **Correctness is non-negotiable**: optimized step must produce the same parameter updates
   as the reference (torch.optim.Muon + AdamW with matching hyperparameters).
3. **State dict compatibility**: the optimized model must have the same state_dict structure
   as the reference model from spectra-learning.
4. **One focused change per experiment**.
5. **Always commit before benchmarking** — enables clean revert.

---

## Prohibited Techniques

1. **No monkey-patching** — Do not replace any function outside the two modifiable files.
2. **No no-op steps** — Every step must perform actual parameter updates.
3. **No benchmark interference** — Do not modify timing, gradient generation, or benchmark behavior.
4. **No reducing algorithm fidelity** — All mathematical operations must be equivalent.

---

## Quick Start

```bash
PY=~/.venv/bin/python

# 1. Generate reference artifacts
$PY sl_prepare.py

# 2. Establish baseline
$PY sl_bench.py --level all

# 3. Begin optimization loop (runs forever)
```

---

## File Reference

| File | Purpose | Modifiable? |
|------|---------|-------------|
| `sl_model_opt.py` | Optimizable model (forward, loss, EMA) | **YES** |
| `sl_train_step.py` | Optimizable training step | **YES** |
| `sl_constants.py` | Small-config hyperparameters | NO |
| `sl_model.py` | Reference model + param groups + batch gen | NO |
| `sl_prepare.py` | Baseline artifact generation | NO |
| `sl_bench.py` | 3-level benchmark harness | NO |
| `sl_program.md` | Agent instructions (this file) | NO |
| `sl_data/` | Reference artifacts | NO |
| `optimizer.py` | MuonAdamW optimizer (used as-is) | NO |
