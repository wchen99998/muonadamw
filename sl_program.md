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
2. Teacher: `model.compute_teacher_targets(batch)` — 1 full-view encoder pass (explicit CUDA graph)
3. Student forward: `model.forward_augmented(batch, teacher_targets)` — context encoder (pack_n=19) + predictor (4 layers, pack_n=35) + L2 loss
4. `loss.backward()`
5. `clip_grad_norm_` with max_norm=1.0
6. `optimizer.step()` (MuonAdamW)
7. `optimizer.zero_grad(set_to_none=True)`
8. `model.update_teacher()` (EMA update every 10 steps)

Batch: 256 samples × 64 peaks, bf16 precision. Config: `representation_regularizer="none"`,
so sigreg/gco are inactive — student forward is context encoder + predictor + L2 loss only.

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

## Current Direction: Fused Megakernels for Transformer Layers

**Current performance: ~16.5 ms / step (~30.5× over reference).**
Both torch.compile tuning AND simple custom Triton kernels have been exhausted.
A custom Triton attention kernel (fwd + bwd) is already in place. The next level of
performance requires **fusing entire sub-layers into megakernels** that eliminate
intermediate tensor materializations between operations.

### Profile breakdown (per step, H100 NVL)

| Category | Time | % | Notes |
|----------|------|---|-------|
| cuBLAS GEMMs | 8.3 ms | 55% | QKV proj, output proj, FFN w1/w2 (fwd + bwd + teacher) |
| LayerNorm | 1.33 ms | 9% | Inductor-generated Triton pointwise |
| Triton attention | 1.24 ms | 8% | Custom fwd (23 µs) + bwd (37 µs) per call |
| SiLU activation | 1.2 ms | 8% | Separate kernel after w1 GEMM |
| Misc pointwise | 1.1 ms | 7% | RoPE, masking, residual add, view ops |
| BMM (optimizer NS) | 0.61 ms | 4% | Newton–Schulz iteration in MuonAdamW |
| Optimizer + grad clip | 0.47 ms | 3% | Muon momentum, AdamW, grad norm |
| Other | 0.7 ms | 5% | Memset, concat, graph launches |

Theoretical lower bound ≈ 5 ms (memory-bandwidth). The 3.3× gap is mainly from many
small-kernel launches and intermediate-tensor round-trips through HBM.

### Current architecture

```
sl_train_step.py:
  Teacher: torch.compile(max-autotune-no-cudagraphs) + explicit CUDA graph  → 3.25 ms
  Student: torch.compile(max-autotune, dynamic=False) via CUDAGraphTree     → 6.1 ms fwd, 5.8 ms bwd
  Optimizer: MuonAdamW (own CUDA graphs)                                     → 0.5 ms

sl_model_opt.py:
  Encoder: PeakSetEncoder with sparse prefix-packing
    - Context: pack_n=19, prefix_pack, pad_to=32, 12 layers
    - Teacher: pack_n=64, prefix_pack, pad_to=64, 12 layers (in CUDA graph)
  Predictor: 4 layers, pack_n=35, pad_to=64 (argsort + gather/scatter)
  Attention: Custom Triton fwd + bwd kernels (fused visibility mask, no FlexAttention)
  Loss: Simplified L2 path (sigreg/gco dead code removed for this config)
```

### Priority 1: Fused FFN megakernel (forward + backward)

The FFN sub-layer is: `h + w2(SiLU(w1(LayerNorm(h))))`. Today this is 5 separate kernels
(LayerNorm, w1 GEMM, SiLU, w2 GEMM, residual add) = 2.5 ms combined. A fused kernel
that tiles along the M dimension, keeping the [BLOCK_M, hidden_dim] intermediate in
registers / L2, would collapse this to ≈ 2 kernels (fwd GEMM1+SiLU, fwd GEMM2+residual)
or even 1 kernel if the two GEMMs can be chained through L2.

**Key constraints:**
- Weight matrices (512 × 1368 each = 1.4 MB bf16) don't fit in shared memory (228 KB)
  but DO fit in H100 L2 cache (50 MB). A tiled approach loads weights once from HBM
  per layer; subsequent M-tiles reuse from L2.
- `torch.autograd.Function` subclasses work inside `torch.compile(fullgraph=True)` —
  verified. The custom autograd function will NOT break the CUDA graph. The key is that
  both `forward()` and `backward()` must be traceable. Standard PyTorch ops in the
  backward are fine (torch.compile generates the backward graph automatically).
- The backward of SiLU+w2 is: `dh_silu = dy @ W2`, `dh = dh_silu * silu'(h)`,
  `dW2 = dy^T @ silu(h)`. This requires saving `h` (pre-SiLU w1 output) for backward.
- Triton GEMMs are ~20–30% slower than cuBLAS for these shapes alone. The fusion must
  save enough from eliminating the SiLU kernel + intermediate tensor writes to compensate.
  Exp 17 showed that naively replacing ALL GEMMs with Triton is a 10% regression. Exp 19
  showed that a custom autograd function with an untuned Triton GEMM regressed 15%
  — but exp 19's regression was from the slow GEMM, NOT from a graph break.
- **The approach to beat cuBLAS:** Don't replace cuBLAS; instead, **fuse the prologue**
  (SiLU applied on-the-fly as w2's input is loaded) into a Triton GEMM for w2. This
  saves the separate SiLU kernel (~36 µs/call × 16 layers) and the intermediate tensor
  write+read. Net target: ≈ 0.5–1.0 ms savings.

### Priority 2: Fused attention sub-layer megakernel

The attention sub-layer is: `x + wo(Attn(RoPE(QKV(LayerNorm(x)))))`. Today this is
LayerNorm kernel → cuBLAS QKV GEMM → RoPE+masking kernel → Triton attention kernel →
cuBLAS output GEMM → residual add kernel = 6 kernels.

Fusing RoPE into the existing Triton attention kernel was analyzed and found to be
**marginal** (~100 µs net, <1%) because:
- The separate RoPE kernel also fuses masking/split/view ops (not just RoPE).
- The backward requires inverse-RoPE on dQ/dK, which needs a memory round-trip for
  register-to-register element swapping (Triton can't shuffle within a tile).
- The forward saves are roughly canceled by the backward overhead.

A more impactful approach: fuse LayerNorm + residual-add across layer boundaries.
The end of layer N (`h + ffn_out`) and the start of layer N+1 (`LayerNorm(h_next)`)
touch the same tensor. A persistent kernel that chains layers without writing `h` to
HBM between them would eliminate ~1.33 ms of LayerNorm overhead.

**This requires a persistent-thread Triton kernel** — extremely complex but the highest
theoretical payoff. Each thread block processes a slice of M and iterates over all 12
layers, keeping the activations in L2/registers.

### Priority 3: Stream overlap (teacher ∥ context encoder)

Teacher CUDA graph (3.25 ms) and student context encoder (~2 ms) are independent.
Overlapping them on separate CUDA streams would save ~2 ms.

**Blocked by CUDAGraphTree:** torch.compile's CUDAGraphTree does not allow its managed
tensors to be accessed from another stream's CUDA graph (RuntimeError: "accessing tensor
output of CUDAGraphs that has been overwritten"). Attempts to use `max-autotune-no-cudagraphs`
for the split functions to avoid this conflict cost ~0.7 ms from losing CUDA graphs,
negating the overlap benefit (exp 13: 16.741 ms vs 16.5 ms baseline).

**Possible unblock:** Compile context encoder + predictor+loss as a SINGLE function with
`max-autotune`, and run the teacher explicit CUDA graph on a side stream. The student
function receives teacher_targets AFTER sync. The issue is that teacher_targets is
produced by an explicit CUDA graph (static output buffer) — the CUDAGraphTree of the
student function sees it as an external tensor. This worked in exp 13 for correctness
but the non-cudagraph compile mode was too slow. The key missing piece: a way to run
the student on `max-autotune` (with its own CUDA graph) while accepting the teacher's
static output buffer as input. This may require `torch.compiler.cudagraph_mark_step_begin()`
and `.clone()` of teacher_targets to detach from the teacher graph's storage.

### What has already been tried and plateaued (experiments 1–23)

**torch.compile tuning (plateaued at ~16.5 ms):**
- mode variations: reduce-overhead, max-autotune, max-autotune-no-cudagraphs
- compile options: coordinate_descent_tuning, aggressive_fusion, max_autotune_pointwise
- max_autotune_gemm_backends="TRITON" (10% regression — cuBLAS is faster for these shapes)
- fullgraph=True (no graph breaks exist; no improvement)
- dynamic=False (kept; marginal improvement from explicit static shapes)

**CUDA graph / stream overlap:**
- Explicit CUDA graph for fwd+bwd+clip (compatibility errors)
- Overlap teacher with context encoder on separate stream (lost CUDA graphs → regression)
- reduce-overhead for teacher (CUDAGraphTree conflict with student)
- max-autotune for teacher inside explicit graph (nested CUDA graph error)
- Teacher warmup iterations 3→5 (no effect)

**Model-level:**
- SDPA for teacher attention (regression)
- Reduced predictor pack_n 35→32 (correctness FAIL — drops valid tokens)
- Custom fused SiLU+w2 Triton GEMM (15% regression — Triton GEMM too slow vs cuBLAS;
  graph break confirmed NOT the cause)
- Expanded attention autotune configs (no improvement — autotuner already found optimal)
- Stripped zero metrics + simplified loss path + removed dead attention branches (kept;
  cleaner compiled graph, no measurable speedup)

**Key finding from exp 19:** `torch.autograd.Function` subclasses do NOT break
`torch.compile(fullgraph=True)`. The regression was purely from the Triton GEMM being
slower than cuBLAS. This means custom autograd functions are a viable delivery mechanism
for fused kernels — the GEMM itself just needs to be competitive.

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
