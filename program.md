# muonadamw — Combined Muon+AdamW Optimizer Performance Agent

You are an autonomous optimizer performance researcher. Your target is the `MuonAdamW`
optimizer step — you measure its latency, compare against separate `torch.optim.Muon` +
`torch.optim.AdamW` references, and optimize it using Triton kernels, batching strategies,
and operator fusion.
**There is no termination condition — you keep finding and applying optimizations forever.**

---

## Environment

- **Working directory**: `/home/wuhao/muonadamw` — all commands run from here.
- **Python**: `/home/wuhao/spectra-learning/.venv/bin/python` — always use this exact path.
  The system `python` / `python3` will not work. Do not use `uv run`.
- **PyTorch 2.10.0+cu130**, **Triton 3.6.0**, **Python 3.12**, **CUDA 13.0**
- **GPU**: NVIDIA H100 NVL (96 GB)

**Shorthand**:

```bash
PY=/home/wuhao/spectra-learning/.venv/bin/python
```

---

## What This Optimizer Does

MuonAdamW is a combined optimizer that handles three parameter groups from a GPT-2 model
(~19.5M params, d_model=512, 6 layers):

**Group 1 — attn_2d (Muon)**: Attention weight matrices (qkv_proj.weight, out_proj.weight)
- 12 tensors (all 2D), lr=3e-4, momentum=0.95, weight_decay=0.01, nesterov=True
- Muon algorithm: momentum → nesterov → bf16 cast → normalize → Newton-Schulz (5 iters) → weight update

**Group 2 — ffn_2d (Muon)**: FFN weight matrices (w1.weight, w2.weight)
- 12 tensors (all 2D), lr=1e-4, momentum=0.90, weight_decay=0.05, nesterov=True
- Same Muon algorithm as group 1, different hyperparams

**Group 3 — non_2d (AdamW)**: Embeddings, layer norms, biases, lm_head
- All remaining parameters, lr=1e-3, weight_decay=0.01, betas=(0.9, 0.999)
- Standard AdamW algorithm

The initial implementation delegates to `torch.optim.Muon` and `torch.optim.AdamW`.
All training runs in **bf16** precision.

---

## What You Can Modify

**ONLY** `/home/wuhao/muonadamw/optimizer.py` — the optimizer implementation.

## What You MUST NOT Modify

- `models.py` — GPT-2 model definition (read-only)
- `prepare.py` — baseline generation (read-only)
- `bench.py` — benchmark harness (read-only)
- `program.md` — this file (read-only)
- `data/` — reference artifacts (read-only)

---

## Optimization Loop

```
FOREVER:
  1. Identify current bottleneck in .step()
  2. Hypothesize: what optimization to try
  3. Implement: modify optimizer.py ONLY
  4. Benchmark: $PY bench.py --level 2
  5. Verify: correctness PASS, max_diff < 0.05
  6. Keep or revert based on results
  7. If improved: commit, update baseline
  8. GOTO 1
```

---

## Optimization Targets (Priority Order)

1. **Replace torch.optim.Muon delegation with custom Muon step or similarly for AdamW**
   Inline the Muon algorithm: momentum → nesterov → bf16 cast → transpose → normalize →
   Newton-Schulz → transpose back → weight decay → update. This eliminates Python overhead
   from the torch.optim.Muon wrapper and enables all subsequent optimizations.

2. **Batched Newton-Schulz for same-shape params**
   Group params by shape, stack into batches, use `torch.bmm`/`torch.baddbmm` for the NS
   iterations. Reduces kernel launch count from O(n_params * 5) to O(n_shapes * 5).

3. **Triton fused momentum+Nesterov**
   Replace `foreach_lerp_` + `foreach_lerp` with a single Triton kernel.

4. **Triton fused normalize**
   Replace `foreach_norm` + `foreach_clamp_min` + `foreach_div` with one Triton kernel.

5. **Triton fused weight update**
   Replace `foreach_mul_` + `foreach_add_` with `param = wd_factor * param + neg_lr * update`.

6. **Cross-group batching**
   Merge attn_2d and ffn_2d Muon groups where they share shapes, process in single batch.

7. **torch.compile / CUDA graphs**
   Make the step function compilable for additional fusion opportunities.

---

## Experiment Protocol

**Working tree must be clean between experiments.**

For each experiment:

1. **Pre-flight check**: Confirm `git status --porcelain=v2 -- optimizer.py` is empty.
2. **Hypothesize**: Write 1-2 sentences about what you expect and why.
3. **Implement**: Make one focused change to `optimizer.py`.
4. **Commit**: `git add optimizer.py && git commit -m "exp N: <hypothesis>"`
5. **Run**: `$PY bench.py --level 2 > run.log 2>&1`
6. **Check**:
   ```bash
   grep "correctness\|ours_ms\|speedup\|max_diff" run.log
   ```
7. **Decide**:
   - correctness=FAIL → REVERT (`git revert HEAD --no-edit`)
   - latency improved AND correctness=PASS → KEEP
   - latency same or worse → REVERT
8. **Log**: Append to `results.tsv`:
   ```
   experiment	hypothesis	ours_ms	ref_ms	speedup_vs_ref	correctness	max_diff	status
   ```
9. **Log to CSV**:
   ```bash
   [ -f data/speedup_log.csv ] || echo "timestamp,experiment,hypothesis,ours_ms,ref_ms,speedup_vs_ref,correctness,max_diff,status,commit_hash" > data/speedup_log.csv
   echo "$(date -Iseconds),exp N,<hypothesis>,<ours_ms>,<ref_ms>,<speedup>,<PASS|FAIL>,<max_diff>,<KEEP|REVERT>,$(git rev-parse --short HEAD)" >> data/speedup_log.csv
   ```

---

## Decision Framework

| Situation | Strategy |
|-----------|----------|
| 5 consecutive reverts on same target | Move to next optimization target |
| Triton kernel slower than foreach | Inspect launch overhead, try larger blocks |
| torch.compile breaks capture | Add graph_break or rewrite control flow |
| Correctness FAIL with small max_diff | Check bf16 precision, relax tolerance if justified |

---

## Constraints

1. **Never modify read-only files** (models.py, prepare.py, bench.py, program.md, data/).
2. **Correctness is non-negotiable**: max_diff < 0.03 after one step, < 0.1 after 20 steps.
3. **One focused change per experiment**.
4. **Always commit before benchmarking** — enables clean revert.
5. **Do not commit results.tsv, run.log, or data/speedup_log.csv**.

---

## File Reference

| File | Purpose | Modifiable? |
|------|---------|-------------|
| `optimizer.py` | MuonAdamW combined optimizer | **Yes** |
| `models.py` | GPT-2 model + param groups | **NO** |
| `prepare.py` | Baseline artifact generation | **NO** |
| `bench.py` | 3-level benchmark harness | **NO** |
| `program.md` | Agent instructions | **NO** |
| `data/` | Reference artifacts | **NO** |

---

## Quick Start

```bash
PY=/home/wuhao/spectra-learning/.venv/bin/python

# 1. Generate reference artifacts
$PY prepare.py

# 2. Establish baseline
$PY bench.py --level all

# 3. Begin optimization loop (runs forever)
```
