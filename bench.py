#!/usr/bin/env python3
"""Benchmark harness: MuonAdamW correctness and speed.

Three benchmark levels:
  Level 1 — Per-group isolated correctness + speed
  Level 2 — Full optimizer step (PRIMARY TARGET)
  Level 3 — Multi-step correctness against trusted reference

Usage:
    python bench.py [--level 1|2|3|all] [--iters 50]
"""

from __future__ import annotations

import argparse
import dis
import math
import os
import sys
import time as _time_mod

import torch
import torch.nn.functional as F

# ---- Step 1: Pre-import snapshot of trusted functions ----
_TRUSTED_RANDN_LIKE = torch.randn_like
_TRUSTED_MODULE_PARAMETERS = torch.nn.Module.parameters
_TRUSTED_EVENT_ELAPSED_TIME = torch.cuda.Event.elapsed_time

from models import GPT2, get_param_groups, D_MODEL, N_LAYERS, D_FF, VOCAB_SIZE, SEQ_LEN

# ---- Import optimizer (may monkey-patch things) ----
from optimizer import MuonAdamW, DEFAULT_HYPERS

# ---- Step 2: Post-import restore and cleanup ----
torch.randn_like = _TRUSTED_RANDN_LIKE
torch.nn.Module.parameters = _TRUSTED_MODULE_PARAMETERS
torch.cuda.Event.elapsed_time = _TRUSTED_EVENT_ELAPSED_TIME

# Clear adversary's global caches if they exist
import optimizer as _optimizer_module
for _cache_name in (
    "_FAST_RANDN_LIKE_TENSORS",
    "_FILTERED_PARAMETER_IDS",
    "_FILTERED_MODULE_IDS",
):
    _cache = getattr(_optimizer_module, _cache_name, None)
    if _cache is not None and hasattr(_cache, "clear"):
        _cache.clear()
if hasattr(_optimizer_module, "_PENDING_FAKE_STEP_TIMING"):
    _optimizer_module._PENDING_FAKE_STEP_TIMING = False

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
BATCH_SIZE = 8
N_STEPS = 20

# ---- Step 3: Guard infrastructure ----
_GUARD_FAILURES: list[str] = []


def _guard_fail(msg: str) -> None:
    """Record a guard failure."""
    _GUARD_FAILURES.append(msg)
    print(f"  [GUARD FAIL] {msg}")


def _check_guards_passed() -> bool:
    """Return True if no guards have been triggered."""
    return len(_GUARD_FAILURES) == 0


def _guard_no_monkey_patches() -> None:
    """Verify torch functions haven't been monkey-patched."""
    if torch.randn_like is not _TRUSTED_RANDN_LIKE:
        _guard_fail("torch.randn_like has been monkey-patched")
        torch.randn_like = _TRUSTED_RANDN_LIKE
    if torch.nn.Module.parameters is not _TRUSTED_MODULE_PARAMETERS:
        _guard_fail("torch.nn.Module.parameters has been monkey-patched")
        torch.nn.Module.parameters = _TRUSTED_MODULE_PARAMETERS
    if torch.cuda.Event.elapsed_time is not _TRUSTED_EVENT_ELAPSED_TIME:
        _guard_fail("torch.cuda.Event.elapsed_time has been monkey-patched")
        torch.cuda.Event.elapsed_time = _TRUSTED_EVENT_ELAPSED_TIME


def _guard_randn_like_is_random() -> None:
    """Verify torch.randn_like actually produces random outputs."""
    test_t = torch.ones(16, device=DEVICE, dtype=torch.bfloat16)
    r1 = torch.randn_like(test_t)
    r2 = torch.randn_like(test_t)
    # Outputs should differ from each other
    if torch.equal(r1, r2):
        _guard_fail("torch.randn_like returned identical outputs on two calls")
    # Outputs should differ from input
    if torch.equal(r1, test_t):
        _guard_fail("torch.randn_like returned the input tensor unchanged")


def _guard_parameter_count(model, label: str = "") -> None:
    """Verify model.parameters() returns expected count and total numel."""
    params = list(_TRUSTED_MODULE_PARAMETERS(model))
    n_params = len(params)
    total_numel = sum(p.numel() for p in params)
    # GPT2 with our config should have 65 params and ~19.5M numel
    if n_params != 65:
        _guard_fail(f"parameter count {label}: expected 65, got {n_params}")
    if total_numel < 15_000_000 or total_numel > 25_000_000:
        _guard_fail(
            f"parameter numel {label}: expected ~19.5M, got {total_numel:,}"
        )


def _guard_step_is_real(opt) -> None:
    """Verify optimizer.step hasn't been replaced with a no-op."""
    # Check 1: No instance-level override
    if "step" in opt.__dict__:
        _guard_fail(
            "optimizer has instance-level 'step' override (self.step assigned in __init__)"
        )
        # Remove the instance override so the class method is used
        del opt.__dict__["step"]

    # Check 2: Class method bytecode isn't a trivial return None
    step_method = type(opt).step
    # Unwrap decorators if needed
    actual_fn = getattr(step_method, "__wrapped__", step_method)
    if hasattr(actual_fn, "__func__"):
        actual_fn = actual_fn.__func__
    if hasattr(actual_fn, "__code__"):
        code = actual_fn.__code__
        bytecode = list(dis.get_instructions(code))
        # Check if first real instruction is RETURN_CONST None or LOAD_CONST None + RETURN_VALUE
        if len(bytecode) >= 1:
            first = bytecode[0]
            if first.opname == "RETURN_CONST" and first.argval is None:
                _guard_fail("step() bytecode starts with RETURN_CONST None (dead code)")
            elif (
                len(bytecode) >= 2
                and first.opname == "LOAD_CONST"
                and first.argval is None
                and bytecode[1].opname == "RETURN_VALUE"
            ):
                _guard_fail("step() bytecode starts with LOAD_CONST None + RETURN_VALUE (dead code)")
        # Also check if the method is suspiciously short (< 5 instructions)
        if len(bytecode) < 5:
            _guard_fail(f"step() has only {len(bytecode)} bytecode instructions (suspiciously short)")


def _guard_params_changed(before: list[torch.Tensor], after_params, label: str = "") -> None:
    """Verify parameters actually changed after step()."""
    max_change = 0.0
    n_unchanged = 0
    for b, p in zip(before, after_params):
        diff = (b.float() - p.float()).abs().max().item()
        max_change = max(max_change, diff)
        if diff < 1e-6:
            n_unchanged += 1
    if max_change < 1e-6:
        _guard_fail(f"params_changed {label}: no parameter changed (max_change={max_change:.2e})")
    elif n_unchanged > len(before) * 0.5:
        _guard_fail(
            f"params_changed {label}: {n_unchanged}/{len(before)} params unchanged"
        )


def _guard_timing_sane(ms: float, label: str = "") -> None:
    """Verify timing value is finite, positive, and reasonable."""
    if not math.isfinite(ms):
        _guard_fail(f"timing {label}: non-finite value {ms}")
    elif ms <= 0.001:
        _guard_fail(f"timing {label}: suspiciously small {ms:.6f}ms")


# ---- Step 4: Hardened timing function ----

def _time_cuda(fn, warmup: int = 5, iters: int = 50) -> float:
    """Time a function using CUDA events with wall-clock cross-check. Returns median ms."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    wall_start = _time_mod.perf_counter()
    times: list[float] = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        t = _TRUSTED_EVENT_ELAPSED_TIME(start, end)
        # Validate each timing
        if not math.isfinite(t) or t <= 0:
            t = float("inf")
        times.append(t)
    wall_end = _time_mod.perf_counter()
    wall_total_ms = (wall_end - wall_start) * 1000.0

    times.sort()
    median_ms = times[len(times) // 2]

    # Cross-check: median * iters should be roughly comparable to wall time
    if math.isfinite(median_ms) and median_ms > 0:
        expected_total = median_ms * iters
        ratio = wall_total_ms / expected_total if expected_total > 0 else float("inf")
        if ratio < 0.1 or ratio > 10.0:
            _guard_fail(
                f"timing cross-check: wall={wall_total_ms:.1f}ms vs "
                f"cuda_median*iters={expected_total:.1f}ms (ratio={ratio:.2f})"
            )

    return median_ms


def _make_ref_optimizers(model: GPT2):
    """Create separate reference Muon + AdamW optimizers with same hyperparams."""
    param_groups = get_param_groups(model)
    attn_params = param_groups[0]["params"]
    ffn_params = param_groups[1]["params"]
    non_2d_params = param_groups[2]["params"]

    ref_muon = torch.optim.Muon(
        [
            {
                "params": attn_params,
                "lr": DEFAULT_HYPERS["attn_2d"]["lr"],
                "momentum": DEFAULT_HYPERS["attn_2d"]["momentum"],
                "weight_decay": DEFAULT_HYPERS["attn_2d"]["weight_decay"],
                "nesterov": DEFAULT_HYPERS["attn_2d"]["nesterov"],
            },
            {
                "params": ffn_params,
                "lr": DEFAULT_HYPERS["ffn_2d"]["lr"],
                "momentum": DEFAULT_HYPERS["ffn_2d"]["momentum"],
                "weight_decay": DEFAULT_HYPERS["ffn_2d"]["weight_decay"],
                "nesterov": DEFAULT_HYPERS["ffn_2d"]["nesterov"],
            },
        ],
    )

    ref_adamw = torch.optim.AdamW(
        [
            {
                "params": non_2d_params,
                "lr": DEFAULT_HYPERS["non_2d"]["lr"],
                "weight_decay": DEFAULT_HYPERS["non_2d"]["weight_decay"],
                "betas": DEFAULT_HYPERS["non_2d"]["betas"],
            },
        ],
    )

    return ref_muon, ref_adamw


# ---------------------------------------------------------------------------
# Level 1: Per-group isolated correctness + speed
# ---------------------------------------------------------------------------

def bench_level1(iters: int = 50) -> dict[str, object]:
    """Benchmark each optimizer group in isolation."""
    print("\n=== Level 1: Per-Group Isolated Correctness + Speed ===")

    _guard_no_monkey_patches()

    # Shapes for each group
    group_shapes = {
        "attn_2d": [
            (D_MODEL, 3 * D_MODEL),  # qkv_proj.weight
            (D_MODEL, D_MODEL),       # out_proj.weight
        ] * N_LAYERS,
        "ffn_2d": [
            (D_MODEL, D_FF),   # w1.weight
            (D_FF, D_MODEL),   # w2.weight
        ] * N_LAYERS,
    }

    all_correct = True

    for group_name, shapes in group_shapes.items():
        hypers = DEFAULT_HYPERS[group_name]
        torch.manual_seed(SEED)

        # Create params + grads
        params_ours = [torch.randn(s, dtype=torch.bfloat16, device=DEVICE) for s in shapes]
        grads = [torch.randn(s, dtype=torch.bfloat16, device=DEVICE) for s in shapes]
        params_ref = [p.clone() for p in params_ours]

        # Our optimizer (single group)
        opt_ours = MuonAdamW([{
            "params": params_ours,
            "name": group_name,
        }])

        _guard_step_is_real(opt_ours)
        # Force-restore trusted functions after optimizer creation
        torch.randn_like = _TRUSTED_RANDN_LIKE
        torch.nn.Module.parameters = _TRUSTED_MODULE_PARAMETERS
        torch.cuda.Event.elapsed_time = _TRUSTED_EVENT_ELAPSED_TIME

        # Reference Muon
        opt_ref = torch.optim.Muon(
            [{
                "params": params_ref,
                "lr": hypers["lr"],
                "momentum": hypers["momentum"],
                "weight_decay": hypers["weight_decay"],
                "nesterov": hypers["nesterov"],
            }],
        )

        # Save params before step
        params_before = [p.clone() for p in params_ours]

        # Set grads and step
        for p, g in zip(params_ours, grads):
            p.grad = g.clone()
        for p, g in zip(params_ref, grads):
            p.grad = g.clone()

        opt_ours.step()
        opt_ref.step()

        # Guard: params must change
        _guard_params_changed(params_before, params_ours, f"level1_{group_name}")

        max_diff = max(
            (po.float() - pr.float()).abs().max().item()
            for po, pr in zip(params_ours, params_ref)
        )
        correct = max_diff < 0.03
        all_correct &= correct

        # Timing — use trusted functions in closures
        def _step_ours(params_ours=params_ours, grads=grads, opt_ours=opt_ours):
            for p, g in zip(params_ours, grads):
                p.grad = g
            opt_ours.step()

        def _step_ref(params_ref=params_ref, grads=grads, opt_ref=opt_ref):
            for p, g in zip(params_ref, grads):
                p.grad = g
            opt_ref.step()

        ours_ms = _time_cuda(_step_ours, iters=iters)
        ref_ms = _time_cuda(_step_ref, iters=iters)

        _guard_timing_sane(ours_ms, f"level1_{group_name}_ours")

        status = "PASS" if correct else "FAIL"
        print(f"  {group_name}: {status}  max_diff={max_diff:.4e}  ours={ours_ms:.3f}ms  ref={ref_ms:.3f}ms")

    # AdamW group — compare against standalone AdamW
    torch.manual_seed(SEED)
    non_2d_shapes = [
        (VOCAB_SIZE, D_MODEL),  # tok_emb
        (SEQ_LEN, D_MODEL),     # pos_emb
        (D_MODEL,),             # ln weights/biases (simplified)
        (D_MODEL,),
    ]
    params_ours_adamw = [torch.randn(s, dtype=torch.bfloat16, device=DEVICE) for s in non_2d_shapes]
    grads_adamw = [torch.randn(s, dtype=torch.bfloat16, device=DEVICE) for s in non_2d_shapes]
    params_ref_adamw = [p.clone() for p in params_ours_adamw]

    opt_ours_adamw = MuonAdamW([{"params": params_ours_adamw, "name": "non_2d"}])
    _guard_step_is_real(opt_ours_adamw)
    torch.randn_like = _TRUSTED_RANDN_LIKE
    torch.nn.Module.parameters = _TRUSTED_MODULE_PARAMETERS
    torch.cuda.Event.elapsed_time = _TRUSTED_EVENT_ELAPSED_TIME

    opt_ref_adamw = torch.optim.AdamW(
        [{"params": params_ref_adamw, "lr": 1e-3, "weight_decay": 0.01, "betas": (0.9, 0.999)}]
    )

    params_before_adamw = [p.clone() for p in params_ours_adamw]

    for p, g in zip(params_ours_adamw, grads_adamw):
        p.grad = g.clone()
    for p, g in zip(params_ref_adamw, grads_adamw):
        p.grad = g.clone()

    opt_ours_adamw.step()
    opt_ref_adamw.step()

    _guard_params_changed(params_before_adamw, params_ours_adamw, "level1_non_2d")

    max_diff_adamw = max(
        (po.float() - pr.float()).abs().max().item()
        for po, pr in zip(params_ours_adamw, params_ref_adamw)
    )
    correct_adamw = max_diff_adamw < 0.03
    all_correct &= correct_adamw
    print(f"  non_2d: {'PASS' if correct_adamw else 'FAIL'}  max_diff={max_diff_adamw:.4e}")

    # Override correctness if guards failed
    guards_ok = _check_guards_passed()
    if not guards_ok:
        all_correct = False

    return {
        "level": 1,
        "correctness": "PASS" if all_correct else "FAIL",
        "guards": "PASS" if guards_ok else "FAIL",
    }


# ---------------------------------------------------------------------------
# Level 2: Full optimizer step (PRIMARY TARGET)
# ---------------------------------------------------------------------------

def bench_level2(iters: int = 50) -> dict[str, object]:
    """Benchmark full optimizer .step() on GPT2 model."""
    print("\n=== Level 2: Full Optimizer Step ===")

    _guard_no_monkey_patches()
    _guard_randn_like_is_random()

    # Create two identical models
    torch.manual_seed(SEED)
    model_ours = GPT2().to(dtype=torch.bfloat16, device=DEVICE)
    torch.manual_seed(SEED)
    model_ref = GPT2().to(dtype=torch.bfloat16, device=DEVICE)

    # Guard: parameter counts
    _guard_parameter_count(model_ours, "model_ours_pre_opt")
    _guard_parameter_count(model_ref, "model_ref_pre_opt")

    # Our optimizer
    opt_ours = MuonAdamW(get_param_groups(model_ours))

    # Guard: step is real + force-restore
    _guard_step_is_real(opt_ours)
    torch.randn_like = _TRUSTED_RANDN_LIKE
    torch.nn.Module.parameters = _TRUSTED_MODULE_PARAMETERS
    torch.cuda.Event.elapsed_time = _TRUSTED_EVENT_ELAPSED_TIME

    # Re-check parameter counts after optimizer creation (may have been filtered)
    _guard_parameter_count(model_ours, "model_ours_post_opt")
    _guard_parameter_count(model_ref, "model_ref_post_opt")

    # Reference: separate Muon + AdamW
    ref_muon, ref_adamw = _make_ref_optimizers(model_ref)

    # Generate grads via forward/backward
    torch.manual_seed(SEED + 100)
    input_ids = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN), device=DEVICE)
    targets = input_ids[:, 1:]

    # Forward/backward for ours
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits_ours = model_ours(input_ids)
        loss_ours = F.cross_entropy(
            logits_ours[:, :-1].contiguous().view(-1, VOCAB_SIZE),
            targets.reshape(-1),
        )
    loss_ours.backward()

    # Forward/backward for ref (same input)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits_ref = model_ref(input_ids)
        loss_ref = F.cross_entropy(
            logits_ref[:, :-1].contiguous().view(-1, VOCAB_SIZE),
            targets.reshape(-1),
        )
    loss_ref.backward()

    # Guard: verify gradients exist and are non-zero before step
    n_grads_ours = sum(1 for p in _TRUSTED_MODULE_PARAMETERS(model_ours) if p.grad is not None)
    if n_grads_ours == 0:
        _guard_fail("no gradients present before step (model_ours)")
    n_grads_ref = sum(1 for p in _TRUSTED_MODULE_PARAMETERS(model_ref) if p.grad is not None)
    if n_grads_ref == 0:
        _guard_fail("no gradients present before step (model_ref)")

    # Save param snapshots before step
    params_before_ours = [p.clone() for p in _TRUSTED_MODULE_PARAMETERS(model_ours)]
    params_before_ref = [p.clone() for p in _TRUSTED_MODULE_PARAMETERS(model_ref)]

    # Step once for correctness
    opt_ours.step()
    ref_muon.step()
    ref_adamw.step()

    # Guard: params must have changed
    _guard_params_changed(
        params_before_ours,
        list(_TRUSTED_MODULE_PARAMETERS(model_ours)),
        "level2_ours",
    )
    _guard_params_changed(
        params_before_ref,
        list(_TRUSTED_MODULE_PARAMETERS(model_ref)),
        "level2_ref",
    )

    # Guard: update magnitudes should be non-trivial and comparable
    ours_update_mag = max(
        (b.float() - p.float()).abs().max().item()
        for b, p in zip(params_before_ours, _TRUSTED_MODULE_PARAMETERS(model_ours))
    )
    ref_update_mag = max(
        (b.float() - p.float()).abs().max().item()
        for b, p in zip(params_before_ref, _TRUSTED_MODULE_PARAMETERS(model_ref))
    )
    if ours_update_mag > 0 and ref_update_mag > 0:
        update_ratio = ours_update_mag / ref_update_mag
        if update_ratio < 0.01 or update_ratio > 100:
            _guard_fail(
                f"update magnitude ratio ours/ref={update_ratio:.4f} "
                f"(ours={ours_update_mag:.4e}, ref={ref_update_mag:.4e})"
            )

    # Check correctness
    max_diff = 0.0
    n_params_compared = 0
    for (n1, p1), (n2, p2) in zip(
        model_ours.named_parameters(), model_ref.named_parameters()
    ):
        diff = (p1.float() - p2.float()).abs().max().item()
        max_diff = max(max_diff, diff)
        n_params_compared += 1

    # Guard: verify expected parameter count in comparison
    if n_params_compared != 65:
        _guard_fail(f"correctness compared {n_params_compared} params, expected 65")

    correct = max_diff < 0.05
    print(f"  Correctness: {'PASS' if correct else 'FAIL'}  max_diff={max_diff:.4e}")

    # Speed: our optimizer — use trusted functions in closures
    def _ours_step():
        for p in _TRUSTED_MODULE_PARAMETERS(model_ours):
            p.grad = _TRUSTED_RANDN_LIKE(p)
        opt_ours.step()

    def _ref_step():
        for p in _TRUSTED_MODULE_PARAMETERS(model_ref):
            p.grad = _TRUSTED_RANDN_LIKE(p)
        ref_muon.step()
        ref_adamw.step()

    ours_ms = _time_cuda(_ours_step, iters=iters)
    ref_ms = _time_cuda(_ref_step, iters=iters)

    # Guard: timing sanity
    _guard_timing_sane(ours_ms, "level2_ours")
    _guard_timing_sane(ref_ms, "level2_ref")

    speedup = ref_ms / ours_ms if ours_ms > 0 else float("inf")

    # Guard: speedup shouldn't be impossibly high
    if speedup > 100:
        _guard_fail(f"speedup={speedup:.2f}x is impossibly high (>100x)")

    print(f"  Our MuonAdamW:   {ours_ms:.3f}ms")
    print(f"  Reference split: {ref_ms:.3f}ms")
    print(f"  Speedup: {speedup:.2f}x")

    # Override correctness if guards failed
    guards_ok = _check_guards_passed()
    if not guards_ok:
        correct = False
        print(f"  Guards: FAIL ({len(_GUARD_FAILURES)} violations)")
    else:
        print("  Guards: PASS")

    return {
        "level": 2,
        "correctness": "PASS" if correct else "FAIL",
        "ours_ms": round(ours_ms, 3),
        "ref_ms": round(ref_ms, 3),
        "speedup": round(speedup, 4),
        "max_diff": round(max_diff, 6),
        "guards": "PASS" if guards_ok else "FAIL",
    }


# ---------------------------------------------------------------------------
# Level 3: Multi-step correctness against trusted reference
# ---------------------------------------------------------------------------

def _generate_trusted_reference(model_state_dict: dict) -> dict[str, torch.Tensor]:
    """Generate reference weights using torch.optim.Muon + torch.optim.AdamW directly.

    This bypasses MuonAdamW entirely, eliminating the Category 7 vulnerability
    where prepare.py imports the potentially compromised optimizer.
    """
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    model = GPT2().to(dtype=torch.bfloat16, device=DEVICE)
    # Load the same initial weights
    model.load_state_dict(model_state_dict)

    param_groups = get_param_groups(model)
    attn_params = param_groups[0]["params"]
    ffn_params = param_groups[1]["params"]
    non_2d_params = param_groups[2]["params"]

    ref_muon = torch.optim.Muon(
        [
            {
                "params": attn_params,
                "lr": DEFAULT_HYPERS["attn_2d"]["lr"],
                "momentum": DEFAULT_HYPERS["attn_2d"]["momentum"],
                "weight_decay": DEFAULT_HYPERS["attn_2d"]["weight_decay"],
                "nesterov": DEFAULT_HYPERS["attn_2d"]["nesterov"],
            },
            {
                "params": ffn_params,
                "lr": DEFAULT_HYPERS["ffn_2d"]["lr"],
                "momentum": DEFAULT_HYPERS["ffn_2d"]["momentum"],
                "weight_decay": DEFAULT_HYPERS["ffn_2d"]["weight_decay"],
                "nesterov": DEFAULT_HYPERS["ffn_2d"]["nesterov"],
            },
        ],
    )
    ref_adamw = torch.optim.AdamW(
        [
            {
                "params": non_2d_params,
                "lr": DEFAULT_HYPERS["non_2d"]["lr"],
                "weight_decay": DEFAULT_HYPERS["non_2d"]["weight_decay"],
                "betas": DEFAULT_HYPERS["non_2d"]["betas"],
            },
        ],
    )

    torch.manual_seed(SEED + 1000)
    all_inputs = [
        torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN), device=DEVICE)
        for _ in range(N_STEPS)
    ]

    for step_idx in range(N_STEPS):
        input_ids = all_inputs[step_idx]
        targets = input_ids[:, 1:]

        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(input_ids)
            logits = logits[:, :-1, :].contiguous()
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.reshape(-1),
            )
        loss.backward()
        ref_muon.step()
        ref_adamw.step()
        ref_muon.zero_grad()
        ref_adamw.zero_grad()

    return {name: param.clone() for name, param in model.named_parameters()}


def bench_level3() -> dict[str, object]:
    """Multi-step: verify training matches trusted reference (generated independently)."""
    print("\n=== Level 3: Multi-Step Correctness (vs trusted reference) ===")

    _guard_no_monkey_patches()

    # Recreate training exactly as prepare.py does
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    model = GPT2().to(dtype=torch.bfloat16, device=DEVICE)

    # Save initial state for trusted reference generation
    initial_state_dict = {name: param.clone() for name, param in model.named_parameters()}

    _guard_parameter_count(model, "level3_model")

    param_groups = get_param_groups(model)
    opt = MuonAdamW(param_groups)

    _guard_step_is_real(opt)
    # Force-restore after optimizer init
    torch.randn_like = _TRUSTED_RANDN_LIKE
    torch.nn.Module.parameters = _TRUSTED_MODULE_PARAMETERS
    torch.cuda.Event.elapsed_time = _TRUSTED_EVENT_ELAPSED_TIME

    _guard_parameter_count(model, "level3_model_post_opt")

    # Same batch generation as prepare.py
    torch.manual_seed(SEED + 1000)
    all_inputs = [
        torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN), device=DEVICE)
        for _ in range(N_STEPS)
    ]

    for step in range(N_STEPS):
        input_ids = all_inputs[step]
        targets = input_ids[:, 1:]

        # Save params before first step for guard check
        if step == 0:
            params_before_first = [p.clone() for p in _TRUSTED_MODULE_PARAMETERS(model)]

        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(input_ids)
            logits = logits[:, :-1, :].contiguous()
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.reshape(-1),
            )
        loss.backward()
        opt.step()
        opt.zero_grad()

        if step == 0:
            _guard_params_changed(
                params_before_first,
                list(_TRUSTED_MODULE_PARAMETERS(model)),
                "level3_first_step",
            )

    # Generate trusted reference independently using torch.optim.Muon/AdamW
    print("  Generating trusted reference (torch.optim.Muon + AdamW)...")
    ref_weights = _generate_trusted_reference(initial_state_dict)

    max_diffs = {}
    for name, param in model.named_parameters():
        ref = ref_weights[name]
        diff = (param.float() - ref.float()).abs().max().item()
        max_diffs[name] = diff

    max_diff = max(max_diffs.values())
    correct = max_diff < 0.1  # accumulated bf16 drift over 20 steps

    # Show worst params
    sorted_diffs = sorted(max_diffs.items(), key=lambda x: x[1], reverse=True)
    print(f"  After {N_STEPS} steps:")
    for name, diff in sorted_diffs[:5]:
        print(f"    {name}: max_diff={diff:.4e}")

    # Override correctness if guards failed
    guards_ok = _check_guards_passed()
    if not guards_ok:
        correct = False

    print(f"  Overall: {'PASS' if correct else 'FAIL'}  max_diff={max_diff:.4e}")
    print(f"  Guards: {'PASS' if guards_ok else 'FAIL'}")

    return {
        "level": 3,
        "correctness": "PASS" if correct else "FAIL",
        "max_diff": round(max_diff, 6),
        "n_steps": N_STEPS,
        "guards": "PASS" if guards_ok else "FAIL",
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def print_summary(results: list[dict[str, object]]) -> None:
    print("\n" + "=" * 50)
    print("  BENCHMARK SUMMARY")
    print("=" * 50)
    all_pass = True
    for r in results:
        level = r["level"]
        status = r["correctness"]
        if status != "SKIP":
            all_pass &= (status == "PASS")
        extras = {k: v for k, v in r.items() if k not in ("level", "correctness")}
        print(f"level_{level}_correctness: {status}")
        for k, v in extras.items():
            print(f"level_{level}_{k}: {v}")

    # Show guard summary
    guards_status = "PASS"
    for r in results:
        if r.get("guards") == "FAIL":
            guards_status = "FAIL"
            break
    print(f"guards: {guards_status}")
    if _GUARD_FAILURES:
        print(f"guard_violations: {len(_GUARD_FAILURES)}")
        for msg in _GUARD_FAILURES:
            print(f"  - {msg}")

    print(f"overall_correctness: {'PASS' if all_pass else 'FAIL'}")

    # Save/compare baseline
    if any(r["level"] == 2 for r in results):
        l2 = next(r for r in results if r["level"] == 2)
        if "ours_ms" in l2:
            ours_ms = l2["ours_ms"]
            baseline_path = os.path.join(DATA_DIR, "baseline_latency_ms.txt")
            os.makedirs(DATA_DIR, exist_ok=True)
            if not os.path.isfile(baseline_path):
                with open(baseline_path, "w") as f:
                    f.write(str(ours_ms))
                print(f"\nBaseline latency saved: {ours_ms}ms")
            else:
                with open(baseline_path) as f:
                    content = f.read().strip()
                try:
                    baseline_ms = float(content)
                    if not math.isfinite(baseline_ms) or baseline_ms <= 0:
                        _guard_fail(f"baseline_latency_ms.txt contains invalid value: {content}")
                        baseline_ms = ours_ms  # fallback
                except ValueError:
                    _guard_fail(f"baseline_latency_ms.txt contains non-numeric value: {content!r}")
                    baseline_ms = ours_ms  # fallback
                speedup = baseline_ms / ours_ms if ours_ms > 0 else float("inf")
                print(f"\nspeedup_vs_baseline: {speedup:.4f}x ({baseline_ms}ms -> {ours_ms}ms)")

    print("=" * 50)


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark MuonAdamW optimizer")
    parser.add_argument("--level", default="all", help="1, 2, 3, or all")
    parser.add_argument("--iters", type=int, default=50, help="Timing iterations")
    args = parser.parse_args()

    # Clear guard state at start
    _GUARD_FAILURES.clear()

    torch.set_float32_matmul_precision("medium")

    levels = [1, 2, 3] if args.level == "all" else [int(args.level)]
    results: list[dict[str, object]] = []

    if 1 in levels:
        results.append(bench_level1(iters=args.iters))
    if 2 in levels:
        results.append(bench_level2(iters=args.iters))
    if 3 in levels:
        results.append(bench_level3())

    print_summary(results)

    # Non-zero exit if guards failed or correctness failed
    guards_failed = any(r.get("guards") == "FAIL" for r in results)
    correctness_failed = any(r["correctness"] not in ("PASS", "SKIP") for r in results)
    return 1 if (guards_failed or correctness_failed) else 0


if __name__ == "__main__":
    sys.exit(main())
