#!/usr/bin/env python3
"""Benchmark harness: full PeakSetSIGReg training step correctness and speed.

Three benchmark levels:
  Level 1 — Component sanity (forward, backward, grad clip, optimizer, EMA)
  Level 2 — Full training step speed + correctness (PRIMARY TARGET)
  Level 3 — Multi-step correctness (20 steps)

Usage:
    python sl_bench.py [--level 1|2|3|all] [--iters 20]

This file is READ-ONLY — do not modify.
"""

from __future__ import annotations

import argparse
import dis
import math
import os
import sys
import time as _time_mod

import torch

# ---- Pre-import snapshot of trusted functions ----
_TRUSTED_RANDN_LIKE = torch.randn_like
_TRUSTED_MODULE_PARAMETERS = torch.nn.Module.parameters
_TRUSTED_EVENT_ELAPSED_TIME = torch.cuda.Event.elapsed_time

from sl_constants import (
    AUTOCAST_DTYPE,
    BATCH_SIZE,
    GRAD_CLIP_NORM,
    LEARNING_RATE,
    N_STEPS,
    NUM_PEAKS,
    SEED,
    TEACHER_EMA_UPDATE_EVERY,
)
from sl_model import (
    build_ours_model,
    build_ours_optimizer,
    build_ref_model,
    build_ref_optimizers,
    generate_batches,
    print_param_group_summary,
)

# ---- Import optimizer (may monkey-patch things) ----
from optimizer import MuonAdamW  # noqa: F401

# ---- Post-import restore ----
torch.randn_like = _TRUSTED_RANDN_LIKE
torch.nn.Module.parameters = _TRUSTED_MODULE_PARAMETERS
torch.cuda.Event.elapsed_time = _TRUSTED_EVENT_ELAPSED_TIME

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sl_data")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---- Guard infrastructure ----
_GUARD_FAILURES: list[str] = []


def _guard_fail(msg: str) -> None:
    _GUARD_FAILURES.append(msg)
    print(f"  [GUARD FAIL] {msg}")


def _check_guards_passed() -> bool:
    return len(_GUARD_FAILURES) == 0


def _guard_no_monkey_patches() -> None:
    if torch.randn_like is not _TRUSTED_RANDN_LIKE:
        _guard_fail("torch.randn_like has been monkey-patched")
        torch.randn_like = _TRUSTED_RANDN_LIKE
    if torch.nn.Module.parameters is not _TRUSTED_MODULE_PARAMETERS:
        _guard_fail("torch.nn.Module.parameters has been monkey-patched")
        torch.nn.Module.parameters = _TRUSTED_MODULE_PARAMETERS
    if torch.cuda.Event.elapsed_time is not _TRUSTED_EVENT_ELAPSED_TIME:
        _guard_fail("torch.cuda.Event.elapsed_time has been monkey-patched")
        torch.cuda.Event.elapsed_time = _TRUSTED_EVENT_ELAPSED_TIME


def _guard_timing_sane(ms: float, label: str = "") -> None:
    if not math.isfinite(ms):
        _guard_fail(f"timing {label}: non-finite value {ms}")
    elif ms <= 0.001:
        _guard_fail(f"timing {label}: suspiciously small {ms:.6f}ms")


def _guard_params_changed(
    before: list[torch.Tensor], after_params, label: str = ""
) -> None:
    max_change = 0.0
    n_unchanged = 0
    for b, p in zip(before, after_params):
        diff = (b.float() - p.float()).abs().max().item()
        max_change = max(max_change, diff)
        if diff < 1e-6:
            n_unchanged += 1
    if max_change < 1e-6:
        _guard_fail(
            f"params_changed {label}: no parameter changed (max_change={max_change:.2e})"
        )
    elif n_unchanged > len(before) * 0.5:
        _guard_fail(
            f"params_changed {label}: {n_unchanged}/{len(before)} params unchanged"
        )


def _guard_step_is_real(opt) -> None:
    if "step" in opt.__dict__:
        _guard_fail("optimizer has instance-level 'step' override")
        del opt.__dict__["step"]
    step_method = type(opt).step
    actual_fn = getattr(step_method, "__wrapped__", step_method)
    if hasattr(actual_fn, "__func__"):
        actual_fn = actual_fn.__func__
    if hasattr(actual_fn, "__code__"):
        bytecode = list(dis.get_instructions(actual_fn.__code__))
        if len(bytecode) >= 1:
            first = bytecode[0]
            if first.opname == "RETURN_CONST" and first.argval is None:
                _guard_fail("step() starts with RETURN_CONST None")
            elif (
                len(bytecode) >= 2
                and first.opname == "LOAD_CONST"
                and first.argval is None
                and bytecode[1].opname == "RETURN_VALUE"
            ):
                _guard_fail("step() starts with LOAD_CONST None + RETURN_VALUE")
        if len(bytecode) < 5:
            _guard_fail(
                f"step() has only {len(bytecode)} bytecode instructions"
            )


# ---- Hardened timing ----


def _time_cuda(fn, warmup: int = 5, iters: int = 20) -> float:
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
        if not math.isfinite(t) or t <= 0:
            t = float("inf")
        times.append(t)
    wall_end = _time_mod.perf_counter()
    wall_total_ms = (wall_end - wall_start) * 1000.0

    times.sort()
    median_ms = times[len(times) // 2]

    if math.isfinite(median_ms) and median_ms > 0:
        expected_total = median_ms * iters
        ratio = wall_total_ms / expected_total if expected_total > 0 else float("inf")
        if ratio < 0.1 or ratio > 10.0:
            _guard_fail(
                f"timing cross-check: wall={wall_total_ms:.1f}ms vs "
                f"cuda_median*iters={expected_total:.1f}ms (ratio={ratio:.2f})"
            )

    return median_ms


# ---------------------------------------------------------------------------
# Reference training step (inline, no optimizer import)
# ---------------------------------------------------------------------------

def _ref_train_step(model, batch, optimizers, autocast_dtype, grad_clip_norm):
    """Reference training step using torch.optim.Muon + AdamW."""
    model.advance_sigreg_lambda_schedule()
    with torch.autocast("cuda", dtype=autocast_dtype):
        metrics = model.forward_augmented(batch)
    metrics["loss"].backward()
    if grad_clip_norm and grad_clip_norm > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
    for opt in optimizers:
        opt.step()
        opt.zero_grad(set_to_none=True)
    model.update_teacher()
    return metrics


# ---------------------------------------------------------------------------
# Level 1: Component Sanity
# ---------------------------------------------------------------------------

def bench_level1(iters: int = 20) -> dict[str, object]:
    print("\n=== Level 1: Component Sanity ===")

    _guard_no_monkey_patches()

    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    model = build_ours_model(seed=SEED, device=DEVICE)
    optimizer = build_ours_optimizer(model)
    _guard_step_is_real(optimizer)
    torch.randn_like = _TRUSTED_RANDN_LIKE
    torch.nn.Module.parameters = _TRUSTED_MODULE_PARAMETERS

    batches = generate_batches(n_batches=3, seed=SEED, device=DEVICE)
    batch = batches[0]

    all_pass = True

    # 1. Forward: loss is finite, reasonable magnitude
    print("  [1] Forward pass...")
    with torch.autocast("cuda", dtype=AUTOCAST_DTYPE):
        metrics = model.forward_augmented(batch)
    loss = metrics["loss"]
    loss_val = loss.item()
    forward_ok = math.isfinite(loss_val) and 0 < loss_val < 1000
    print(f"      loss={loss_val:.6f}  {'PASS' if forward_ok else 'FAIL'}")
    all_pass &= forward_ok

    # 2. Backward: all trainable params have gradients
    print("  [2] Backward pass...")
    loss.backward()
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    n_with_grad = sum(1 for p in trainable_params if p.grad is not None)
    backward_ok = n_with_grad == len(trainable_params)
    print(
        f"      {n_with_grad}/{len(trainable_params)} params with gradients  "
        f"{'PASS' if backward_ok else 'FAIL'}"
    )
    all_pass &= backward_ok

    # 3. Grad clipping: norm <= max_norm after clip
    print("  [3] Grad clipping...")
    pre_clip_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(), max_norm=GRAD_CLIP_NORM
    ).item()
    post_clip_norms = []
    for p in trainable_params:
        if p.grad is not None:
            post_clip_norms.append(p.grad.float().norm().item())
    total_post = math.sqrt(sum(n ** 2 for n in post_clip_norms))
    clip_ok = total_post <= GRAD_CLIP_NORM * 1.01  # small tolerance
    print(
        f"      pre_clip_norm={pre_clip_norm:.4f}  post_clip_total={total_post:.4f}  "
        f"{'PASS' if clip_ok else 'FAIL'}"
    )
    all_pass &= clip_ok

    # 4. Optimizer: params actually change after step
    print("  [4] Optimizer step...")
    params_before = [p.clone() for p in trainable_params]
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    _guard_params_changed(params_before, trainable_params, "level1_opt_step")
    max_change = max(
        (b.float() - p.float()).abs().max().item()
        for b, p in zip(params_before, trainable_params)
    )
    opt_ok = max_change > 1e-6
    print(f"      max_change={max_change:.6e}  {'PASS' if opt_ok else 'FAIL'}")
    all_pass &= opt_ok

    # 5. EMA: teacher updates only every TEACHER_EMA_UPDATE_EVERY steps
    print("  [5] EMA teacher update cadence...")
    if model.teacher_encoder is not None:
        teacher_before = {
            name: p.clone()
            for name, p in model.teacher_encoder.state_dict().items()
        }
        # Steps 1..9 should NOT update teacher (step 0 already happened in step 4 if
        # update_teacher was called). Reset the counter to test fresh.
        model.teacher_ema_update_step.zero_()
        # Step 0: should update
        model.update_teacher()
        teacher_after_0 = {
            name: p.clone()
            for name, p in model.teacher_encoder.state_dict().items()
        }
        changed_at_0 = any(
            not torch.equal(teacher_before[n], teacher_after_0[n])
            for n in teacher_before
            if "n_averaged" not in n
        )
        # Steps 1..9: should NOT update
        for i in range(1, TEACHER_EMA_UPDATE_EVERY):
            model.update_teacher()
        teacher_after_9 = {
            name: p.clone()
            for name, p in model.teacher_encoder.state_dict().items()
        }
        changed_1_to_9 = any(
            not torch.equal(teacher_after_0[n], teacher_after_9[n])
            for n in teacher_after_0
            if "n_averaged" not in n
        )
        ema_ok = changed_at_0 and not changed_1_to_9
        print(
            f"      changed_at_step_0={changed_at_0}  "
            f"changed_steps_1_to_{TEACHER_EMA_UPDATE_EVERY - 1}={changed_1_to_9}  "
            f"{'PASS' if ema_ok else 'FAIL'}"
        )
    else:
        ema_ok = True
        print("      No teacher encoder, skipping")
    all_pass &= ema_ok

    # 6. Param count sanity
    print("  [6] Parameter counts...")
    print_param_group_summary(model, "ours")

    guards_ok = _check_guards_passed()
    if not guards_ok:
        all_pass = False
    print(f"  Guards: {'PASS' if guards_ok else 'FAIL'}")

    return {
        "level": 1,
        "correctness": "PASS" if all_pass else "FAIL",
        "guards": "PASS" if guards_ok else "FAIL",
    }


# ---------------------------------------------------------------------------
# Level 2: Full Step Speed + Correctness (PRIMARY TARGET)
# ---------------------------------------------------------------------------

def bench_level2(iters: int = 20) -> dict[str, object]:
    print("\n=== Level 2: Full Training Step Speed + Correctness ===")

    _guard_no_monkey_patches()

    # 1. Create two identical models from same seed
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    model_ref = build_ref_model(seed=SEED, device=DEVICE)

    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    model_ours = build_ours_model(seed=SEED, device=DEVICE)

    # Load same initial weights
    ref_sd = model_ref.state_dict()
    model_ours.load_state_dict(ref_sd)

    # 2. Build optimizers
    ref_optimizers = build_ref_optimizers(model_ref, device=DEVICE)
    ours_optimizer = build_ours_optimizer(model_ours)
    _guard_step_is_real(ours_optimizer)
    torch.randn_like = _TRUSTED_RANDN_LIKE
    torch.nn.Module.parameters = _TRUSTED_MODULE_PARAMETERS
    torch.cuda.Event.elapsed_time = _TRUSTED_EVENT_ELAPSED_TIME

    # 3. Generate same batch
    batches = generate_batches(n_batches=max(iters + 10, N_STEPS + 5), seed=SEED, device=DEVICE)
    batch = batches[0]

    # 4. Save pre-step snapshots
    params_before_ref = [
        p.clone() for p in model_ref.parameters() if p.requires_grad
    ]
    params_before_ours = [
        p.clone() for p in model_ours.parameters() if p.requires_grad
    ]

    # 5. Run one step each for correctness
    torch.manual_seed(SEED + 3000)
    torch.cuda.manual_seed(SEED + 3000)
    ref_metrics = _ref_train_step(
        model_ref, batch, ref_optimizers, AUTOCAST_DTYPE, GRAD_CLIP_NORM
    )

    from sl_train_step import train_step

    torch.manual_seed(SEED + 3000)
    torch.cuda.manual_seed(SEED + 3000)
    ours_metrics = train_step(
        model_ours, batch, ours_optimizer, AUTOCAST_DTYPE, GRAD_CLIP_NORM
    )

    # 6. Guard: params changed
    trainable_ref = [p for p in model_ref.parameters() if p.requires_grad]
    trainable_ours = [p for p in model_ours.parameters() if p.requires_grad]
    _guard_params_changed(params_before_ref, trainable_ref, "level2_ref")
    _guard_params_changed(params_before_ours, trainable_ours, "level2_ours")

    # 7. Compare trainable params
    max_diff = 0.0
    n_compared = 0
    for p_ours, p_ref in zip(trainable_ours, trainable_ref):
        diff = (p_ours.float() - p_ref.float()).abs().max().item()
        max_diff = max(max_diff, diff)
        n_compared += 1
    params_correct = max_diff < 0.05
    print(
        f"  Trainable params: max_diff={max_diff:.4e}  "
        f"({n_compared} params)  {'PASS' if params_correct else 'FAIL'}"
    )

    # 8. Compare teacher params
    teacher_max_diff = 0.0
    if model_ours.teacher_encoder is not None and model_ref.teacher_encoder is not None:
        ours_teacher_sd = model_ours.teacher_encoder.state_dict()
        ref_teacher_sd = model_ref.teacher_encoder.state_dict()
        for key in ref_teacher_sd:
            if ref_teacher_sd[key].is_floating_point():
                diff = (
                    ours_teacher_sd[key].float() - ref_teacher_sd[key].float()
                ).abs().max().item()
                teacher_max_diff = max(teacher_max_diff, diff)
    teacher_correct = teacher_max_diff < 0.01
    print(
        f"  Teacher params:   max_diff={teacher_max_diff:.4e}  "
        f"{'PASS' if teacher_correct else 'FAIL'}"
    )

    # 9. Compare loss
    ref_loss = ref_metrics["loss"].item()
    ours_loss = ours_metrics["loss"].item()
    loss_rel_diff = abs(ref_loss - ours_loss) / max(abs(ref_loss), 1e-8)
    loss_correct = loss_rel_diff < 1e-3
    print(
        f"  Loss: ref={ref_loss:.6f}  ours={ours_loss:.6f}  "
        f"rel_diff={loss_rel_diff:.2e}  {'PASS' if loss_correct else 'FAIL'}"
    )

    correct = params_correct and teacher_correct and loss_correct

    # 10. Timing
    # Reset models for timing (fresh state)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    model_ref_t = build_ref_model(seed=SEED, device=DEVICE)
    ref_sd_t = model_ref_t.state_dict()

    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    model_ours_t = build_ours_model(seed=SEED, device=DEVICE)
    model_ours_t.load_state_dict(ref_sd_t)

    ref_opts_t = build_ref_optimizers(model_ref_t, device=DEVICE)
    ours_opt_t = build_ours_optimizer(model_ours_t)
    torch.randn_like = _TRUSTED_RANDN_LIKE
    torch.nn.Module.parameters = _TRUSTED_MODULE_PARAMETERS
    torch.cuda.Event.elapsed_time = _TRUSTED_EVENT_ELAPSED_TIME

    batch_idx = [0]

    def _ref_step_fn():
        b = batches[batch_idx[0] % len(batches)]
        batch_idx[0] += 1
        _ref_train_step(model_ref_t, b, ref_opts_t, AUTOCAST_DTYPE, GRAD_CLIP_NORM)

    def _ours_step_fn():
        b = batches[batch_idx[0] % len(batches)]
        batch_idx[0] += 1
        train_step(model_ours_t, b, ours_opt_t, AUTOCAST_DTYPE, GRAD_CLIP_NORM)

    batch_idx[0] = 0
    ours_ms = _time_cuda(_ours_step_fn, warmup=5, iters=iters)
    batch_idx[0] = 0
    ref_ms = _time_cuda(_ref_step_fn, warmup=5, iters=iters)

    _guard_timing_sane(ours_ms, "level2_ours")
    _guard_timing_sane(ref_ms, "level2_ref")

    speedup = ref_ms / ours_ms if ours_ms > 0 else float("inf")
    if speedup > 100:
        _guard_fail(f"speedup={speedup:.2f}x is impossibly high")

    print(f"  Ours:      {ours_ms:.3f}ms")
    print(f"  Reference: {ref_ms:.3f}ms")
    print(f"  Speedup:   {speedup:.2f}x")

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
        "teacher_max_diff": round(teacher_max_diff, 6),
        "loss_rel_diff": round(loss_rel_diff, 8),
        "guards": "PASS" if guards_ok else "FAIL",
    }


# ---------------------------------------------------------------------------
# Level 3: Multi-Step Correctness (20 steps)
# ---------------------------------------------------------------------------

def _generate_trusted_reference_multistep(initial_sd, batches):
    """Run N_STEPS with reference optimizers. Returns final state."""
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    model = build_ref_model(seed=SEED, device=DEVICE)
    model.load_state_dict(initial_sd)

    ref_optimizers = build_ref_optimizers(model, device=DEVICE)

    for step in range(N_STEPS):
        torch.manual_seed(SEED + 3000 + step)
        torch.cuda.manual_seed(SEED + 3000 + step)
        _ref_train_step(
            model, batches[step], ref_optimizers, AUTOCAST_DTYPE, GRAD_CLIP_NORM
        )

    trainable = {
        name: param.clone()
        for name, param in model.named_parameters()
        if param.requires_grad
    }
    teacher = (
        {name: p.clone() for name, p in model.teacher_encoder.state_dict().items()}
        if model.teacher_encoder is not None
        else {}
    )
    return trainable, teacher


def bench_level3() -> dict[str, object]:
    print("\n=== Level 3: Multi-Step Correctness ===")

    _guard_no_monkey_patches()

    # 1. Create identical models
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    model_ref = build_ref_model(seed=SEED, device=DEVICE)
    initial_sd = model_ref.state_dict()

    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    model_ours = build_ours_model(seed=SEED, device=DEVICE)
    model_ours.load_state_dict(initial_sd)

    # 2. Build optimizers
    ref_optimizers = build_ref_optimizers(model_ref, device=DEVICE)
    ours_optimizer = build_ours_optimizer(model_ours)
    _guard_step_is_real(ours_optimizer)
    torch.randn_like = _TRUSTED_RANDN_LIKE
    torch.nn.Module.parameters = _TRUSTED_MODULE_PARAMETERS
    torch.cuda.Event.elapsed_time = _TRUSTED_EVENT_ELAPSED_TIME

    # 3. Generate batches
    batches = generate_batches(n_batches=N_STEPS, seed=SEED, device=DEVICE)

    # 4. Run N_STEPS with both
    from sl_train_step import train_step

    params_before = [
        p.clone() for p in model_ours.parameters() if p.requires_grad
    ]

    for step in range(N_STEPS):
        torch.manual_seed(SEED + 3000 + step)
        torch.cuda.manual_seed(SEED + 3000 + step)
        _ref_train_step(
            model_ref, batches[step], ref_optimizers, AUTOCAST_DTYPE, GRAD_CLIP_NORM
        )

        torch.manual_seed(SEED + 3000 + step)
        torch.cuda.manual_seed(SEED + 3000 + step)
        train_step(
            model_ours, batches[step], ours_optimizer, AUTOCAST_DTYPE, GRAD_CLIP_NORM
        )

        if step == 0:
            trainable_ours = [
                p for p in model_ours.parameters() if p.requires_grad
            ]
            _guard_params_changed(params_before, trainable_ours, "level3_first_step")

    # 5. Compare final trainable params
    max_diff = 0.0
    worst_param = ""
    trainable_ref = {
        name: p for name, p in model_ref.named_parameters() if p.requires_grad
    }
    trainable_ours_dict = {
        name: p for name, p in model_ours.named_parameters() if p.requires_grad
    }
    diffs = {}
    for name in trainable_ref:
        diff = (
            trainable_ours_dict[name].float() - trainable_ref[name].float()
        ).abs().max().item()
        diffs[name] = diff
        if diff > max_diff:
            max_diff = diff
            worst_param = name

    params_correct = max_diff < 0.1
    print(f"  After {N_STEPS} steps:")
    sorted_diffs = sorted(diffs.items(), key=lambda x: x[1], reverse=True)
    for name, diff in sorted_diffs[:5]:
        print(f"    {name}: max_diff={diff:.4e}")
    print(
        f"  Trainable params: max_diff={max_diff:.4e}  "
        f"{'PASS' if params_correct else 'FAIL'}"
    )

    # 6. Compare teacher params
    teacher_max_diff = 0.0
    if model_ours.teacher_encoder is not None and model_ref.teacher_encoder is not None:
        ours_teacher_sd = model_ours.teacher_encoder.state_dict()
        ref_teacher_sd = model_ref.teacher_encoder.state_dict()
        for key in ref_teacher_sd:
            if ref_teacher_sd[key].is_floating_point():
                diff = (
                    ours_teacher_sd[key].float() - ref_teacher_sd[key].float()
                ).abs().max().item()
                teacher_max_diff = max(teacher_max_diff, diff)
    teacher_correct = teacher_max_diff < 0.05
    print(
        f"  Teacher params:   max_diff={teacher_max_diff:.4e}  "
        f"{'PASS' if teacher_correct else 'FAIL'}"
    )

    correct = params_correct and teacher_correct

    guards_ok = _check_guards_passed()
    if not guards_ok:
        correct = False
    print(f"  Overall: {'PASS' if correct else 'FAIL'}")
    print(f"  Guards: {'PASS' if guards_ok else 'FAIL'}")

    return {
        "level": 3,
        "correctness": "PASS" if correct else "FAIL",
        "max_diff": round(max_diff, 6),
        "teacher_max_diff": round(teacher_max_diff, 6),
        "n_steps": N_STEPS,
        "guards": "PASS" if guards_ok else "FAIL",
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def print_summary(results: list[dict[str, object]]) -> None:
    print("\n" + "=" * 60)
    print("  SL BENCHMARK SUMMARY")
    print("=" * 60)
    all_pass = True
    for r in results:
        level = r["level"]
        status = r["correctness"]
        if status != "SKIP":
            all_pass &= status == "PASS"
        extras = {k: v for k, v in r.items() if k not in ("level", "correctness")}
        print(f"level_{level}_correctness: {status}")
        for k, v in extras.items():
            print(f"level_{level}_{k}: {v}")

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
                        baseline_ms = ours_ms
                except ValueError:
                    baseline_ms = ours_ms
                speedup = baseline_ms / ours_ms if ours_ms > 0 else float("inf")
                print(
                    f"\nspeedup_vs_baseline: {speedup:.4f}x "
                    f"({baseline_ms}ms -> {ours_ms}ms)"
                )

    print("=" * 60)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark PeakSetSIGReg full training step"
    )
    parser.add_argument("--level", default="all", help="1, 2, 3, or all")
    parser.add_argument("--iters", type=int, default=20, help="Timing iterations")
    args = parser.parse_args()

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

    guards_failed = any(r.get("guards") == "FAIL" for r in results)
    correctness_failed = any(
        r["correctness"] not in ("PASS", "SKIP") for r in results
    )
    return 1 if (guards_failed or correctness_failed) else 0


if __name__ == "__main__":
    sys.exit(main())
