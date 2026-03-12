#!/usr/bin/env python3
"""Benchmark harness: MuonAdamW correctness and speed.

Three benchmark levels:
  Level 1 — Per-group isolated correctness + speed
  Level 2 — Full optimizer step (PRIMARY TARGET)
  Level 3 — Multi-step correctness against prepare.py baseline

Usage:
    python bench.py [--level 1|2|3|all] [--iters 50]
"""

from __future__ import annotations

import argparse
import os
import sys

import torch
import torch.nn.functional as F

from models import GPT2, get_param_groups, D_MODEL, N_LAYERS, D_FF, VOCAB_SIZE, SEQ_LEN
from optimizer import MuonAdamW, DEFAULT_HYPERS

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
BATCH_SIZE = 8
N_STEPS = 20


def _time_cuda(fn, warmup: int = 5, iters: int = 50) -> float:
    """Time a function using CUDA events. Returns median ms."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    times: list[float] = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))

    times.sort()
    return times[len(times) // 2]


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

        # Set grads and step
        for p, g in zip(params_ours, grads):
            p.grad = g.clone()
        for p, g in zip(params_ref, grads):
            p.grad = g.clone()

        opt_ours.step()
        opt_ref.step()

        max_diff = max(
            (po.float() - pr.float()).abs().max().item()
            for po, pr in zip(params_ours, params_ref)
        )
        correct = max_diff < 0.03
        all_correct &= correct

        # Timing
        def _step_ours():
            for p, g in zip(params_ours, grads):
                p.grad = g
            opt_ours.step()

        def _step_ref():
            for p, g in zip(params_ref, grads):
                p.grad = g
            opt_ref.step()

        ours_ms = _time_cuda(_step_ours, iters=iters)
        ref_ms = _time_cuda(_step_ref, iters=iters)

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
    opt_ref_adamw = torch.optim.AdamW(
        [{"params": params_ref_adamw, "lr": 1e-3, "weight_decay": 0.01, "betas": (0.9, 0.999)}]
    )

    for p, g in zip(params_ours_adamw, grads_adamw):
        p.grad = g.clone()
    for p, g in zip(params_ref_adamw, grads_adamw):
        p.grad = g.clone()

    opt_ours_adamw.step()
    opt_ref_adamw.step()

    max_diff_adamw = max(
        (po.float() - pr.float()).abs().max().item()
        for po, pr in zip(params_ours_adamw, params_ref_adamw)
    )
    correct_adamw = max_diff_adamw < 0.03
    all_correct &= correct_adamw
    print(f"  non_2d: {'PASS' if correct_adamw else 'FAIL'}  max_diff={max_diff_adamw:.4e}")

    return {
        "level": 1,
        "correctness": "PASS" if all_correct else "FAIL",
    }


# ---------------------------------------------------------------------------
# Level 2: Full optimizer step (PRIMARY TARGET)
# ---------------------------------------------------------------------------

def bench_level2(iters: int = 50) -> dict[str, object]:
    """Benchmark full optimizer .step() on GPT2 model."""
    print("\n=== Level 2: Full Optimizer Step ===")

    # Create two identical models
    torch.manual_seed(SEED)
    model_ours = GPT2().to(dtype=torch.bfloat16, device=DEVICE)
    torch.manual_seed(SEED)
    model_ref = GPT2().to(dtype=torch.bfloat16, device=DEVICE)

    # Our optimizer
    opt_ours = MuonAdamW(get_param_groups(model_ours))

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

    # Step once for correctness
    opt_ours.step()
    ref_muon.step()
    ref_adamw.step()

    # Check correctness
    max_diff = 0.0
    for (n1, p1), (n2, p2) in zip(
        model_ours.named_parameters(), model_ref.named_parameters()
    ):
        diff = (p1.float() - p2.float()).abs().max().item()
        max_diff = max(max_diff, diff)

    correct = max_diff < 0.05
    print(f"  Correctness: {'PASS' if correct else 'FAIL'}  max_diff={max_diff:.4e}")

    # Speed: our optimizer
    def _ours_step():
        # Set synthetic grads
        for p in model_ours.parameters():
            p.grad = torch.randn_like(p)
        opt_ours.step()

    def _ref_step():
        for p in model_ref.parameters():
            p.grad = torch.randn_like(p)
        ref_muon.step()
        ref_adamw.step()

    ours_ms = _time_cuda(_ours_step, iters=iters)
    ref_ms = _time_cuda(_ref_step, iters=iters)
    speedup = ref_ms / ours_ms if ours_ms > 0 else float("inf")

    print(f"  Our MuonAdamW:   {ours_ms:.3f}ms")
    print(f"  Reference split: {ref_ms:.3f}ms")
    print(f"  Speedup: {speedup:.2f}x")

    return {
        "level": 2,
        "correctness": "PASS" if correct else "FAIL",
        "ours_ms": round(ours_ms, 3),
        "ref_ms": round(ref_ms, 3),
        "speedup": round(speedup, 4),
        "max_diff": round(max_diff, 6),
    }


# ---------------------------------------------------------------------------
# Level 3: Multi-step correctness against prepare.py baseline
# ---------------------------------------------------------------------------

def bench_level3() -> dict[str, object]:
    """Multi-step: verify training matches prepare.py reference."""
    print("\n=== Level 3: Multi-Step Correctness (vs prepare.py baseline) ===")

    ref_weights_path = os.path.join(DATA_DIR, "ref_weights.pt")
    if not os.path.exists(ref_weights_path):
        print("  SKIP: data/ref_weights.pt not found. Run prepare.py first.")
        return {"level": 3, "correctness": "SKIP"}

    # Recreate training exactly as prepare.py does
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    model = GPT2().to(dtype=torch.bfloat16, device=DEVICE)
    param_groups = get_param_groups(model)
    opt = MuonAdamW(param_groups)

    # Same batch generation as prepare.py
    torch.manual_seed(SEED + 1000)
    all_inputs = [
        torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN), device=DEVICE)
        for _ in range(N_STEPS)
    ]

    for step in range(N_STEPS):
        input_ids = all_inputs[step]
        targets = input_ids[:, 1:]

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

    # Compare against saved reference
    ref_weights = torch.load(ref_weights_path, map_location=DEVICE, weights_only=True)

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
    print(f"  Overall: {'PASS' if correct else 'FAIL'}  max_diff={max_diff:.4e}")

    return {
        "level": 3,
        "correctness": "PASS" if correct else "FAIL",
        "max_diff": round(max_diff, 6),
        "n_steps": N_STEPS,
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
                    baseline_ms = float(f.read().strip())
                speedup = baseline_ms / ours_ms if ours_ms > 0 else float("inf")
                print(f"\nspeedup_vs_baseline: {speedup:.4f}x ({baseline_ms}ms -> {ours_ms}ms)")

    print("=" * 50)


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark MuonAdamW optimizer")
    parser.add_argument("--level", default="all", help="1, 2, 3, or all")
    parser.add_argument("--iters", type=int, default=50, help="Timing iterations")
    args = parser.parse_args()

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
    return 0 if all(r["correctness"] in ("PASS", "SKIP") for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
