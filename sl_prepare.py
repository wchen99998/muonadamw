#!/usr/bin/env python3
"""Generate deterministic reference artifacts for sl_bench.py.

This file is READ-ONLY — do not modify.

Outputs in sl_data/:
  - initial_weights.pt      — model state_dict at init
  - ref_final_weights.pt    — trainable params after N_STEPS reference steps
  - ref_final_teacher_weights.pt — teacher params after N_STEPS steps
  - ref_loss_trajectory.pt  — loss per step
  - baseline_latency_ms.txt — reference timing (written by sl_bench.py)
"""

import os
import time

import torch

from sl_constants import (
    AUTOCAST_DTYPE,
    GRAD_CLIP_NORM,
    N_STEPS,
    SEED,
)
from sl_model import (
    build_ref_model,
    build_ref_optimizers,
    generate_batches,
    print_param_group_summary,
)

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sl_data")


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    device = "cuda"

    print("=== sl_prepare: Generating reference artifacts ===")

    # 1. Build reference model
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    model = build_ref_model(seed=SEED, device=device)
    print_param_group_summary(model, "reference")

    # 2. Save initial weights
    initial_sd = {
        name: param.clone().cpu()
        for name, param in model.state_dict().items()
    }
    torch.save(initial_sd, os.path.join(DATA_DIR, "initial_weights.pt"))
    print(f"  Saved initial_weights.pt ({len(initial_sd)} entries)")

    # 3. Build reference optimizers
    ref_optimizers = build_ref_optimizers(model, device=device)

    # 4. Generate batches
    batches = generate_batches(n_batches=N_STEPS, seed=SEED, device=device)
    print(f"  Generated {len(batches)} batches")

    # 5. Run N_STEPS reference training steps
    loss_trajectory = []
    torch.manual_seed(SEED + 2000)
    torch.cuda.manual_seed(SEED + 2000)

    t0 = time.perf_counter()
    for step in range(N_STEPS):
        batch = batches[step]
        model.advance_sigreg_lambda_schedule()
        with torch.autocast("cuda", dtype=AUTOCAST_DTYPE):
            metrics = model.forward_augmented(batch)
        loss = metrics["loss"]
        loss.backward()
        if GRAD_CLIP_NORM and GRAD_CLIP_NORM > 0:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=GRAD_CLIP_NORM
            )
        for opt in ref_optimizers:
            opt.step()
            opt.zero_grad(set_to_none=True)
        model.update_teacher()
        loss_val = loss.detach().item()
        loss_trajectory.append(loss_val)
        if (step + 1) % 5 == 0 or step == 0:
            print(f"  step {step + 1}/{N_STEPS}: loss={loss_val:.6f}")
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    print(f"  {N_STEPS} steps completed in {elapsed:.2f}s")

    # 6. Save final trainable weights
    ref_final = {
        name: param.clone().cpu()
        for name, param in model.named_parameters()
        if param.requires_grad
    }
    torch.save(ref_final, os.path.join(DATA_DIR, "ref_final_weights.pt"))
    print(f"  Saved ref_final_weights.pt ({len(ref_final)} entries)")

    # 7. Save final teacher weights
    if model.teacher_encoder is not None:
        teacher_sd = {
            name: param.clone().cpu()
            for name, param in model.teacher_encoder.state_dict().items()
        }
        torch.save(teacher_sd, os.path.join(DATA_DIR, "ref_final_teacher_weights.pt"))
        print(f"  Saved ref_final_teacher_weights.pt ({len(teacher_sd)} entries)")

    # 8. Save loss trajectory
    torch.save(loss_trajectory, os.path.join(DATA_DIR, "ref_loss_trajectory.pt"))
    print(f"  Saved ref_loss_trajectory.pt ({len(loss_trajectory)} values)")

    print("=== sl_prepare: Done ===")


if __name__ == "__main__":
    main()
