#!/usr/bin/env python3
"""Generate reference artifacts for correctness testing.

Creates data/ with:
  - ref_grads.pt: gradients before first optimizer step
  - ref_weights.pt: model weights after 20 training steps
  - ref_optim_state.pt: optimizer state_dict after 20 steps
"""

from __future__ import annotations

import os

import torch
import torch.nn.functional as F

from models import GPT2, get_param_groups
from optimizer import MuonAdamW

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
N_STEPS = 20
BATCH_SIZE = 8
SEQ_LEN = 128
VOCAB_SIZE = 512
SEED = 42


def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    # Deterministic
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Create model in bf16
    model = GPT2()
    model = model.to(dtype=torch.bfloat16, device="cuda")

    # Build optimizer
    param_groups = get_param_groups(model)
    opt = MuonAdamW(param_groups)

    # Generate all batches deterministically upfront
    torch.manual_seed(SEED + 1000)
    all_inputs = [
        torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN), device="cuda")
        for _ in range(N_STEPS)
    ]

    for step in range(N_STEPS):
        input_ids = all_inputs[step]
        targets = input_ids[:, 1:]  # shifted targets

        # Forward/backward with autocast
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(input_ids)
            logits = logits[:, :-1, :].contiguous()  # shift logits
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.reshape(-1),
            )

        loss.backward()

        # Save gradients before first step
        if step == 0:
            ref_grads = {
                name: param.grad.clone()
                for name, param in model.named_parameters()
                if param.grad is not None
            }
            torch.save(ref_grads, os.path.join(DATA_DIR, "ref_grads.pt"))
            print(f"Saved ref_grads.pt ({len(ref_grads)} tensors)")

        opt.step()
        opt.zero_grad()

        if step % 5 == 0 or step == N_STEPS - 1:
            print(f"Step {step}: loss={loss.item():.4f}")

    # Save final weights
    ref_weights = {name: param.clone() for name, param in model.named_parameters()}
    torch.save(ref_weights, os.path.join(DATA_DIR, "ref_weights.pt"))
    print(f"Saved ref_weights.pt ({len(ref_weights)} tensors)")

    # Save optimizer state
    torch.save(opt.state_dict(), os.path.join(DATA_DIR, "ref_optim_state.pt"))
    print("Saved ref_optim_state.pt")

    # Print model stats
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel: {total_params:,} parameters ({total_params/1e6:.1f}M)")
    for group in param_groups:
        n = sum(p.numel() for p in group["params"])
        print(f"  {group['name']}: {len(group['params'])} tensors, {n:,} params")


if __name__ == "__main__":
    main()
