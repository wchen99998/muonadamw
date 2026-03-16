"""Optimizer hyperparameters and constants.

This file is READ-ONLY — the agent must not modify it.
All values here define the ground-truth algorithm configuration
used by both the optimizer and the benchmark harness.
"""

# Default hyperparameters per group
DEFAULT_HYPERS = {
    "attn_2d": {
        "optimizer": "muon",
        "lr": 3e-4,
        "momentum": 0.95,
        "weight_decay": 0.01,
        "nesterov": True,
    },
    "ffn_2d": {
        "optimizer": "muon",
        "lr": 1e-4,
        "momentum": 0.90,
        "weight_decay": 0.05,
        "nesterov": True,
    },
    "non_2d": {
        "optimizer": "adamw",
        "lr": 1e-3,
        "weight_decay": 0.01,
        "betas": (0.9, 0.999),
    },
}

# Muon algorithm constants
MUON_EPS = 1e-7
MUON_NS_COEFFICIENTS = (3.4445, -4.7750, 2.0315)
MUON_NS_STEPS = 5
MUON_A, MUON_B, MUON_C = MUON_NS_COEFFICIENTS

# Triton kernel tuning defaults
MUON_MOMENTUM_BLOCK_SIZE = 1024
MUON_MOMENTUM_NUM_WARPS = 8
MUON_WEIGHT_UPDATE_BLOCK_M = 32
MUON_WEIGHT_UPDATE_BLOCK_N = 64
MUON_WEIGHT_UPDATE_NUM_WARPS = 4

# AdamW constants
ADAMW_EPS = 1e-8
