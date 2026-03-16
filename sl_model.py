"""Reference model construction, param grouping, and synthetic batch generation.

This file is READ-ONLY — do not modify.
"""

import sys

import torch

from sl_constants import (
    MODEL_DIM,
    NUM_LAYERS,
    NUM_HEADS,
    NUM_KV_HEADS,
    ATTENTION_MLP_MULTIPLE,
    FEATURE_MLP_HIDDEN_DIM,
    MASKED_LATENT_PREDICTOR_NUM_LAYERS,
    NORM_TYPE,
    ENCODER_USE_ROPE,
    ENCODER_QK_NORM,
    BATCH_SIZE,
    NUM_PEAKS,
    JEPA_NUM_TARGET_BLOCKS,
    CONTEXT_FRACTION,
    TARGET_FRACTION,
    MASKED_TOKEN_LOSS_WEIGHT,
    MASKED_TOKEN_LOSS_TYPE,
    REPRESENTATION_REGULARIZER,
    USE_EMA_TEACHER_TARGET,
    TEACHER_EMA_DECAY,
    TEACHER_EMA_DECAY_START,
    TEACHER_EMA_DECAY_WARMUP_STEPS,
    TEACHER_EMA_UPDATE_EVERY,
    LEARNING_RATE,
    WEIGHT_DECAY,
    B2,
    MUON_MOMENTUM,
    MUON_NESTEROV,
    MUON_NS_STEPS,
    MUON_ADJUST_LR_FN,
    SEED,
    N_STEPS,
    TOTAL_STEPS,
)

# Ensure spectra-learning is importable
_SL_PATH = "/home/wuhao/spectra-learning"
if _SL_PATH not in sys.path:
    sys.path.insert(0, _SL_PATH)


# ---------------------------------------------------------------------------
# Model kwargs shared by ref and ours
# ---------------------------------------------------------------------------

_MODEL_KWARGS = dict(
    model_dim=MODEL_DIM,
    encoder_num_layers=NUM_LAYERS,
    encoder_num_heads=NUM_HEADS,
    encoder_num_kv_heads=NUM_KV_HEADS,
    attention_mlp_multiple=ATTENTION_MLP_MULTIPLE,
    feature_mlp_hidden_dim=FEATURE_MLP_HIDDEN_DIM,
    masked_latent_predictor_num_layers=MASKED_LATENT_PREDICTOR_NUM_LAYERS,
    norm_type=NORM_TYPE,
    encoder_use_rope=ENCODER_USE_ROPE,
    encoder_qk_norm=ENCODER_QK_NORM,
    jepa_num_target_blocks=JEPA_NUM_TARGET_BLOCKS,
    masked_token_loss_weight=MASKED_TOKEN_LOSS_WEIGHT,
    masked_token_loss_type=MASKED_TOKEN_LOSS_TYPE,
    representation_regularizer=REPRESENTATION_REGULARIZER,
    use_ema_teacher_target=USE_EMA_TEACHER_TARGET,
    teacher_ema_decay=TEACHER_EMA_DECAY,
    teacher_ema_decay_start=TEACHER_EMA_DECAY_START,
    teacher_ema_decay_warmup_steps=TEACHER_EMA_DECAY_WARMUP_STEPS,
    teacher_ema_update_every=TEACHER_EMA_UPDATE_EVERY,
)


def _is_weight_decay_target(name: str, param: torch.nn.Parameter) -> bool:
    """Replicated from spectra-learning/train.py:40-48."""
    return (
        param.ndim >= 2
        and name.endswith("weight")
        and any(
            t in name
            for t in ("attention.", "feed_forward.", "cross_attn.")
        )
    )


# ---------------------------------------------------------------------------
# Reference model (from spectra-learning imports)
# ---------------------------------------------------------------------------

def build_ref_model(seed=SEED, device="cuda"):
    """Create a PeakSetSIGReg model using spectra-learning code."""
    from models.model import PeakSetSIGReg as RefPeakSetSIGReg

    torch.manual_seed(seed)
    model = RefPeakSetSIGReg(**_MODEL_KWARGS)
    model.to(dtype=torch.bfloat16, device=device).train()
    return model


# ---------------------------------------------------------------------------
# "Ours" model (from sl_model_opt.py)
# ---------------------------------------------------------------------------

def build_ours_model(seed=SEED, device="cuda"):
    """Create a PeakSetSIGReg model using the optimizable model code."""
    from sl_model_opt import PeakSetSIGReg as OptPeakSetSIGReg

    torch.manual_seed(seed)
    model = OptPeakSetSIGReg(**_MODEL_KWARGS)
    model.to(dtype=torch.bfloat16, device=device).train()
    return model


# ---------------------------------------------------------------------------
# Reference optimizers (torch.optim.Muon + AdamW)
# ---------------------------------------------------------------------------

def build_ref_optimizers(model, device="cuda"):
    """Create reference optimizers: torch.optim.Muon + torch.optim.AdamW.

    Replicates spectra-learning/train.py:_build_optimizers for muon mode.
    """
    device = torch.device(device)
    muon_decay_params, muon_no_decay_params, adamw_params = [], [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim == 2:
            if _is_weight_decay_target(name, param):
                muon_decay_params.append(param)
            else:
                muon_no_decay_params.append(param)
        else:
            adamw_params.append(param)

    muon_param_groups = []
    if muon_decay_params:
        muon_param_groups.append(
            {"params": muon_decay_params, "weight_decay": WEIGHT_DECAY}
        )
    if muon_no_decay_params:
        muon_param_groups.append(
            {"params": muon_no_decay_params, "weight_decay": 0.0}
        )

    muon_opt = torch.optim.Muon(
        muon_param_groups,
        lr=torch.tensor(LEARNING_RATE, device=device),
        momentum=MUON_MOMENTUM,
        nesterov=MUON_NESTEROV,
        ns_steps=MUON_NS_STEPS,
        weight_decay=0.0,
        adjust_lr_fn=MUON_ADJUST_LR_FN,
    )
    adamw_opt = torch.optim.AdamW(
        adamw_params,
        lr=torch.tensor(LEARNING_RATE, device=device),
        betas=(0.9, B2),
        weight_decay=0.0,
        capturable=True,
        fused=True,
    )
    return [muon_opt, adamw_opt]


# ---------------------------------------------------------------------------
# "Ours" optimizer (MuonAdamW)
# ---------------------------------------------------------------------------

def build_ours_optimizer(model):
    """Create MuonAdamW optimizer with param groups from the model."""
    from optimizer import MuonAdamW

    muon_decay_params, muon_no_decay_params, adamw_params = [], [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim == 2:
            if _is_weight_decay_target(name, param):
                muon_decay_params.append(param)
            else:
                muon_no_decay_params.append(param)
        else:
            adamw_params.append(param)

    param_groups = []
    if muon_decay_params:
        param_groups.append({
            "params": muon_decay_params,
            "name": "attn_2d",
            "optimizer": "muon",
            "lr": LEARNING_RATE,
            "momentum": MUON_MOMENTUM,
            "weight_decay": WEIGHT_DECAY,
            "nesterov": MUON_NESTEROV,
            "adjust_lr_fn": MUON_ADJUST_LR_FN,
        })
    if muon_no_decay_params:
        param_groups.append({
            "params": muon_no_decay_params,
            "name": "ffn_2d",
            "optimizer": "muon",
            "lr": LEARNING_RATE,
            "momentum": MUON_MOMENTUM,
            "weight_decay": 0.0,
            "nesterov": MUON_NESTEROV,
            "adjust_lr_fn": MUON_ADJUST_LR_FN,
        })
    if adamw_params:
        param_groups.append({
            "params": adamw_params,
            "name": "non_2d",
            "optimizer": "adamw",
            "lr": LEARNING_RATE,
            "weight_decay": 0.0,
            "betas": (0.9, B2),
        })

    return MuonAdamW(param_groups)


# ---------------------------------------------------------------------------
# Synthetic batch generation
# ---------------------------------------------------------------------------

def generate_batches(n_batches=N_STEPS + 5, seed=SEED, device="cuda"):
    """Pre-generate a pool of deterministic synthetic batches on GPU."""
    rng = torch.Generator(device="cpu")
    rng.manual_seed(seed + 7777)
    device = torch.device(device)
    batches = []
    B, N, K = BATCH_SIZE, NUM_PEAKS, JEPA_NUM_TARGET_BLOCKS

    for _ in range(n_batches):
        # peak_mz: sorted random values in [0, 1]
        peak_mz = torch.rand(B, N, generator=rng).sort(dim=1).values

        # peak_intensity: random [0, 1]
        peak_intensity = torch.rand(B, N, generator=rng)

        # peak_valid_mask: contiguous valid prefix of random length 32-64
        valid_lens = torch.randint(32, 65, (B,), generator=rng)
        positions = torch.arange(N).unsqueeze(0).expand(B, -1)
        peak_valid_mask = positions < valid_lens.unsqueeze(1)

        # Divide valid range into non-overlapping segments:
        # context = first ~30%, target1 = next ~25%, target2 = next ~25%
        ctx_len = (valid_lens.float() * CONTEXT_FRACTION).long().clamp(min=1)
        tgt_len = (valid_lens.float() * TARGET_FRACTION).long().clamp(min=1)

        ctx_end = ctx_len
        tgt1_start = ctx_end
        tgt1_end = (tgt1_start + tgt_len).clamp(max=valid_lens)
        tgt2_start = tgt1_end
        tgt2_end = (tgt2_start + tgt_len).clamp(max=valid_lens)

        context_mask = positions < ctx_end.unsqueeze(1)
        context_mask = context_mask & peak_valid_mask

        target_masks = torch.zeros(B, K, N, dtype=torch.bool)
        target_masks[:, 0] = (
            (positions >= tgt1_start.unsqueeze(1))
            & (positions < tgt1_end.unsqueeze(1))
        )
        if K > 1:
            target_masks[:, 1] = (
                (positions >= tgt2_start.unsqueeze(1))
                & (positions < tgt2_end.unsqueeze(1))
            )
        target_masks = target_masks & peak_valid_mask.unsqueeze(1)

        batches.append({
            "peak_mz": peak_mz.to(dtype=torch.bfloat16, device=device),
            "peak_intensity": peak_intensity.to(dtype=torch.bfloat16, device=device),
            "peak_valid_mask": peak_valid_mask.to(device=device),
            "context_mask": context_mask.to(device=device),
            "target_masks": target_masks.to(device=device),
        })

    return batches


# ---------------------------------------------------------------------------
# Param group diagnostics
# ---------------------------------------------------------------------------

def print_param_group_summary(model, label=""):
    """Print summary of how model params split across optimizer groups."""
    muon_decay, muon_no_decay, adamw = [], [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim == 2:
            if _is_weight_decay_target(name, param):
                muon_decay.append((name, param))
            else:
                muon_no_decay.append((name, param))
        else:
            adamw.append((name, param))

    total = sum(p.numel() for _, p in muon_decay + muon_no_decay + adamw)
    print(f"  {label} param group summary:")
    print(f"    muon_decay (attn_2d):   {len(muon_decay):3d} params, "
          f"{sum(p.numel() for _, p in muon_decay):,} elements")
    print(f"    muon_no_decay (ffn_2d): {len(muon_no_decay):3d} params, "
          f"{sum(p.numel() for _, p in muon_no_decay):,} elements")
    print(f"    adamw (non_2d):         {len(adamw):3d} params, "
          f"{sum(p.numel() for _, p in adamw):,} elements")
    print(f"    total trainable:        {len(muon_decay) + len(muon_no_decay) + len(adamw):3d} params, "
          f"{total:,} elements")
