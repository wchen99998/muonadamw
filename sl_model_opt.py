"""Optimizable PeakSetSIGReg model — self-contained copy from spectra-learning.

This file is the MODEL optimization target. The agent can replace components
with custom Triton kernels, fused ops, or other GPU-level optimizations.
All changes must preserve mathematical equivalence with the reference model.
"""

import math

import torch
import torch.nn.functional as F
from torch import nn
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn

import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Custom Triton masked attention (seq_len=64, head_dim=32, non-causal)
# ---------------------------------------------------------------------------

@triton.autotune(
    configs=[
        triton.Config({}, num_warps=2, num_stages=2),
        triton.Config({}, num_warps=4, num_stages=2),
        triton.Config({}, num_warps=4, num_stages=3),
        triton.Config({}, num_warps=8, num_stages=2),
    ],
    key=["BLOCK_N", "D"],
)
@triton.jit
def _masked_attn_fwd_kernel(
    Q_ptr, K_ptr, V_ptr, Mask_ptr, O_ptr, LSE_ptr,
    stride_b, stride_n, stride_h, stride_d,
    v_stride_b, v_stride_n, v_stride_h, v_stride_d,
    stride_mb,
    sm_scale,
    actual_N,
    BLOCK_N: tl.constexpr, D: tl.constexpr, H: tl.constexpr,
):
    """Fused masked attention forward. One program per (batch*head).
    Q, K in [B, actual_N, H, D] layout. BLOCK_N >= actual_N (power of 2).
    Uses masked loads/stores when BLOCK_N > actual_N."""
    off_bh = tl.program_id(0)
    b = off_bh // H
    h = off_bh % H

    base = b * stride_b + h * stride_h
    mask_base = b * stride_mb

    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, D)
    valid_n = offs_n < actual_N
    nd_idx = offs_n[:, None] * stride_n + offs_d[None, :] * stride_d

    q = tl.load(Q_ptr + base + nd_idx, mask=valid_n[:, None], other=0.0)
    k = tl.load(K_ptr + base + nd_idx, mask=valid_n[:, None], other=0.0)
    v_base = b * v_stride_b + h * v_stride_h
    v_nd_idx = offs_n[:, None] * v_stride_n + offs_d[None, :] * v_stride_d
    v = tl.load(V_ptr + v_base + v_nd_idx, mask=valid_n[:, None], other=0.0)
    mask = tl.load(Mask_ptr + mask_base + offs_n, mask=valid_n, other=False)

    # S = Q @ K^T * scale
    s = tl.dot(q, tl.trans(k)) * sm_scale

    # Visibility mask
    attn_mask = mask[:, None] & mask[None, :]
    s = tl.where(attn_mask, s, float("-inf"))

    # Softmax
    row_max = tl.max(s, axis=1)
    row_max = tl.maximum(row_max, float("-1e20"))
    s = s - row_max[:, None]
    p = tl.exp(s)
    p = tl.where(attn_mask, p, 0.0)
    row_sum = tl.sum(p, axis=1)
    row_sum = tl.maximum(row_sum, 1e-6)
    p = p / row_sum[:, None]

    # O = P @ V
    o = tl.dot(p.to(v.dtype), v)

    tl.store(O_ptr + base + nd_idx, o, mask=valid_n[:, None])
    lse_val = row_max + tl.log(row_sum)
    tl.store(LSE_ptr + off_bh * BLOCK_N + offs_n, lse_val)


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=2, num_stages=2),
        triton.Config({}, num_warps=4, num_stages=2),
        triton.Config({}, num_warps=4, num_stages=3),
        triton.Config({}, num_warps=8, num_stages=2),
    ],
    key=["BLOCK_N", "D"],
)
@triton.jit
def _masked_attn_bwd_kernel(
    Q_ptr, K_ptr, V_ptr, Mask_ptr, O_ptr, LSE_ptr, DO_ptr,
    DQ_ptr, DK_ptr, DV_ptr,
    stride_b, stride_n, stride_h, stride_d,
    v_stride_b, v_stride_n, v_stride_h, v_stride_d,
    do_stride_b, do_stride_n, do_stride_h, do_stride_d,
    stride_mb,
    sm_scale,
    actual_N,
    BLOCK_N: tl.constexpr, D: tl.constexpr, H: tl.constexpr,
):
    """Fused masked attention backward. One program per (batch*head)."""
    off_bh = tl.program_id(0)
    b = off_bh // H
    h = off_bh % H

    base = b * stride_b + h * stride_h
    mask_base = b * stride_mb

    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, D)
    valid_n = offs_n < actual_N
    nd_idx = offs_n[:, None] * stride_n + offs_d[None, :] * stride_d

    q = tl.load(Q_ptr + base + nd_idx, mask=valid_n[:, None], other=0.0)
    k = tl.load(K_ptr + base + nd_idx, mask=valid_n[:, None], other=0.0)
    v_base = b * v_stride_b + h * v_stride_h
    v_nd_idx = offs_n[:, None] * v_stride_n + offs_d[None, :] * v_stride_d
    v = tl.load(V_ptr + v_base + v_nd_idx, mask=valid_n[:, None], other=0.0)
    o = tl.load(O_ptr + base + nd_idx, mask=valid_n[:, None], other=0.0)

    do_base = b * do_stride_b + h * do_stride_h
    do_nd_idx = offs_n[:, None] * do_stride_n + offs_d[None, :] * do_stride_d
    do = tl.load(DO_ptr + do_base + do_nd_idx, mask=valid_n[:, None], other=0.0)

    mask = tl.load(Mask_ptr + mask_base + offs_n, mask=valid_n, other=False)
    lse = tl.load(LSE_ptr + off_bh * BLOCK_N + offs_n)

    # Recompute P from Q, K, mask, LSE
    s = tl.dot(q, tl.trans(k)) * sm_scale
    attn_mask = mask[:, None] & mask[None, :]
    p = tl.exp(s - lse[:, None])
    p = tl.where(attn_mask, p, 0.0)

    # dV = P^T @ dO
    dv = tl.dot(tl.trans(p.to(do.dtype)), do)

    # dP = dO @ V^T
    dp = tl.dot(do, tl.trans(v))

    # D_i = rowsum(dO * O)
    di = tl.sum(do * o, axis=1)

    # dS = P * (dP - D_i) * scale
    ds = p * (dp - di[:, None]) * sm_scale
    ds = tl.where(attn_mask, ds, 0.0)

    # dQ = dS @ K, dK = dS^T @ Q
    dq = tl.dot(ds.to(k.dtype), k)
    dk = tl.dot(tl.trans(ds.to(q.dtype)), q)

    tl.store(DQ_ptr + base + nd_idx, dq, mask=valid_n[:, None])
    tl.store(DK_ptr + base + nd_idx, dk, mask=valid_n[:, None])
    tl.store(DV_ptr + base + nd_idx, dv, mask=valid_n[:, None])


class _MaskedAttentionFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, xq, xk, xv, vis_mask, block_n):
        """xq, xk: [B, N, H, D] contiguous, xv: [B, N, H, D] (maybe non-contiguous),
        vis_mask: [B, N] bool. block_n: power-of-2 BLOCK_N >= N (0 means N).
        Returns output in [B, N, H, D] contiguous layout."""
        B, N, H, D = xq.shape
        BLOCK_N = block_n if block_n > 0 else N
        sm_scale = 1.0 / math.sqrt(D)

        o = torch.empty_like(xq)
        lse = torch.empty(B * H, BLOCK_N, device=xq.device, dtype=torch.float32)

        s_b, s_n, s_h, s_d = xq.stride()
        vs_b, vs_n, vs_h, vs_d = xv.stride()

        m_stride_b = vis_mask.stride(0)
        _masked_attn_fwd_kernel[(B * H,)](
            xq, xk, xv, vis_mask, o, lse,
            s_b, s_n, s_h, s_d,
            vs_b, vs_n, vs_h, vs_d,
            m_stride_b, sm_scale, N,
            BLOCK_N=BLOCK_N, D=D, H=H,
        )

        ctx.save_for_backward(xq, xk, xv, vis_mask, o, lse)
        ctx.sm_scale = sm_scale
        ctx.shape = (B, N, H, D)
        ctx.block_n = BLOCK_N
        return o

    @staticmethod
    def backward(ctx, do):
        xq, xk, xv, vis_mask, o, lse = ctx.saved_tensors
        B, N, H, D = ctx.shape
        BLOCK_N = ctx.block_n

        dq = torch.empty_like(xq)
        dk = torch.empty_like(xk)
        dv = torch.empty(B, N, H, D, device=xq.device, dtype=xq.dtype)

        s_b, s_n, s_h, s_d = xq.stride()
        vs_b, vs_n, vs_h, vs_d = xv.stride()
        do_s_b, do_s_n, do_s_h, do_s_d = do.stride()

        m_stride_b = vis_mask.stride(0)
        _masked_attn_bwd_kernel[(B * H,)](
            xq, xk, xv, vis_mask, o, lse, do,
            dq, dk, dv,
            s_b, s_n, s_h, s_d,
            vs_b, vs_n, vs_h, vs_d,
            do_s_b, do_s_n, do_s_h, do_s_d,
            m_stride_b, ctx.sm_scale, N,
            BLOCK_N=BLOCK_N, D=D, H=H,
        )

        return dq, dk, dv, None, None


def masked_attention(xq, xk, xv, vis_mask, block_n=0):
    """Triton fused attention with visibility mask.
    xq, xk: [B, N, H, D] contiguous, xv: [B, N, H, D] (any strides ok),
    vis_mask: [B, N] bool. block_n: power-of-2 >= N for kernel block size (0=use N).
    Returns [B, N, H, D] contiguous.
    """
    return _MaskedAttentionFunc.apply(xq, xk, xv, vis_mask, block_n)


# ---------------------------------------------------------------------------
# Transformer building blocks (from networks/transformer_torch.py)
# ---------------------------------------------------------------------------


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    rotated = torch.empty_like(x)
    rotated[..., ::2] = -x[..., 1::2]
    rotated[..., 1::2] = x[..., ::2]
    return rotated


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cos: torch.Tensor,
    freqs_sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    q_rot = _rotate_half(xq)
    k_rot = _rotate_half(xk)
    return (xq * freqs_cos) + (q_rot * freqs_sin), (xk * freqs_cos) + (
        k_rot * freqs_sin
    )


def _build_norm(dim: int, eps: float, norm_type: str) -> nn.Module:
    kind = str(norm_type).lower()
    if kind == "rmsnorm":
        return nn.RMSNorm(dim, eps=eps)
    if kind == "layernorm":
        return nn.LayerNorm(dim, eps=eps)
    raise ValueError(f"Unsupported norm_type: {norm_type}")


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        n_heads: int,
        *,
        n_kv_heads: int | None = None,
        qk_norm: bool = False,
        norm_type: str = "rmsnorm",
    ):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.n_kv_heads = n_heads if n_kv_heads is None else n_kv_heads
        self.head_dim = self.dim // self.n_heads
        self.qk_norm = qk_norm

        out_features = (self.n_heads + 2 * self.n_kv_heads) * self.head_dim
        self.wqkv = nn.Linear(self.dim, out_features, bias=False)
        self.wo = nn.Linear(self.dim, self.dim, bias=False)

        if qk_norm:
            self.q_norm = _build_norm(self.head_dim, eps=1e-5, norm_type=norm_type)
            self.k_norm = _build_norm(self.head_dim, eps=1e-5, norm_type=norm_type)

        nn.init.xavier_normal_(self.wqkv.weight)
        nn.init.xavier_normal_(self.wo.weight)

    def forward(
        self,
        x: torch.Tensor,
        *,
        freqs_cos: torch.Tensor | None = None,
        freqs_sin: torch.Tensor | None = None,
        vis_mask: torch.Tensor | None = None,
        pad_to: int = 0,
    ) -> torch.Tensor:
        bsz, seqlen, _ = x.shape
        qkv = self.wqkv(x)

        q_size = self.n_heads * self.head_dim
        kv_size = self.n_kv_heads * self.head_dim

        xq, xk, xv = torch.split(qkv, [q_size, kv_size, kv_size], dim=-1)

        xq = xq.view(bsz, seqlen, self.n_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_kv_heads, self.head_dim)

        if self.qk_norm:
            xq = self.q_norm(xq)
            xk = self.k_norm(xk)

        if freqs_cos is not None and freqs_sin is not None:
            xq, xk = apply_rotary_emb(
                xq,
                xk,
                freqs_cos,
                freqs_sin,
            )

        xq = xq.to(dtype=xv.dtype)
        xk = xk.to(dtype=xv.dtype)

        if self.n_kv_heads != self.n_heads:
            rep = self.n_heads // self.n_kv_heads
            xk = xk.repeat_interleave(rep, dim=2)
            xv = xv.repeat_interleave(rep, dim=2)

        if vis_mask is not None:
            attn = masked_attention(xq, xk, xv, vis_mask, block_n=pad_to)
            attn = attn.reshape(bsz, seqlen, self.dim)
        else:
            q = xq.transpose(1, 2)
            k = xk.transpose(1, 2)
            v = xv.transpose(1, 2)
            attn = F.scaled_dot_product_attention(q, k, v)
            attn = attn.transpose(1, 2).contiguous().view(bsz, seqlen, self.dim)
        return self.wo(attn)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=8, num_stages=2),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def _silu_gemm_kernel(
    X_ptr, W_ptr, Y_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    stride_ym, stride_yn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """Y = SiLU(X) @ W^T. X: [M, K], W: [N, K], Y: [M, N]."""
    pid = tl.program_id(0)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        rk = k_start + tl.arange(0, BLOCK_K)
        x = tl.load(X_ptr + rm[:, None] * stride_xm + rk[None, :] * stride_xk,
                     mask=(rm[:, None] < M) & (rk[None, :] < K), other=0.0)
        x_f32 = x.to(tl.float32)
        x_silu = (x_f32 * tl.sigmoid(x_f32)).to(tl.bfloat16)
        w = tl.load(W_ptr + rn[:, None] * stride_wn + rk[None, :] * stride_wk,
                     mask=(rn[:, None] < N) & (rk[None, :] < K), other=0.0)
        acc = tl.dot(x_silu, tl.trans(w), acc=acc)

    result = acc.to(tl.bfloat16)
    mask = (rm[:, None] < M) & (rn[None, :] < N)
    tl.store(Y_ptr + rm[:, None] * stride_ym + rn[None, :] * stride_yn, result, mask=mask)


class _FusedSiLULinearFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, h, w2_weight):
        """Compute F.linear(silu(h), w2_weight) using fused Triton kernel."""
        M, K = h.shape
        N = w2_weight.shape[0]
        y = torch.empty(M, N, device=h.device, dtype=h.dtype)
        grid = lambda META: (
            triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),
        )
        _silu_gemm_kernel[grid](
            h, w2_weight, y,
            M, N, K,
            h.stride(0), h.stride(1),
            w2_weight.stride(0), w2_weight.stride(1),
            y.stride(0), y.stride(1),
        )
        ctx.save_for_backward(h, w2_weight)
        return y

    @staticmethod
    def backward(ctx, dy):
        h, w2_weight = ctx.saved_tensors
        # silu(h) for dW2 computation
        h_silu = torch.nn.functional.silu(h)
        # dW2 = dy^T @ silu(h)
        dw2 = dy.t() @ h_silu
        # dh_silu = dy @ W2
        dh_silu = dy @ w2_weight
        # dh = dh_silu * silu'(h) = dh_silu * (sigmoid(h) + h * sigmoid(h) * (1 - sigmoid(h)))
        sig_h = torch.sigmoid(h.float()).to(h.dtype)
        dh = dh_silu * (sig_h + h * sig_h * (1 - sig_h))
        return dh, dw2


def _fused_silu_linear(h, w2_weight):
    """Fused SiLU activation + linear: F.linear(silu(h), w2_weight)."""
    return _FusedSiLULinearFunc.apply(h, w2_weight)


class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        *,
        hidden_dim: int | None = None,
    ):
        super().__init__()

        hidden_dim = hidden_dim or int((4 * dim) * 2 / 3)
        hidden_dim = 4 * math.ceil(hidden_dim / 4)

        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)

        for w in (self.w1, self.w2):
            nn.init.trunc_normal_(w.weight, std=1.0 / math.sqrt(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.w1(x)
        orig_shape = h.shape
        h_2d = h.reshape(-1, h.shape[-1])
        out_2d = _fused_silu_linear(h_2d, self.w2.weight)
        return out_2d.reshape(*orig_shape[:-1], out_2d.shape[-1])


class TransformerBlock(nn.Module):
    def __init__(
        self,
        *,
        dim: int,
        n_heads: int,
        n_kv_heads: int | None,
        norm_eps: float,
        hidden_dim: int | None,
        qk_norm: bool = False,
        norm_type: str = "rmsnorm",
    ):
        super().__init__()
        self.attention = Attention(
            dim,
            n_heads,
            n_kv_heads=n_kv_heads,
            qk_norm=qk_norm,
            norm_type=norm_type,
        )
        self.feed_forward = FeedForward(
            dim,
            hidden_dim=hidden_dim,
        )
        self.attention_norm = _build_norm(dim, eps=norm_eps, norm_type=norm_type)
        self.ffn_norm = _build_norm(dim, eps=norm_eps, norm_type=norm_type)

    def forward(
        self,
        x: torch.Tensor,
        *,
        freqs_cos: torch.Tensor | None,
        freqs_sin: torch.Tensor | None,
        vis_mask: torch.Tensor | None = None,
        pad_to: int = 0,
    ) -> torch.Tensor:
        h = x + self.attention(
            self.attention_norm(x),
            freqs_cos=freqs_cos,
            freqs_sin=freqs_sin,
            vis_mask=vis_mask,
            pad_to=pad_to,
        )
        return h + self.feed_forward(self.ffn_norm(h))


# ---------------------------------------------------------------------------
# SIGReg loss (from models/losses.py)
# ---------------------------------------------------------------------------

class SIGReg(nn.Module):
    def __init__(self, knots: int = 17, num_slices: int = 256):
        super().__init__()
        self.num_slices = num_slices
        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3 / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(
        self,
        proj: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        flat = proj.reshape(-1, proj.size(-1))
        A = torch.randn(
            flat.size(-1), self.num_slices, device=flat.device, dtype=flat.dtype
        )
        A = A.div_(A.norm(p=2, dim=0))
        x_t = (flat @ A).unsqueeze(-1) * self.t
        if valid_mask is None:
            cos_mean = x_t.cos().mean(0)
            sin_mean = x_t.sin().mean(0)
        else:
            weights = valid_mask.reshape(-1).to(dtype=flat.dtype, device=flat.device)
            sample_count = weights.sum().clamp_min(1.0)
            weight_view = weights.unsqueeze(-1).unsqueeze(-1)
            cos_mean = (x_t.cos() * weight_view).sum(0) / sample_count
            sin_mean = (x_t.sin() * weight_view).sum(0) / sample_count
        err = (cos_mean - self.phi).square() + sin_mean.square()
        statistic = err @ self.weights
        return statistic.mean()


# ---------------------------------------------------------------------------
# Model helpers (from models/model.py)
# ---------------------------------------------------------------------------

def _build_non_causal_blocks(
    *,
    dim: int,
    num_layers: int,
    num_heads: int,
    num_kv_heads: int | None,
    attention_mlp_multiple: float,
    norm_eps: float = 1e-5,
    qk_norm: bool = False,
    norm_type: str = "rmsnorm",
) -> nn.ModuleList:
    block_kwargs = dict(
        dim=dim,
        n_heads=int(num_heads),
        n_kv_heads=int(num_heads) if num_kv_heads is None else int(num_kv_heads),
        norm_eps=norm_eps,
        hidden_dim=int(math.ceil(dim * attention_mlp_multiple)),
        qk_norm=qk_norm,
        norm_type=norm_type,
    )
    return nn.ModuleList(
        [TransformerBlock(**block_kwargs) for _ in range(num_layers)]
    )


def _compute_rope_freqs(
    use_rope: bool,
    seq_len: int,
    inv_freq: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    if not use_rope:
        return None, None
    positions = torch.arange(seq_len, device=device, dtype=torch.float32).unsqueeze(0)
    angles = positions.unsqueeze(-1) * inv_freq.to(device=device).view(1, 1, -1)
    angles = torch.repeat_interleave(angles, repeats=2, dim=-1)
    return angles.cos().to(dtype=dtype).unsqueeze(2), angles.sin().to(
        dtype=dtype
    ).unsqueeze(2)


def _masked_embedding_stats(
    emb: torch.Tensor,
    valid_mask: torch.Tensor,
) -> dict[str, torch.Tensor]:
    flat = emb.float().reshape(-1, emb.shape[-1])
    weights = valid_mask.reshape(-1).float()
    count = weights.sum().clamp_min(1.0)
    weights_col = weights.unsqueeze(-1)
    centered = flat - (flat * weights_col).sum(0) / count
    cov = centered.transpose(0, 1) @ (centered * weights_col) / count
    var = cov.diagonal()
    vs = var.clamp_min(1e-12)
    corr = cov / torch.sqrt(vs.unsqueeze(0) * vs.unsqueeze(1))
    d = cov.shape[0] * (cov.shape[0] - 1)
    return {
        "emb_std": torch.sqrt(var + 1e-6).mean(),
        "emb_norm": (flat.norm(dim=-1) * weights).sum() / count,
        "emb_var_mean": var.mean(),
        "emb_var_floor": var.amin(),
        "emb_cov_offdiag_abs_mean": (cov.abs().sum() - cov.diagonal().abs().sum()) / d,
        "emb_corr_offdiag_abs_mean": (corr.abs().sum() - corr.diagonal().abs().sum())
        / d,
    }


# ---------------------------------------------------------------------------
# PeakSetEncoder (from models/model.py:80-148)
# ---------------------------------------------------------------------------

def _precompute_rope_freqs(
    seq_len: int, inv_freq: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Precompute RoPE cos/sin for a fixed seq_len. Returns float32 tensors."""
    positions = torch.arange(seq_len, dtype=torch.float32).unsqueeze(0)
    angles = positions.unsqueeze(-1) * inv_freq.view(1, 1, -1)
    angles = torch.repeat_interleave(angles, repeats=2, dim=-1)
    # Shape: [1, seq_len, 1, head_dim]
    return angles.cos().unsqueeze(2), angles.sin().unsqueeze(2)


class PeakSetEncoder(nn.Module):
    def __init__(
        self,
        *,
        model_dim: int,
        num_layers: int,
        num_heads: int,
        num_kv_heads: int | None = None,
        attention_mlp_multiple: float = 4.0,
        feature_mlp_hidden_dim: int = 128,
        use_rope: bool = False,
        qk_norm: bool = False,
        norm_type: str = "rmsnorm",
        seq_len: int = 64,
    ):
        super().__init__()
        self.use_rope = bool(use_rope)
        norm_type = str(norm_type).lower()
        self.embedder = nn.Sequential(
            nn.Linear(3, feature_mlp_hidden_dim),
            nn.SiLU(),
            nn.Linear(feature_mlp_hidden_dim, model_dim),
        )
        for layer in self.embedder:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                nn.init.zeros_(layer.bias)
        _h = model_dim // int(num_heads) // 2
        self.register_buffer(
            "rope_inv_freq",
            1.0 / (10000.0 ** (torch.arange(_h, dtype=torch.float32) / _h)),
            persistent=False,
        )
        # Precompute RoPE cos/sin for fixed seq_len
        if self.use_rope:
            rope_cos, rope_sin = _precompute_rope_freqs(seq_len, self.rope_inv_freq)
            self.register_buffer("rope_cos", rope_cos, persistent=False)
            self.register_buffer("rope_sin", rope_sin, persistent=False)
        else:
            self.rope_cos = None
            self.rope_sin = None
        self.blocks = _build_non_causal_blocks(
            dim=model_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            attention_mlp_multiple=attention_mlp_multiple,
            qk_norm=qk_norm,
            norm_type=norm_type,
        )

    def forward(
        self,
        peak_mz: torch.Tensor,
        peak_intensity: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
        visible_mask: torch.Tensor | None = None,
        pack_n: int = 32,
        prefix_pack: bool = False,
        pad_to: int = 0,
    ) -> torch.Tensor:
        if visible_mask is not None and valid_mask is not None:
            vis = visible_mask & valid_mask
        else:
            vis = visible_mask if visible_mask is not None else valid_mask

        # Sparse packing: gather only visible tokens to reduce GEMM/attention cost
        if vis is not None:
            B, N = peak_mz.shape
            PACK_N = pack_n

            if prefix_pack:
                # Context tokens are a contiguous prefix — use slicing
                packed_mz = peak_mz[:, :PACK_N]
                packed_intensity = peak_intensity[:, :PACK_N]
                vis = vis[:, :PACK_N]  # [B, PACK_N] (non-contiguous view, kernel handles strides)

                log_intensity = torch.log1p(packed_intensity.clamp(min=0.0))
                x = self.embedder(torch.stack([packed_mz, packed_intensity, log_intensity], dim=-1))
                D = x.shape[-1]

                # RoPE: slice (positions are 0..PACK_N-1 for all samples)
                if self.rope_cos is not None:
                    freqs_cos = self.rope_cos.to(dtype=x.dtype)[:, :PACK_N]
                    freqs_sin = self.rope_sin.to(dtype=x.dtype)[:, :PACK_N]
                else:
                    freqs_cos = freqs_sin = None

                for block in self.blocks:
                    x = block(x, freqs_cos=freqs_cos, freqs_sin=freqs_sin, vis_mask=vis, pad_to=pad_to)

                # Pad back to original N with zeros (autograd-compatible)
                return F.pad(x, (0, 0, 0, N - PACK_N))
            else:
                # Sort by visibility (visible first), take top PACK_N
                sort_idx = vis.to(dtype=torch.int8).argsort(dim=1, descending=True, stable=True)
                pack_idx = sort_idx[:, :PACK_N]  # [B, PACK_N]

                # Gather features BEFORE embedder (gather 3 features not 256-dim embeddings)
                packed_mz = peak_mz.gather(1, pack_idx)
                packed_intensity = peak_intensity.gather(1, pack_idx)
                vis = vis.gather(1, pack_idx)  # [B, PACK_N]

                log_intensity = torch.log1p(packed_intensity.clamp(min=0.0))
                x = self.embedder(torch.stack([packed_mz, packed_intensity, log_intensity], dim=-1))
                D = x.shape[-1]

                # RoPE: gather per-sample frequencies based on original positions
                if self.rope_cos is not None:
                    rc = self.rope_cos.to(dtype=x.dtype).view(N, -1)  # [N, head_dim]
                    rs = self.rope_sin.to(dtype=x.dtype).view(N, -1)  # [N, head_dim]
                    freqs_cos = rc[pack_idx].unsqueeze(2)  # [B, PACK_N, 1, head_dim]
                    freqs_sin = rs[pack_idx].unsqueeze(2)
                else:
                    freqs_cos = freqs_sin = None

                for block in self.blocks:
                    x = block(x, freqs_cos=freqs_cos, freqs_sin=freqs_sin, vis_mask=vis, pad_to=pad_to)

                # Scatter back to original positions
                idx_expand = pack_idx.unsqueeze(-1).expand(-1, -1, D)
                out = torch.zeros(B, N, D, device=x.device, dtype=x.dtype).scatter(
                    1, idx_expand, x
                )
                return out
        else:
            # No visibility mask - run full sequence (no packing)
            log_intensity = torch.log1p(peak_intensity.clamp(min=0.0))
            x = self.embedder(torch.stack([peak_mz, peak_intensity, log_intensity], dim=-1))
            if self.rope_cos is not None:
                freqs_cos = self.rope_cos.to(dtype=x.dtype)
                freqs_sin = self.rope_sin.to(dtype=x.dtype)
            else:
                freqs_cos = freqs_sin = None
            for block in self.blocks:
                x = block(x, freqs_cos=freqs_cos, freqs_sin=freqs_sin, vis_mask=vis)
            return x


# ---------------------------------------------------------------------------
# PeakSetSIGReg (from models/model.py:151-581)
# ---------------------------------------------------------------------------

class PeakSetSIGReg(nn.Module):
    def __init__(
        self,
        *,
        model_dim: int = 768,
        encoder_num_layers: int = 20,
        encoder_num_heads: int = 12,
        encoder_num_kv_heads: int | None = None,
        attention_mlp_multiple: float = 4.0,
        feature_mlp_hidden_dim: int = 128,
        encoder_use_rope: bool = False,
        masked_token_loss_weight: float = 0.0,
        masked_token_loss_type: str = "l1",
        normalize_jepa_targets: bool = False,
        representation_regularizer: str = "sigreg",
        masked_latent_predictor_num_layers: int = 2,
        sigreg_num_slices: int = 256,
        sigreg_lambda: float = 0.1,
        sigreg_lambda_warmup_steps: int = 0,
        gco_constraints: list[dict] = (),
        gco_alpha: float = 0.99,
        gco_eta: float = 1e-3,
        gco_log_lambda_init: float = -8.0,
        gco_log_lambda_min: float = -12.0,
        gco_log_lambda_max: float = 2.0,
        jepa_num_target_blocks: int = 2,
        use_ema_teacher_target: bool = False,
        teacher_ema_decay: float = 0.996,
        teacher_ema_decay_start: float = 0.0,
        teacher_ema_decay_warmup_steps: int = 0,
        teacher_ema_update_every: int = 1,
        encoder_qk_norm: bool = False,
        norm_type: str = "rmsnorm",
        use_precursor_token: bool = False,
    ):
        super().__init__()
        self.model_dim = model_dim
        self.use_precursor_token = bool(use_precursor_token)
        self.jepa_num_target_blocks = int(jepa_num_target_blocks)
        self.sigreg_lambda = float(sigreg_lambda)
        self.sigreg_lambda_warmup_steps = int(sigreg_lambda_warmup_steps)
        self.representation_regularizer = str(representation_regularizer).lower()
        if self.representation_regularizer == "gco":
            self.representation_regularizer = "gco-sigreg"
        if self.representation_regularizer not in ("sigreg", "gco-sigreg", "none", ""):
            raise ValueError(
                f"Unsupported regularizer: {self.representation_regularizer!r}"
            )
        self.gco_alpha = float(gco_alpha)
        self.gco_eta = float(gco_eta)
        self.gco_log_lambda_min = float(gco_log_lambda_min)
        self.gco_log_lambda_max = float(gco_log_lambda_max)
        self.gco_constraint_keys: list[str] = []
        self.gco_constraint_signs: list[float] = []
        gco_targets: list[float] = []
        for c in gco_constraints:
            self.gco_constraint_keys.append(c["metric"])
            self.gco_constraint_signs.append(
                -1.0 if c["bound"] == "lower" else 1.0
            )
            gco_targets.append(float(c["target"]))
        _f = torch.float32
        _reg = self.register_buffer
        _reg(
            "gco_constraint_targets",
            torch.tensor(gco_targets, dtype=_f) if gco_targets else torch.empty(0, dtype=_f),
        )
        _reg("gco_log_lambda", torch.tensor(float(gco_log_lambda_init), dtype=_f))
        _reg("gco_c_ema", torch.tensor(0.0, dtype=_f))
        _sr_init = self.sigreg_lambda if self.sigreg_lambda_warmup_steps <= 0 else 0.0
        _reg("sigreg_lambda_target", torch.tensor(self.sigreg_lambda, dtype=_f))
        _reg("sigreg_lambda_current", torch.tensor(_sr_init, dtype=_f))
        _reg("sigreg_lambda_step", torch.zeros((), dtype=torch.int64))
        _reg(
            "sigreg_lambda_warmup_steps_tensor",
            torch.tensor(max(self.sigreg_lambda_warmup_steps, 1), dtype=_f),
            persistent=False,
        )
        self.teacher_ema_update_every = int(teacher_ema_update_every)
        teacher_ema_decay_start = float(teacher_ema_decay_start)
        teacher_ema_decay = float(teacher_ema_decay)
        teacher_ema_decay_warmup_steps = int(teacher_ema_decay_warmup_steps)
        _reg(
            "teacher_ema_decay_start_tensor",
            torch.tensor(teacher_ema_decay_start, dtype=_f),
            persistent=False,
        )
        _reg(
            "teacher_ema_decay_target",
            torch.tensor(teacher_ema_decay, dtype=_f),
            persistent=False,
        )
        _reg(
            "teacher_ema_decay_current",
            torch.tensor(
                teacher_ema_decay
                if teacher_ema_decay_warmup_steps <= 0
                else teacher_ema_decay_start,
                dtype=_f,
            ),
            persistent=False,
        )
        _reg(
            "teacher_ema_decay_step",
            torch.zeros((), dtype=torch.int64),
            persistent=False,
        )
        _reg(
            "teacher_ema_decay_warmup_steps_tensor",
            torch.tensor(max(teacher_ema_decay_warmup_steps, 1), dtype=_f),
            persistent=False,
        )
        _reg("teacher_ema_update_step", torch.zeros((), dtype=torch.int64))
        self.masked_token_loss_weight = float(masked_token_loss_weight)
        self.masked_token_loss_type = str(masked_token_loss_type).lower()
        self.normalize_jepa_targets = bool(normalize_jepa_targets)
        self.norm_type = str(norm_type).lower()
        if self.jepa_num_target_blocks < 1:
            raise ValueError("jepa_num_target_blocks must be >= 1")
        self.encoder = PeakSetEncoder(
            model_dim=model_dim,
            num_layers=encoder_num_layers,
            num_heads=encoder_num_heads,
            num_kv_heads=encoder_num_kv_heads,
            attention_mlp_multiple=attention_mlp_multiple,
            feature_mlp_hidden_dim=feature_mlp_hidden_dim,
            use_rope=encoder_use_rope,
            qk_norm=encoder_qk_norm,
            norm_type=self.norm_type,
        )
        if bool(use_ema_teacher_target):
            self.teacher_encoder: AveragedModel | None = AveragedModel(
                self.encoder,
                multi_avg_fn=get_ema_multi_avg_fn(teacher_ema_decay),
                use_buffers=True,
            )
            self.teacher_encoder.requires_grad_(False)
            self.teacher_encoder.eval()
            teacher_module = self.teacher_encoder.module
            self._teacher_ema_dst = [
                tensor.detach() for tensor in teacher_module.parameters()
            ]
            self._teacher_ema_dst.extend(
                tensor.detach() for tensor in teacher_module.buffers()
            )
            self._teacher_ema_src = [tensor.detach() for tensor in self.encoder.parameters()]
            self._teacher_ema_src.extend(
                tensor.detach() for tensor in self.encoder.buffers()
            )
        else:
            self.teacher_encoder = None
            self._teacher_ema_dst = None
            self._teacher_ema_src = None
        self._encoder_forward = self.encoder.forward
        self._teacher_encoder_forward = (
            self.teacher_encoder.forward
            if self.teacher_encoder is not None
            else None
        )
        self.latent_mask_token = nn.Parameter(torch.empty(self.model_dim))
        nn.init.normal_(self.latent_mask_token, std=0.02)
        pred_heads = max(1, min(int(encoder_num_heads), self.model_dim // 16))
        while self.model_dim % pred_heads != 0:
            pred_heads -= 1
        _ph = self.model_dim // pred_heads // 2
        predictor_inv_freq = 1.0 / (10000.0 ** (torch.arange(_ph, dtype=torch.float32) / _ph))
        self.register_buffer(
            "predictor_rope_inv_freq",
            predictor_inv_freq,
            persistent=False,
        )
        # Precompute predictor RoPE freqs (seq_len=64 always)
        pred_rope_cos, pred_rope_sin = _precompute_rope_freqs(64, predictor_inv_freq)
        self.register_buffer("predictor_rope_cos", pred_rope_cos, persistent=False)
        self.register_buffer("predictor_rope_sin", pred_rope_sin, persistent=False)
        self.masked_latent_predictor = _build_non_causal_blocks(
            dim=self.model_dim,
            num_layers=int(masked_latent_predictor_num_layers),
            num_heads=pred_heads,
            num_kv_heads=None,
            attention_mlp_multiple=attention_mlp_multiple,
            qk_norm=encoder_qk_norm,
            norm_type=self.norm_type,
        )
        self._predict_masked_latents = self.predict_masked_latents
        self.sigreg = SIGReg(num_slices=int(sigreg_num_slices))

    def train(self, mode: bool = True) -> "PeakSetSIGReg":
        super().train(mode)
        if self.teacher_encoder is not None:
            self.teacher_encoder.eval()
        return self

    @torch.no_grad()
    def update_teacher(self) -> None:
        if self.teacher_encoder is None:
            return
        step = int(self.teacher_ema_update_step.item())
        self.teacher_ema_update_step.add_(1)
        if step % self.teacher_ema_update_every != 0:
            return
        self.advance_teacher_ema_decay_schedule()
        if int(self.teacher_encoder.n_averaged.item()) == 0:
            for teacher_tensor, encoder_tensor in zip(
                self._teacher_ema_dst,
                self._teacher_ema_src,
                strict=True,
            ):
                teacher_tensor.copy_(encoder_tensor)
        else:
            torch._foreach_lerp_(
                self._teacher_ema_dst,
                self._teacher_ema_src,
                1.0 - float(self.teacher_ema_decay_current),
            )
        self.teacher_encoder.n_averaged.add_(1)

    @torch.no_grad()
    def advance_teacher_ema_decay_schedule(self) -> None:
        step = self.teacher_ema_decay_step.to(
            dtype=self.teacher_ema_decay_current.dtype
        )
        ratio = torch.clamp(
            step / self.teacher_ema_decay_warmup_steps_tensor, max=1.0
        )
        delta = self.teacher_ema_decay_target - self.teacher_ema_decay_start_tensor
        self.teacher_ema_decay_current.copy_(
            self.teacher_ema_decay_start_tensor + delta * ratio
        )
        self.teacher_ema_decay_step.add_(1)

    @torch.no_grad()
    def advance_sigreg_lambda_schedule(self) -> None:
        if self.representation_regularizer != "sigreg":
            return
        if self.sigreg_lambda_warmup_steps <= 0:
            return
        step = self.sigreg_lambda_step.to(dtype=self.sigreg_lambda_current.dtype)
        ratio = torch.clamp(step / self.sigreg_lambda_warmup_steps_tensor, max=1.0)
        self.sigreg_lambda_current.copy_(self.sigreg_lambda_target * ratio)
        self.sigreg_lambda_step.add_(1)

    def predict_masked_latents(
        self,
        x: torch.Tensor,
        visible_mask: torch.Tensor,
        pack_n: int = 35,
    ) -> torch.Tensor:
        BK, N, D = x.shape
        PACK_N = pack_n

        # Sort visible tokens first, pack top PACK_N
        sort_idx = visible_mask.to(dtype=torch.int8).argsort(
            dim=1, descending=True, stable=True
        )
        pack_idx = sort_idx[:, :PACK_N]  # [BK, PACK_N]

        packed_x = x.gather(1, pack_idx.unsqueeze(-1).expand(-1, -1, D))
        packed_vis = visible_mask.gather(1, pack_idx)

        # RoPE: gather per-sample frequencies based on original positions
        rc = self.predictor_rope_cos.to(dtype=x.dtype).view(N, -1)  # [N, head_dim]
        rs = self.predictor_rope_sin.to(dtype=x.dtype).view(N, -1)
        freqs_cos = rc[pack_idx].unsqueeze(2)  # [BK, PACK_N, 1, head_dim]
        freqs_sin = rs[pack_idx].unsqueeze(2)

        # Pad to next power of 2 for Triton attention kernel
        pad_to = 64

        for block in self.masked_latent_predictor:
            packed_x = block(
                packed_x,
                freqs_cos=freqs_cos,
                freqs_sin=freqs_sin,
                vis_mask=packed_vis,
                pad_to=pad_to,
            )

        # Scatter back to original positions
        idx_expand = pack_idx.unsqueeze(-1).expand(-1, -1, D)
        out = torch.zeros(BK, N, D, device=packed_x.device, dtype=packed_x.dtype).scatter(
            1, idx_expand, packed_x
        )
        return out

    def pool(
        self,
        embeddings: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        mask = valid_mask.unsqueeze(-1).float()
        return (embeddings * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)

    @staticmethod
    def prepend_precursor_token(
        peak_mz: torch.Tensor,
        peak_intensity: torch.Tensor,
        peak_valid_mask: torch.Tensor,
        precursor_mz: torch.Tensor,
        context_mask: torch.Tensor | None = None,
        target_masks: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        B, N = peak_mz.shape
        device = peak_mz.device
        pre_int = torch.full((B, 1), -1.0, device=device, dtype=peak_mz.dtype)
        pre_valid = torch.ones(B, 1, device=device, dtype=torch.bool)
        result: dict[str, torch.Tensor] = {
            "peak_mz": torch.cat([precursor_mz.unsqueeze(1), peak_mz], dim=1),
            "peak_intensity": torch.cat([pre_int, peak_intensity], dim=1),
            "peak_valid_mask": torch.cat([pre_valid, peak_valid_mask], dim=1),
        }
        if context_mask is not None:
            pre_ctx = torch.ones(B, 1, device=device, dtype=torch.bool)
            result["context_mask"] = torch.cat([pre_ctx, context_mask], dim=1)
        if target_masks is not None:
            K = target_masks.shape[1]
            pre_tgt = torch.zeros(B, K, 1, device=device, dtype=torch.bool)
            result["target_masks"] = torch.cat([pre_tgt, target_masks], dim=2)
        return result

    @torch.no_grad()
    def compute_teacher_targets(
        self,
        augmented_batch: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Compute teacher encoder targets. Runs under no_grad.

        Full-view encoding: teacher sees the entire valid spectrum once,
        result is expanded to K target views.
        """
        peak_mz = augmented_batch["peak_mz"]
        peak_intensity = augmented_batch["peak_intensity"]
        peak_valid_mask = augmented_batch["peak_valid_mask"]
        B, N = peak_mz.shape
        K = self.jepa_num_target_blocks
        # Single full-view forward pass (visible_mask = peak_valid_mask)
        # pack_n=64 since all valid tokens (up to 64) must be visible
        teacher_full = self._teacher_encoder_forward(
            peak_mz,
            peak_intensity,
            valid_mask=peak_valid_mask,
            visible_mask=peak_valid_mask,
            pack_n=64,
            prefix_pack=True,
            pad_to=64,
        )
        # Expand to K views (expanded view is fine with CUDA graphs)
        return teacher_full.unsqueeze(1).expand(-1, K, -1, -1)

    def forward_augmented(
        self,
        augmented_batch: dict[str, torch.Tensor],
        teacher_targets: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        peak_mz = augmented_batch["peak_mz"]
        peak_intensity = augmented_batch["peak_intensity"]
        peak_valid_mask = augmented_batch["peak_valid_mask"]
        context_mask = augmented_batch["context_mask"] & peak_valid_mask
        target_masks = augmented_batch["target_masks"] & peak_valid_mask.unsqueeze(1)
        B, N = peak_mz.shape
        K = self.jepa_num_target_blocks
        # Student encoder: only context views (target views unused when teacher is active
        # and representation_regularizer is "none")
        context_emb = self._encoder_forward(
            peak_mz,
            peak_intensity,
            valid_mask=peak_valid_mask,
            visible_mask=context_mask,
            pack_n=19,
            prefix_pack=True,  # context tokens are always a contiguous prefix
            pad_to=32,  # pad to power-of-2 only for Triton attention
        )  # [B, N, D]
        if teacher_targets is not None:
            target_token_target = teacher_targets
        elif self.teacher_encoder is not None:
            # Full-view teacher encoding: single pass with peak_valid_mask
            with torch.no_grad():
                teacher_full = self._teacher_encoder_forward(
                    peak_mz,
                    peak_intensity,
                    valid_mask=peak_valid_mask,
                    visible_mask=peak_valid_mask,
                    pack_n=64,
                    prefix_pack=True,
                    pad_to=64,
                ).detach()
            target_token_target = teacher_full.unsqueeze(1).expand(-1, K, -1, -1)
        else:
            # Without teacher_encoder, use student encoder with full-view encoding
            with torch.no_grad():
                teacher_full = self._encoder_forward(
                    peak_mz,
                    peak_intensity,
                    valid_mask=peak_valid_mask,
                    visible_mask=peak_valid_mask,
                    pack_n=64,
                    prefix_pack=True,
                    pad_to=64,
                ).detach()
            target_token_target = teacher_full.unsqueeze(1).expand(-1, K, -1, -1)

        ctx_mask_v = context_mask.unsqueeze(1)
        context_emb_by_view = context_emb.unsqueeze(1).expand(-1, K, -1, -1)
        predictor_input = context_emb_by_view * ctx_mask_v.unsqueeze(-1)
        predictor_input = torch.where(
            target_masks.unsqueeze(-1),
            self.latent_mask_token.view(1, 1, 1, -1).to(context_emb),
            predictor_input,
        )
        predictor_output = self._predict_masked_latents(
            predictor_input.reshape(B * K, N, -1),
            (ctx_mask_v | target_masks).reshape(B * K, N),
        ).reshape(B, K, N, -1)
        loss_pred = predictor_output
        loss_target = target_token_target
        if self.normalize_jepa_targets:
            loss_pred = F.normalize(loss_pred, dim=-1)
            loss_target = F.normalize(loss_target, dim=-1)
        if self.masked_token_loss_type == "l2":
            per_token_reg = (
                (loss_pred - loss_target).square().mean(dim=-1)
            )
        elif self.masked_token_loss_type == "l2_sum":
            per_token_reg = (
                (loss_pred - loss_target).square().sum(dim=-1)
            )
        elif self.masked_token_loss_type == "l1":
            per_token_reg = (loss_pred - loss_target).abs().mean(dim=-1)
        else:
            raise ValueError(
                f"Unsupported masked_token_loss_type: {self.masked_token_loss_type}"
            )
        target_mask_float = target_masks.float()
        reg_num = (per_token_reg * target_mask_float).sum()
        reg_den = target_mask_float.sum().clamp_min(1.0)
        local_global_loss = reg_num / reg_den
        sigreg_lambda_current = (
            self.sigreg_lambda_current.to(dtype=context_emb.dtype)
            if self.sigreg_lambda_warmup_steps > 0
            else context_emb.new_tensor(self.sigreg_lambda)
        )
        jepa_term = self.masked_token_loss_weight * local_global_loss
        use_sigreg = (
            self.representation_regularizer == "sigreg" and self.sigreg_lambda > 0
        )
        use_gco = self.representation_regularizer == "gco-sigreg"
        gco_lambda = self.gco_log_lambda.exp().to(dtype=context_emb.dtype)
        valid_peak_count = peak_valid_mask.float().sum().clamp_min(1.0)
        if not use_sigreg and not use_gco and not self.gco_constraint_keys:
            zero = context_emb.new_zeros(())
            encoder_metric_names = (
                "emb_std",
                "emb_norm",
                "emb_var_mean",
                "emb_var_floor",
                "emb_cov_offdiag_abs_mean",
                "emb_corr_offdiag_abs_mean",
            )
            collapse_metrics = {
                f"{prefix}_{name}": zero
                for prefix in ("global", "local")
                for name in encoder_metric_names
            }
            metrics = {
                "loss": jepa_term,
                "token_sigreg_loss": zero,
                "local_global_loss": local_global_loss,
                "sigreg_term": zero,
                "jepa_term": jepa_term,
                "target_sigreg_term_over_jepa_term": zero,
                "context_fraction": context_mask.float().sum() / valid_peak_count,
                "masked_fraction": target_masks.float().sum() / valid_peak_count,
                "sigreg_lambda_current": sigreg_lambda_current,
                "gco_lambda": gco_lambda,
                "gco_log_lambda": self.gco_log_lambda.to(dtype=context_emb.dtype),
                "gco_c_ema": self.gco_c_ema.to(dtype=context_emb.dtype),
                "gco_constraint": zero,
                **{
                    f"encoder_{name}": zero
                    for name in encoder_metric_names
                },
                "local_to_global_emb_std_ratio": zero,
                **collapse_metrics,
            }
            return metrics
        # Compute target student embeddings for sigreg/gco metrics
        target_emb = (
            self._encoder_forward(
                peak_mz.repeat_interleave(K, dim=0),
                peak_intensity.repeat_interleave(K, dim=0),
                valid_mask=peak_valid_mask.repeat_interleave(K, dim=0),
                visible_mask=target_masks.reshape(B * K, N),
            ).reshape(B, K, N, -1)
        )
        branch_emb = torch.cat([context_emb.unsqueeze(1), target_emb], dim=1)
        branch_visible = torch.cat([context_mask.unsqueeze(1), target_masks], dim=1)
        V = branch_emb.shape[1]
        fused_emb = branch_emb.reshape(V * B, N, -1)
        fused_visible = branch_visible.reshape(V * B, N)
        with torch.no_grad():
            emb_f = fused_emb.float().reshape(B, V, N, -1)
            mask_f = fused_visible.reshape(B, V, N)
            collapse_metrics: dict[str, torch.Tensor] = {}
            for prefix, e, m in [
                ("global", emb_f[:, 0], mask_f[:, 0]),
                ("local", emb_f[:, 1:], mask_f[:, 1:]),
            ]:
                for k, v in _masked_embedding_stats(e, m).items():
                    collapse_metrics[f"{prefix}_{k}"] = v
            reg_stats = _masked_embedding_stats(fused_emb, fused_visible)
            all_stats = {**reg_stats, **collapse_metrics}
            if self.gco_constraint_keys:
                constraint_vals = torch.stack([
                    sign * (all_stats[key].float() - self.gco_constraint_targets[i])
                    for i, (key, sign) in enumerate(
                        zip(self.gco_constraint_keys, self.gco_constraint_signs)
                    )
                ])
                gco_constraint = constraint_vals.amax(dim=0)
            else:
                gco_constraint = torch.tensor(0.0, device=fused_emb.device)
                constraint_vals = None
        if self.representation_regularizer == "gco-sigreg":
            with torch.no_grad():
                if self.training:
                    self.gco_c_ema.mul_(self.gco_alpha).add_(
                        (1.0 - self.gco_alpha) * gco_constraint
                    )
                    self.gco_log_lambda.add_(self.gco_eta * self.gco_c_ema)
                    self.gco_log_lambda.clamp_(
                        self.gco_log_lambda_min, self.gco_log_lambda_max
                    )
                gco_lambda = self.gco_log_lambda.exp().to(dtype=context_emb.dtype)
        if use_sigreg or use_gco:
            token_sigreg_loss = self.sigreg(fused_emb, valid_mask=fused_visible)
            if use_gco:
                sigreg_lambda_current = gco_lambda
            sigreg_term = sigreg_lambda_current * token_sigreg_loss
        else:
            token_sigreg_loss = context_emb.new_tensor(0.0)
            sigreg_term = context_emb.new_tensor(0.0)
        metrics = {
            "loss": jepa_term + sigreg_term,
            "token_sigreg_loss": token_sigreg_loss,
            "local_global_loss": local_global_loss,
            "sigreg_term": sigreg_term,
            "jepa_term": jepa_term,
            "target_sigreg_term_over_jepa_term": sigreg_term
            / jepa_term.clamp_min(1e-8),
            "context_fraction": context_mask.float().sum() / valid_peak_count,
            "masked_fraction": target_masks.float().sum() / valid_peak_count,
            "sigreg_lambda_current": sigreg_lambda_current,
            "gco_lambda": gco_lambda,
            "gco_log_lambda": self.gco_log_lambda.to(dtype=context_emb.dtype),
            "gco_c_ema": self.gco_c_ema.to(dtype=context_emb.dtype),
            "gco_constraint": gco_constraint.to(dtype=context_emb.dtype),
            **{f"encoder_{k}": v.to(context_emb.dtype) for k, v in reg_stats.items()},
            "local_to_global_emb_std_ratio": collapse_metrics["local_emb_std"]
            / collapse_metrics["global_emb_std"],
            **collapse_metrics,
        }
        if constraint_vals is not None:
            for i, key in enumerate(self.gco_constraint_keys):
                metrics[f"gco_constraint_{key}"] = constraint_vals[i].to(
                    dtype=context_emb.dtype
                )
        return metrics

    def encode(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        mz, intensity, valid = (
            batch["peak_mz"],
            batch["peak_intensity"],
            batch["peak_valid_mask"],
        )
        return self.pool(self.encoder(mz, intensity, valid_mask=valid), valid)
