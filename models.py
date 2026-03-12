"""Self-contained GPT-2 model (~19.5M params) for optimizer testing.

Architecture: Pre-norm transformer (LayerNorm → Attention/FFN → residual)
Config: d_model=512, n_layers=6, n_heads=8, d_ff=2048, vocab=512, seq_len=128
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# Model config
D_MODEL = 512
N_LAYERS = 6
N_HEADS = 8
D_FF = 2048
VOCAB_SIZE = 512
SEQ_LEN = 128
DROPOUT = 0.0


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int = D_MODEL, n_heads: int = N_HEADS):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        B, T, C = x.shape
        qkv = self.qkv_proj(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(out)


class FeedForward(nn.Module):
    def __init__(self, d_model: int = D_MODEL, d_ff: int = D_FF):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=True)
        self.w2 = nn.Linear(d_ff, d_model, bias=True)

    def forward(self, x: Tensor) -> Tensor:
        return self.w2(F.gelu(self.w1(x)))


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int = D_MODEL, n_heads: int = N_HEADS, d_ff: int = D_FF):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, d_ff)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class GPT2(nn.Module):
    def __init__(
        self,
        vocab_size: int = VOCAB_SIZE,
        seq_len: int = SEQ_LEN,
        d_model: int = D_MODEL,
        n_layers: int = N_LAYERS,
        n_heads: int = N_HEADS,
        d_ff: int = D_FF,
    ):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(seq_len, d_model)
        self.blocks = nn.ModuleList(
            [TransformerBlock(d_model, n_heads, d_ff) for _ in range(n_layers)]
        )
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, idx: Tensor) -> Tensor:
        B, T = idx.shape
        x = self.tok_emb(idx) + self.pos_emb(torch.arange(T, device=idx.device))
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        return self.lm_head(x)


def get_param_groups(model: GPT2) -> list[dict]:
    """Partition model parameters into 3 optimizer groups.

    Returns:
        List of 3 dicts, each with 'params' (list[Tensor]) and 'name' (str):
        1. attn_2d: attention weight matrices (qkv_proj.weight, out_proj.weight) → Muon
        2. ffn_2d: FFN weight matrices (w1.weight, w2.weight) → Muon
        3. non_2d: everything else (embeddings, layer norms, biases, lm_head) → AdamW
    """
    attn_2d = []
    ffn_2d = []
    non_2d = []

    for name, param in model.named_parameters():
        if "attn" in name and ".weight" in name and param.ndim == 2:
            attn_2d.append(param)
        elif "ff" in name and ".weight" in name and param.ndim == 2:
            ffn_2d.append(param)
        else:
            non_2d.append(param)

    return [
        {"params": attn_2d, "name": "attn_2d"},
        {"params": ffn_2d, "name": "ffn_2d"},
        {"params": non_2d, "name": "non_2d"},
    ]
