from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int) -> None:
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q = q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        scale = math.sqrt(self.d_head)
        attn = (q @ k.transpose(-2, -1)) / scale
        causal_mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
        attn = attn.masked_fill(causal_mask, float("-inf"))
        attn = F.softmax(attn, dim=-1)

        out = (attn @ v).transpose(1, 2).reshape(B, T, C)
        return self.out(out)


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class ToyTransformer(nn.Module):
    """
    A small causal transformer for the group-prediction task.

    Input:  token sequence of length 4: [BOS, cat_x, cat_x_grp_y, cat_z]
    Output: logits at the last position predict cat_z_grp_y.

    Standard pre-norm architecture: attention + MLP sublayers with residuals.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        max_seq_len: int = 4,
    ) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_seq_len, d_model)
        self.blocks = nn.ModuleList([TransformerBlock(d_model, n_heads) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.unembed = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T = x.shape
        pos = torch.arange(T, device=x.device)
        h = self.embed(x) + self.pos_embed(pos)
        for block in self.blocks:
            h = block(h)
        return self.unembed(self.ln_f(h))

    def get_residual_at_layer(self, x: torch.Tensor, layer_idx: int) -> torch.Tensor:
        """Return the hidden state after block[layer_idx] on the final token.

        Shape: (B, d_model). layer_idx is clamped to [0, n_layers-1].
        """
        B, T = x.shape
        pos = torch.arange(T, device=x.device)
        h = self.embed(x) + self.pos_embed(pos)
        target = min(layer_idx, len(self.blocks) - 1)
        for i, block in enumerate(self.blocks):
            h = block(h)
            if i == target:
                return h[:, -1, :]
        return h[:, -1, :]

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
