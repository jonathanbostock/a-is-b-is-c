"""
Magic transformer using a standard single-head causal attention architecture.

Key idea: superposition codes.
  - Category identity uses b_cat = ceil(log2(N)) binary ±1 features.
  - Group identity uses b_grp = ceil(log2(k)) binary ±1 features.
  - We add 2 flags: is_grp and is_cat.

This gives d_model = b_cat + b_grp + 2, much smaller than one-hot N + k + 2.

Architecture (standard components):
  - token embedding: nn.Embedding(V, d_model)
  - single-head self-attention with qkv projection: nn.Linear(d, 3d, bias=False)
  - output projection: nn.Linear(d, d, bias=False)
  - token unembedding: nn.Linear(d, V, bias=False)

All parameters are set analytically (no training).
For N=8, k=8: d_model=8, V=73, params = 1,456.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .vocab import Vocab


class MagicTransformer(nn.Module):
    def __init__(self, vocab: Vocab, sharpness: float = 20.0) -> None:
        super().__init__()
        n_categories, n_groups = vocab.n_categories, vocab.k
        b_cat = max(1, math.ceil(math.log2(max(2, n_categories))))
        b_grp = max(1, math.ceil(math.log2(max(2, n_groups))))
        d_model = b_cat + b_grp + 2
        vocab_size = vocab.vocab_size

        self.d_model = d_model

        cat_slice = slice(0, b_cat)
        grp_slice = slice(b_cat, b_cat + b_grp)
        is_grp_idx = b_cat + b_grp
        is_cat_idx = b_cat + b_grp + 1

        def binary_pm1_code(index: int, n_bits: int) -> torch.Tensor:
            bits = [(index >> bit_idx) & 1 for bit_idx in range(n_bits)]
            return torch.tensor([1.0 if bit == 1 else -1.0 for bit in bits], dtype=torch.float32)

        cat_codes = torch.stack([binary_pm1_code(i, b_cat) for i in range(n_categories)], dim=0)
        grp_codes = torch.stack([binary_pm1_code(i, b_grp) for i in range(n_groups)], dim=0)

        embedding_weight = torch.zeros(vocab_size, d_model, dtype=torch.float32)
        for cat_idx in range(n_categories):
            cat_token = vocab.cat(cat_idx)
            embedding_weight[cat_token, cat_slice] = cat_codes[cat_idx]
            embedding_weight[cat_token, is_cat_idx] = 1.0
            for grp_idx in range(n_groups):
                grp_token = vocab.grp(cat_idx, grp_idx)
                embedding_weight[grp_token, cat_slice] = cat_codes[cat_idx]
                embedding_weight[grp_token, grp_slice] = grp_codes[grp_idx]
                embedding_weight[grp_token, is_grp_idx] = 1.0

        positional_weight = torch.zeros(4, d_model, dtype=torch.float32)

        qkv_weight = torch.zeros(3 * d_model, d_model, dtype=torch.float32)
        qkv_weight[0, is_cat_idx] = sharpness
        qkv_weight[d_model, is_grp_idx] = sharpness
        for bit_idx in range(b_grp):
            qkv_weight[2 * d_model + (b_cat + bit_idx), b_cat + bit_idx] = 1.0

        out_weight = torch.zeros(d_model, d_model, dtype=torch.float32)
        for bit_idx in range(b_grp):
            out_weight[b_cat + bit_idx, b_cat + bit_idx] = 1.0

        unembed_weight = torch.zeros(vocab_size, d_model, dtype=torch.float32)
        for cat_idx in range(n_categories):
            unembed_weight[vocab.cat(cat_idx), cat_slice] = 0.5 * cat_codes[cat_idx]
            for grp_idx in range(n_groups):
                grp_token = vocab.grp(cat_idx, grp_idx)
                unembed_weight[grp_token, cat_slice] = cat_codes[cat_idx]
                unembed_weight[grp_token, grp_slice] = grp_codes[grp_idx]

        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(4, d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)
        self.unembed = nn.Linear(d_model, vocab_size, bias=False)

        with torch.no_grad():
            self.embed.weight.copy_(embedding_weight)
            self.pos_embed.weight.copy_(positional_weight)
            self.qkv.weight.copy_(qkv_weight)
            self.out.weight.copy_(out_weight)
            self.unembed.weight.copy_(unembed_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, seq_len = x.shape
        positions = torch.arange(seq_len, device=x.device)
        hidden = self.embed(x) + self.pos_embed(positions)

        q, k, v = self.qkv(hidden).chunk(3, dim=-1)
        scale = math.sqrt(self.d_model)
        scores = (q @ k.transpose(-2, -1)) / scale

        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool), diagonal=1
        )
        scores = scores.masked_fill(causal_mask, float("-inf"))
        attn = F.softmax(scores, dim=-1)

        context = attn @ v
        hidden = hidden + self.out(context)

        return self.unembed(hidden)

    def count_params(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())
