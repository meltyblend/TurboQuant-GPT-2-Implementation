import inspect
import math
import os
import time
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F



class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(C, dim=2)
        k = k.view(B, T, C // 3, 3).transpose(1, 2)
        q = q.view(B, T, C // 3, 3).transpose(1, 2)
        v = v.view(B, T, C // 3, 3).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        # linear projections surrounding our GELU activation function
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU()                                  # GPT2 used the approx version, we will not
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    # forward pass of what the block computes
    def forward(self, x):
        # attention is the reduce
        # mlp is the map
        x = x + self.attn(self.ln1(x))  # residual around attention sub-layer
        x = x + self.mlp(self.ln2(x))  # residual around MLP sub-layer or FFN
        return x


@dataclass
class GPTConfig:
    block_size: int = 256
    vocab_size: int = 65
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # module that allows us to index into the submodules using keys
        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),  # weight of the token embeddings
                wpe=nn.Embedding(config.block_size, config.n_embd),  # weight of the positional embeddings

                drop=nn.Dropout(0.1),

                h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]), # transformer blocks: Add & Norm,
                                                                                 # Masked Multi-Head Attention, feed forward

                ln_f=nn.LayerNorm(config.n_embd),   # final layer norm
            )
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False) # "linear part"
