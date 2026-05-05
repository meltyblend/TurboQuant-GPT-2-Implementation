import inspect
import math
import os
import time
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F


# multi-head masked self-attention
class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)

        # this is our output projection, maps concatenated heads back to n_embd
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

        # regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    # attention computation
    def forward(self, x):
        B, T, C = x.size()# batch size, sequence length, embedding dimensionality (n_embd)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        # nh is "number of heads", hs is "head size", and C (number of channels) = nh * hs
        # e.g. in GPT-2 (124M), n_head=12, hs=64, so nh*hs=C=768 channels in the Transformer
        # directly from Andrej Karpathy's GPT-2 implementation comments

        qkv = self.c_attn(x) # tensor of shape (B, T, 3 * n_embd)

        q, k, v = qkv.split(self.n_embd, dim=2) # split into q, k, v tensors

        # reshape q, k, v for multi-head attention
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        # attention(Q, K, V) = softmax((QK^T / sqrt(d_k))+M)V, M is a causal mask
        # setting negative infinity to the upper triangular part of the attention matrix
        # to prevent the model from attending to future tokens
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        # merge heads and project back to n_embd
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
    block_size: int = 1024 # maximum sequence length
    vocab_size: int = 50257 # num of tokens: 50,000 BPE merges + 256 bytes tokens + 1 EOS token
    n_layer: int = 12       # 12 layers
    n_head: int = 12        # 12 heads
    n_embd: int = 768       # embedding dimensionality


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
