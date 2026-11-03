from dataclasses import replace

import torch
from torch import nn
from torch.nn import functional as F

from .data import PAD


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        d = config.d_model
        self.n_heads = config.n_heads
        self.norm1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.projection = nn.Linear(d, d)
        self.norm2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, config.ff_multiplier * d), nn.GELU(),
                                 nn.Linear(config.ff_multiplier * d, d), nn.Dropout(config.dropout))
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        batch, length, width = x.shape
        qkv = self.qkv(self.norm1(x)).reshape(batch, length, 3, self.n_heads, width // self.n_heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        attention = F.scaled_dot_product_attention(
            q, k, v, is_causal=True, dropout_p=self.dropout.p if self.training else 0.0)
        x = x + self.dropout(self.projection(attention.transpose(1, 2).reshape(batch, length, width)))
        return x + self.mlp(self.norm2(x))


class AutoregressiveTransformer(nn.Module):
    def __init__(self, config, vocab_size, max_seq_len):
        super().__init__()
        self.config = config
        self.max_seq_len = max_seq_len
        self.token_embedding = nn.Embedding(vocab_size, config.d_model)
        self.position_embedding = nn.Embedding(max_seq_len, config.d_model) if config.positional_embeddings else None
        # Continuous numeric V grade and angle/90; no categorical grade lookup.
        self.condition_projection = nn.Linear(2, config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layers)])
        self.norm = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight
        self.apply(self._initialize)

    @staticmethod
    def _initialize(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, tokens, conditions, output_mask=None):
        length = tokens.shape[1]
        if length > self.max_seq_len:
            raise ValueError(f"Context {length} exceeds {self.max_seq_len}")
        x = self.token_embedding(tokens) + self.condition_projection(conditions).unsqueeze(1)
        if self.position_embedding is not None:
            x = x + self.position_embedding(torch.arange(length, device=tokens.device))
        x = self.dropout(x)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        # Avoid projecting padding to the full vocabulary during loss/metrics.
        if output_mask is not None:
            x = x[output_mask]
        return self.lm_head(x)


def parameter_count(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def resolve_model(config, vocab_size, max_seq_len, train_rows):
    target = round(train_rows * config.parameters_per_example)
    if config.d_model != "auto":
        return config, target
    # Size against the learned-position baseline for matched-width ablations.
    # The formula counts the tied embedding once and every trainable bias/norm.
    def count(width):
        f = config.ff_multiplier
        return ((vocab_size + max_seq_len + 5) * width
                + config.n_layers * ((4 + 2 * f) * width * width + (9 + f) * width))
    widths = range(config.n_heads, max(config.n_heads * 2, int(target**0.5) + config.n_heads), config.n_heads)
    width = min(widths, key=lambda d: abs(count(d) - target))
    return replace(config, d_model=width), target


def token_loss(logits, targets, reduction="mean"):
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1),
                           ignore_index=PAD, reduction=reduction)
