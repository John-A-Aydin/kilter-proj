import torch
from torch import nn
from torch.nn import functional as F

from .data import PAD, SEP

MAX_GRADE = 16.0
# Stands for "no grade" in the grade slot of conditions; real grades are never negative.
NULL_GRADE = -1.0
# Columns of condition_features that depend on the grade.
GRADE_FEATURES = [0, 2, 3, 4, 5, 10, 11, 12, 13]


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


def condition_features(conditions, null_grade=False):
    """[V grade, angle/90] -> normalized values with sinusoidal features, so their effect can be nonlinear.

    With ``null_grade``, null rows get zeros in the grade columns and a 1 in one extra column.
    """
    values = torch.stack([conditions[:, 0] / MAX_GRADE, conditions[:, 1]], dim=1)
    scaled = values.unsqueeze(-1) * (torch.pi * 2.0 ** torch.arange(4, device=values.device))
    features = torch.cat([values, scaled.sin().flatten(1), scaled.cos().flatten(1)], dim=1)
    if not null_grade:
        return features
    null = conditions[:, :1] == NULL_GRADE
    features[:, GRADE_FEATURES] = features[:, GRADE_FEATURES].masked_fill(null, 0.0)
    return torch.cat([features, null.to(features.dtype)], dim=1)


class ClimbTransformer(nn.Module):
    """Causal decoder over BOS given SEP rest EOS, with geometry-aware tied hold/role embeddings."""

    def __init__(self, config, hold_features, max_len):
        super().__init__()
        d = config.d_model
        self.config = config
        self.max_len = max_len
        self.register_buffer("hold_features", torch.as_tensor(hold_features, dtype=torch.float32), persistent=False)
        n_holds = self.hold_features.shape[0]
        self.vocab_size = 4 + 4 * n_holds
        self.special = nn.Embedding(4, d)
        self.hold = nn.Embedding(n_holds, d)
        self.role = nn.Embedding(4, d)
        self.geometry = nn.Sequential(nn.Linear(self.hold_features.shape[1], d), nn.GELU(), nn.Linear(d, d))
        self.position = nn.Embedding(max_len, d)
        self.segment = nn.Embedding(2, d)
        self.condition = nn.Sequential(nn.Linear(19 if config.null_grade else 18, d), nn.GELU(), nn.Linear(d, d))
        self.dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layers)])
        self.norm = nn.LayerNorm(d)
        self.output_bias = nn.Parameter(torch.zeros(self.vocab_size))
        self.apply(self._initialize)

    @staticmethod
    def _initialize(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def token_table(self):
        """Embedding of every token: specials, then hold identity + geometry + role, in token-id order."""
        holds = self.hold.weight + self.geometry(self.hold_features)
        hold_roles = (holds.unsqueeze(1) + self.role.weight.unsqueeze(0)).reshape(-1, holds.shape[1])
        return torch.cat([self.special.weight, hold_roles])

    def forward(self, tokens, conditions, output_mask=None):
        length = tokens.shape[1]
        if length > self.max_len:
            raise ValueError(f"Context {length} exceeds {self.max_len}")
        # Positions restart at SEP so the remaining-hold positions do not depend on how many were given.
        index = torch.arange(length, device=tokens.device).unsqueeze(0)
        is_sep = tokens == SEP
        segment = is_sep.long().cumsum(1).clamp(max=1)
        sep_index = (is_sep.long() * index).amax(1, keepdim=True)
        position = index - sep_index * segment
        table = self.token_table()
        x = (F.embedding(tokens, table) + self.position(position) + self.segment(segment)
             + self.condition(condition_features(conditions, self.config.null_grade)).unsqueeze(1))
        x = self.dropout(x)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        if output_mask is not None:
            x = x[output_mask]
        return x @ table.T + self.output_bias


def parameter_count(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def weighted_token_loss(logits, targets, row_weights):
    """Cross entropy summed per target, weighted per row, normalized by weighted target count.

    ``logits`` covers only non-PAD targets (row-major), matching ``targets[targets != PAD]``.
    """
    mask = targets != PAD
    flat_targets = targets[mask]
    weights = row_weights.unsqueeze(1).expand_as(targets)[mask]
    losses = F.cross_entropy(logits, flat_targets, reduction="none")
    return (losses * weights).sum() / weights.sum(), losses
