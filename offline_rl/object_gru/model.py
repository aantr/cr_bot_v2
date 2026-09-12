"""Objects -> per-frame Transformer CLS -> eight-state GRU -> 2/4/576 heads."""
from dataclasses import asdict, dataclass
from contextlib import nullcontext
import math

import torch
from torch import nn

from .features import auxiliary_features, object_features

ARCHITECTURE = "object_transformer_gru_v1"


@dataclass(frozen=True)
class ObjectGRUConfig:
    num_cards: int
    num_units: int
    sequence_length: int = 8
    class_embedding: int = 32
    d_model: int = 128
    n_heads: int = 4
    object_layers: int = 3
    ff_multiplier: int = 4
    state_dim: int = 512
    gru_hidden: int = 256
    gru_layers: int = 2
    dropout: float = .1

    def __post_init__(self):
        for key, value in asdict(self).items():
            if key != "dropout" and (isinstance(value, bool) or not isinstance(value, int) or value < 1):
                raise ValueError(f"{key} must be a positive integer")
        if self.num_cards < 3 or self.num_units < 2:
            raise ValueError("Vocabulary must include reserved card/unit IDs")
        if self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        if not math.isfinite(self.dropout) or not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0,1)")

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_encoding_config(cls, encoding, **kwargs):
        return cls(num_cards=len(encoding["vocabulary"]["cards"]),
                   num_units=len(encoding["vocabulary"]["units"]),
                   sequence_length=encoding["sequence_length"], **kwargs)


class ObjectGRUPolicy(nn.Module):
    """No object-index embeddings: CLS is permutation invariant in eval mode.

    forward returns heads for each timestep for shared masked training/metrics.
    Dataset supervision defaults to the last REAL timestep, and online inference
    selects that same timestep (not the right-padded final array position).
    GRU memory is recomputed over the bounded window, reset for each battle;
    persistent hidden state is deliberately not used, matching training exactly.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        c = config
        self.unit_embedding = nn.Embedding(c.num_units, c.class_embedding, padding_idx=0)
        self.object_projection = nn.Linear(c.class_embedding + 6, c.d_model)
        self.cls = nn.Parameter(torch.empty(1, 1, c.d_model))
        nn.init.normal_(self.cls, std=.02)
        layer = nn.TransformerEncoderLayer(c.d_model, c.n_heads, c.d_model * c.ff_multiplier,
                                            c.dropout, activation="gelu", batch_first=True, norm_first=True)
        self.object_transformer = nn.TransformerEncoder(layer, c.object_layers,
                                                        norm=nn.LayerNorm(c.d_model), enable_nested_tensor=False)
        # TransformerEncoder clones its prototype: initialize each layer independently.
        for module in self.object_transformer.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.MultiheadAttention):
                nn.init.xavier_uniform_(module.in_proj_weight)
                nn.init.zeros_(module.in_proj_bias)
        self.tower_projection = nn.Sequential(nn.Linear(4 * 5, 64), nn.GELU())
        self.card_embedding = nn.Embedding(c.num_cards, c.class_embedding, padding_idx=0)
        self.hand_projection = nn.Sequential(nn.Linear(c.class_embedding + 3, 64), nn.GELU())
        self.global_projection = nn.Sequential(nn.Linear(4, 32), nn.GELU())
        self.state_projection = nn.Sequential(nn.Linear(c.d_model + 64 + 4 * 64 + 32, c.state_dim),
                                               nn.LayerNorm(c.state_dim), nn.GELU(), nn.Dropout(c.dropout))
        self.gru = nn.GRU(c.state_dim, c.gru_hidden, c.gru_layers, batch_first=True,
                          dropout=c.dropout if c.gru_layers > 1 else 0)
        self.action_head = nn.Linear(c.gru_hidden, 2)
        self.card_head = nn.Linear(c.gru_hidden, 4)
        # One joint cell distribution, not independent row and column predictions.
        self.position_head = nn.Linear(c.gru_hidden, 32 * 18)

    def forward(self, batch):
        valid = batch["attention_mask"]
        if valid.ndim != 2 or valid.dtype != torch.bool:
            raise ValueError("attention_mask must be boolean [B,T]")
        B, T = valid.shape
        if not 1 <= T <= self.config.sequence_length:
            raise ValueError("History exceeds model sequence_length")
        if not valid.any(1).all() or ((~valid[:, :-1]) & valid[:, 1:]).any():
            raise ValueError("Every history needs real states followed only by right padding")
        ids, numeric, object_mask = object_features(batch)
        N = ids.shape[-1]
        objects = self.object_projection(torch.cat((self.unit_embedding(ids), numeric), dim=-1))
        objects = objects.reshape(B * T, N, -1)
        tokens = torch.cat((self.cls.expand(B * T, -1, -1), objects), dim=1)
        # CLS is always valid, including frames with zero objects and padded frames.
        padding = torch.cat((torch.zeros(B * T, 1, dtype=torch.bool, device=valid.device),
                             ~object_mask.reshape(B * T, N)), dim=1)
        arena = self.object_transformer(tokens, src_key_padding_mask=padding)[:, 0].reshape(B, T, -1)
        towers, hand_ids, hand_extra, globals_ = auxiliary_features(batch)
        hand = self.hand_projection(torch.cat((self.card_embedding(hand_ids), hand_extra), dim=-1)).flatten(-2)
        state = self.state_projection(torch.cat((arena, self.tower_projection(towers), hand,
                                                self.global_projection(globals_)), dim=-1))
        # Unidirectional recurrence cannot see future states; padding comes after
        # real states and its outputs are excluded by all losses and predictions.
        # Native PyTorch CUDA GRU avoids a reproduced Windows cuDNN RNN backward
        # shutdown crash (0xC0000409). Keep GPU execution; do NOT disable cuDNN
        # globally for the existing YOLO/StARformer pipelines. Backward follows
        # the native operations recorded here, after the flag is restored.
        context = torch.backends.cudnn.flags(enabled=False) if state.is_cuda else nullcontext()
        with context:
            temporal, _ = self.gru(state.masked_fill(~valid.unsqueeze(-1), 0))
        return {"action_type": self.action_head(temporal), "card_slot": self.card_head(temporal),
                "position": self.position_head(temporal)}
