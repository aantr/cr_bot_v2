"""StARformer-inspired imitation policy for schema-v1 TrajectoryDataset batches.

This is a project-specific local-then-temporal transformer, not a reproduction
of the paper's interleaved architecture or compatible with its checkpoints.
Inputs are recognized states, previous executed actions and previous rewards.
Targets, current rewards, terminal flags and return-to-go are NEVER read.
No video recognition, optimizer, training loop or game controls live here.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import torch
from torch import nn

from .dataset import EMPTY_CARD_ID, PHASES, PREV_BOS, PREV_PLAY, PREV_UNKNOWN


@dataclass(frozen=True)
class StARformerConfig:
    num_cards: int
    num_units: int
    sequence_length: int = 32
    d_model: int = 128
    n_heads: int = 4
    local_layers: int = 1
    temporal_layers: int = 3
    ff_multiplier: int = 4
    dropout: float = 0.1

    def __post_init__(self):
        for name in ("num_cards", "num_units", "sequence_length", "d_model",
                     "n_heads", "local_layers", "temporal_layers", "ff_multiplier"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.num_cards < 3 or self.num_units < 2:
            raise ValueError("Vocabulary sizes must include reserved dataset IDs")
        if self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        if not math.isfinite(self.dropout) or not 0 <= self.dropout < 1:
            raise ValueError("dropout must be finite and in [0, 1)")

    @classmethod
    def from_encoding_config(cls, encoding: dict, **overrides):
        if encoding.get("encoding_version") != 1:
            raise ValueError("Only dataset encoding_version=1 is supported")
        values = dict(num_cards=len(encoding["vocabulary"]["cards"]),
                      num_units=len(encoding["vocabulary"]["units"]),
                      sequence_length=encoding["sequence_length"])
        values.update(overrides)
        return cls(**values)

    def to_dict(self) -> dict:
        return asdict(self)


def _encoder(config: StARformerConfig, layers: int) -> nn.TransformerEncoder:
    layer = nn.TransformerEncoderLayer(
        config.d_model, config.n_heads, config.d_model * config.ff_multiplier,
        config.dropout, activation="gelu", batch_first=True, norm_first=True,
    )
    return nn.TransformerEncoder(layer, layers, norm=nn.LayerNorm(config.d_model),
                                 enable_nested_tensor=False)


class StARformer(nn.Module):
    """forward(batch) accepts a collated dataset batch on the model's device.

    Returns unmasked logits: action_type [B,T,2], card_slot [B,T,4],
    row_by_slot [B,T,4,32], column_by_slot [B,T,4,18]. Coordinates are conditional
    on the hand slot: train using the TARGET slot, decode using the chosen slot.
    Rows/columns/slots are zero-based. Card name comes from the selected hand ID.
    Padding logits are zero; loss masks must still be applied by the trainer.
    """

    def __init__(self, config: StARformerConfig):
        super().__init__()
        self.config = config
        D = config.d_model
        self.card_embedding = nn.Embedding(config.num_cards, D, padding_idx=0)
        self.unit_embedding = nn.Embedding(config.num_units, D, padding_idx=0)
        self.side_embedding = nn.Embedding(4, D, padding_idx=0)
        # Spatial indices reserve 0 for absence, unlike zero-based state cells.
        self.row_embedding = nn.Embedding(33, D, padding_idx=0)
        self.column_embedding = nn.Embedding(19, D, padding_idx=0)
        self.slot_embedding = nn.Embedding(5, D, padding_idx=0)
        self.action_embedding = nn.Embedding(5, D, padding_idx=0)
        self.phase_embedding = nn.Embedding(len(PHASES), D, padding_idx=0)
        self.unit_features = nn.Linear(3, D)
        self.hand_features = nn.Linear(3, D)
        self.elixir_features = nn.Linear(2, D)
        self.tower_features = nn.Linear(6, D)
        self.tower_embedding = nn.Embedding(4, D)
        self.time_features = nn.Linear(4, D)
        self.reward_features = nn.Linear(2, D)
        # 4x3 patches -> 8x6=48 tokens, with # as an informational fourth channel.
        self.patch_projection = nn.Conv2d(4, D, kernel_size=(4, 3), stride=(4, 3))
        self.patch_positions = nn.Embedding(48, D)
        self.token_types = nn.Embedding(9, D)
        self.summary_token = nn.Parameter(torch.empty(1, 1, D))
        self.local_transformer = _encoder(config, config.local_layers)
        self.temporal_positions = nn.Embedding(config.sequence_length, D)
        self.temporal_transformer = _encoder(config, config.temporal_layers)
        self.action_head = nn.Linear(D, 2)
        self.slot_fusion = nn.Sequential(nn.Linear(2 * D, D), nn.GELU(), nn.LayerNorm(D))
        self.slot_head = nn.Linear(D, 1)
        self.row_head = nn.Linear(D, 32)
        self.column_head = nn.Linear(D, 18)
        # Initialize cloned encoder layers independently (including QKV weights).
        self.apply(self._initialize)
        for module in self.modules():
            if isinstance(module, nn.MultiheadAttention):
                nn.init.xavier_uniform_(module.in_proj_weight)
                nn.init.zeros_(module.in_proj_bias)
        nn.init.normal_(self.summary_token, std=0.02)

    @staticmethod
    def _initialize(module):
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)
            if module.padding_idx is not None:
                with torch.no_grad():
                    module.weight[module.padding_idx].zero_()

    @staticmethod
    def _real_steps(batch):
        real = batch["attention_mask"]
        if real.dtype != torch.bool or real.ndim != 2 or not real.shape[0] or not real.shape[1]:
            raise ValueError("attention_mask must be nonempty boolean [B,T]")
        if not real[:, 0].all() or (real[:, 1:] & ~real[:, :-1]).any():
            raise ValueError("Each sequence needs >=1 real step, followed only by right padding")
        return real

    def forward(self, batch: dict) -> dict[str, torch.Tensor]:
        real = self._real_steps(batch)
        B, T = real.shape
        if T > self.config.sequence_length:
            raise ValueError("Input exceeds configured sequence_length")
        s = batch["states"]
        D, device = self.config.d_model, real.device
        dtype = self.summary_token.dtype
        tokens, masks = [], []

        def add(value, kind, valid=None):
            count = value.shape[2]
            value = value + self.token_types.weight[kind]
            tokens.append(value.reshape(B * T, count, D))
            if valid is None:
                valid = torch.ones(B, T, count, dtype=torch.bool, device=device)
            masks.append(~valid.reshape(B * T, count))

        add(self.summary_token.view(1, 1, 1, D).expand(B, T, 1, D), 0)
        terrain = batch["terrain_mask"].to(dtype).unsqueeze(1).expand(B, T, 1, 32, 18)
        grid = torch.cat((s["grid_counts"].to(dtype), terrain), dim=2)
        patches = self.patch_projection(grid.reshape(B * T, 4, 32, 18))
        patches = patches.flatten(2).transpose(1, 2).reshape(B, T, 48, D)
        add(patches + self.patch_positions.weight, 1)

        valid_units = s["unit_mask"]
        ids = s["unit_ids"].masked_fill(~valid_units, 0)
        sides = s["unit_sides"].masked_fill(~valid_units, 0)
        cells = (s["unit_cells"] + 1).masked_fill(~valid_units.unsqueeze(-1), 0)
        features = s["unit_features"].to(dtype).masked_fill(~valid_units.unsqueeze(-1), 0)
        units = (self.unit_embedding(ids) + self.side_embedding(sides)
                 + self.row_embedding(cells[..., 0]) + self.column_embedding(cells[..., 1])
                 + self.unit_features(features))
        add(units, 2, valid_units)

        hand_meta = torch.stack((s["hand_confidence"], s["hand_known_mask"],
                                 s["hand_nonempty_mask"]), dim=-1).to(dtype)
        hand = (self.card_embedding(s["hand_ids"])
                + self.slot_embedding(torch.arange(1, 5, device=device))
                + self.hand_features(hand_meta))
        add(hand, 3)
        known = s["elixir_mask"]
        elixir = torch.cat((s["elixir"].masked_fill(~known, 0), known), dim=-1).to(dtype)
        add(self.elixir_features(elixir).unsqueeze(2), 4)
        known = s["tower_hp_known_mask"]
        towers = torch.stack((s["tower_hp"].masked_fill(~known, 0), known,
                              s["tower_hp_fresh_mask"] & known,
                              s["tower_hp_confidence"].masked_fill(~known, 0),
                              s["tower_hp_age"].masked_fill(~known, 0), ~known), dim=-1).to(dtype)
        add(self.tower_features(towers) + self.tower_embedding.weight, 5)
        time = s["time_features"].to(dtype).clone()
        time[..., 2].masked_fill_(~s["remaining_time_mask"], 0)
        time = torch.cat((time, s["remaining_time_mask"].unsqueeze(-1)), dim=-1).to(dtype)
        add((self.time_features(time) + self.phase_embedding(s["phase_ids"])).unsqueeze(2), 6)

        action = batch["previous_actions"]
        action_type = action[..., 0].clone()
        invalid = ~batch["previous_action_valid"] & (action_type != PREV_BOS)
        action_type.masked_fill_(invalid, PREV_UNKNOWN)
        fields = action[..., 1:].masked_fill((action_type != PREV_PLAY).unsqueeze(-1), 0)
        previous = (self.action_embedding(action_type) + self.card_embedding(fields[..., 0])
                    + self.slot_embedding(fields[..., 1]) + self.row_embedding(fields[..., 2])
                    + self.column_embedding(fields[..., 3]))
        add(previous.unsqueeze(2), 7)
        known = batch["previous_reward_mask"].unsqueeze(-1)
        reward = torch.cat((batch["previous_rewards"].masked_fill(~known, 0), known), dim=-1).to(dtype)
        add(self.reward_features(reward).unsqueeze(2), 8)

        local = self.local_transformer(torch.cat(tokens, dim=1),
                                       src_key_padding_mask=torch.cat(masks, dim=1))
        steps = local[:, 0].reshape(B, T, D)
        steps = steps + self.temporal_positions(torch.arange(T, device=device))
        # Construct our own causal mask: callers cannot accidentally enable future attention.
        causal = torch.ones(T, T, dtype=torch.bool, device=device).triu(1)
        context = self.temporal_transformer(steps, mask=causal, src_key_padding_mask=~real)
        per_slot = self.slot_fusion(torch.cat((context.unsqueeze(2).expand(-1, -1, 4, -1),
                                               hand), dim=-1))
        outputs = {"action_type": self.action_head(context),
                   "card_slot": self.slot_head(per_slot).squeeze(-1),
                   "row_by_slot": self.row_head(per_slot),
                   "column_by_slot": self.column_head(per_slot)}
        for name, value in outputs.items():
            outputs[name] = value.masked_fill(~real.reshape(B, T, *([1] * (value.ndim - 2))), 0)
        return outputs

    @torch.no_grad()
    def predict(self, batch: dict, *, allowed_slots=None, allowed_cells=None) -> dict:
        """Greedy last-real-step recommendation; call model.eval() first.

        Optional boolean allowed_slots [B,4], allowed_cells [B,4,32,18]
        must be on the same device. Cells may depend on the selected slot.
        Empty/unknown hand slots are excluded; no legal play falls back to noop.
        This does NOT infer card costs/placement rules from the terrain (#).
        Non-play fields are -1. Nothing is executed in the game.
        """
        if self.training:
            raise RuntimeError("Call model.eval() before predict()")
        outputs = self(batch)
        real = self._real_steps(batch)
        B = real.shape[0]
        bi = torch.arange(B, device=real.device)
        ti = real.sum(dim=1) - 1
        hand = batch["states"]["hand_ids"][bi, ti]
        slots = hand > EMPTY_CARD_ID
        cells = torch.ones(B, 4, 32, 18, dtype=torch.bool, device=real.device)
        for name, mask, shape in (("allowed_slots", allowed_slots, (B, 4)),
                                  ("allowed_cells", allowed_cells, (B, 4, 32, 18))):
            if mask is not None and (mask.dtype != torch.bool or tuple(mask.shape) != shape
                                     or mask.device != real.device):
                raise ValueError(f"{name} must be boolean {shape} on {real.device}")
        if allowed_slots is not None:
            slots = slots & allowed_slots
        if allowed_cells is not None:
            cells = cells & allowed_cells
        slots = slots & cells.flatten(2).any(dim=-1)
        play = (outputs["action_type"][bi, ti].argmax(-1) == 1) & slots.any(-1)
        slot = outputs["card_slot"][bi, ti].masked_fill(~slots, -torch.inf).argmax(-1)
        rows = outputs["row_by_slot"][bi, ti, slot]
        columns = outputs["column_by_slot"][bi, ti, slot]
        joint = rows.unsqueeze(-1) + columns.unsqueeze(-2)
        cell = joint.masked_fill(~cells[bi, slot], -torch.inf).flatten(1).argmax(-1)
        absent = torch.full_like(slot, -1)
        return {"action_type": play.long(), "card_slot": torch.where(play, slot, absent),
                "card_id": torch.where(play, hand[bi, slot], absent),
                "row": torch.where(play, cell // 18, absent),
                "column": torch.where(play, cell % 18, absent)}
