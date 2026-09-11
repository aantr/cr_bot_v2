"""Discrete, factorized Implicit Q-Learning with independent history encoders.

V: expectile regression to min(target Q1, target Q2) on dataset actions.
Q: squared Bellman error r + stored_discount * V(next_history).
Actor: exp(beta * advantage)-weighted joint action log likelihood.
No maximization or sampling of unseen actions in critic targets; no RTG input.
Adapted from https://github.com/ikostrikov/implicit_q_learning (JAX reference).
"""
from copy import deepcopy
import math

import torch
from torch import nn
from torch.nn import functional as F

from .starformer import StARformer


def last_step(values, history):
    indices = torch.arange(values.shape[0], device=values.device)
    return values[indices, history["attention_mask"].sum(-1) - 1]


def expectile_loss(residual, expectile):
    return torch.where(residual > 0, expectile, 1 - expectile) * residual.square()


def bellman_target(reward, discount, next_value):
    # Discount already includes elapsed time and is 0 for terminal transitions.
    # Truncation alone does not imply termination. Do not apply gamma twice.
    return reward + torch.where(discount != 0, discount * next_value, 0)


def advantage_weights(advantage, beta, max_weight):
    # Clip BEFORE exp to prevent overflow. All policy weights are stop-gradient.
    return (beta * advantage.detach()).clamp(max=math.log(max_weight)).exp()


def action_log_probability(outputs, history, action):
    """Joint log P(type) + [play] log P(slot,row,col | history,slot).

    Card ID is determined by the hand slot, not an extra predicted action head.
    Noop has no slot/position likelihood, even if its unused fields are garbage.
    """
    logits = {key: last_step(value, history) for key, value in outputs.items()}
    result = -F.cross_entropy(logits["action_type"], action[:, 0], reduction="none")
    play = action[:, 0] == 1
    if play.any():
        slots = action[play, 2] - 1
        indices = torch.arange(slots.numel(), device=slots.device)
        extra = -F.cross_entropy(logits["card_slot"][play], slots, reduction="none")
        for name, field in (("row", 3), ("column", 4)):
            extra = extra - F.cross_entropy(logits[name + "_by_slot"][play][indices, slots],
                                             action[play, field] - 1, reduction="none")
        result = result.index_add(0, play.nonzero(as_tuple=True)[0], extra)
    return result


class HistoryBackbone(StARformer):
    def __init__(self, config):
        super().__init__(config)
        # Do not optimize unused policy heads in value networks. encode() keeps
        # the same input contract and a wholly independent set of parameters.
        for name in ("action_head", "slot_fusion", "slot_head", "row_head", "column_head"):
            delattr(self, name)

    def forward(self, history):
        context, _ = self.encode(history)
        return last_step(context, history)


class ValueNetwork(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.backbone = HistoryBackbone(config)
        self.head = nn.Sequential(nn.Linear(config.d_model, config.d_model), nn.GELU(),
                                  nn.Linear(config.d_model, 1))

    def forward(self, history):
        return self.head(self.backbone(history)).squeeze(-1)


class QNetwork(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.backbone = HistoryBackbone(config)
        self.action_type = nn.Embedding(2, config.d_model)
        self.head = nn.Sequential(nn.Linear(2 * config.d_model, config.d_model), nn.GELU(),
                                  nn.Linear(config.d_model, 1))

    def forward(self, history, action):
        context = self.backbone(history)
        fields = action[:, 1:].masked_fill((action[:, :1] == 0), 0)
        b = self.backbone
        encoded = (self.action_type(action[:, 0]) + b.card_embedding(fields[:, 0])
                   + b.slot_embedding(fields[:, 1]) + b.row_embedding(fields[:, 2])
                   + b.column_embedding(fields[:, 3]))
        return self.head(torch.cat((context, encoded), dim=-1)).squeeze(-1)


class IQL(nn.Module):
    def __init__(self, model_config, *, expectile=.7, beta=3., max_weight=100., target_rate=.005):
        super().__init__()
        if not math.isfinite(expectile) or not .5 < expectile < 1:
            raise ValueError("expectile must be finite and in (0.5, 1)")
        if not math.isfinite(beta) or beta <= 0 or not math.isfinite(max_weight) or max_weight < 1:
            raise ValueError("beta must be positive; max_weight must be >=1; both finite")
        if not math.isfinite(target_rate) or not 0 < target_rate <= 1:
            raise ValueError("target_rate must be in (0, 1]")
        self.expectile, self.beta = expectile, beta
        self.max_weight, self.target_rate = max_weight, target_rate
        self.actor = StARformer(model_config)
        self.q1, self.q2 = QNetwork(model_config), QNetwork(model_config)
        self.value = ValueNetwork(model_config)
        self.target_q1 = deepcopy(self.q1).requires_grad_(False).eval()
        self.target_q2 = deepcopy(self.q2).requires_grad_(False).eval()

    def train(self, mode=True):
        super().train(mode)
        self.target_q1.eval()
        self.target_q2.eval()
        return self

    def losses(self, batch):
        history, action = batch["history"], batch["action"]
        with torch.no_grad():
            target_q = torch.minimum(self.target_q1(history, action), self.target_q2(history, action))
            # Deterministic bootstrap even if the user enabled critic dropout.
            mode = self.value.training
            self.value.eval()
            try:
                next_value = self.value(batch["next_history"])
            finally:
                self.value.train(mode)
            target = bellman_target(batch["reward"], batch["discount"], next_value)
        value = self.value(history)
        q1, q2 = self.q1(history, action), self.q2(history, action)
        advantage = target_q - value.detach()
        weights = advantage_weights(advantage, self.beta, self.max_weight)
        outputs = self.actor(history)
        log_prob = action_log_probability(outputs, history, action)
        losses = {"value": expectile_loss(target_q - value, self.expectile).mean(),
                  "q1": F.mse_loss(q1, target), "q2": F.mse_loss(q2, target),
                  "actor": -(weights * log_prob).mean()}
        with torch.no_grad():
            logits = {key: last_step(val, history) for key, val in outputs.items()}
            predicted_type = logits["action_type"].argmax(-1)
            play = action[:, 0] == 1
            predicted_play = predicted_type == 1
            slot = logits["card_slot"].argmax(-1)
            bi = torch.arange(len(action), device=action.device)
            row = logits["row_by_slot"][bi, slot].argmax(-1)
            col = logits["column_by_slot"][bi, slot].argmax(-1)
            placement = (slot == action[:, 2] - 1) & (row == action[:, 3] - 1) & (col == action[:, 4] - 1)
            full = (predicted_type == action[:, 0]) & (~play | placement)
            metrics = {"policy_nll": -log_prob.mean(), "q_mean": ((q1 + q2) / 2).mean(),
                       "value_mean": value.mean(), "target_mean": target.mean(),
                       "advantage_mean": advantage.mean(), "weight_mean": weights.mean(),
                       "weight_clipped_fraction": (self.beta * advantage >= math.log(self.max_weight)).float().mean(),
                       "type_accuracy": (predicted_type == action[:, 0]).float().mean(),
                       "full_action_accuracy": full.float().mean(),
                       "plays": play.sum(), "predicted_plays": predicted_play.sum(),
                       "true_positive_plays": (play & predicted_play).sum(),
                       "correct_plays": (full & play).sum()}
        return losses, metrics

    @torch.no_grad()
    def update_targets(self):
        for source, target in ((self.q1, self.target_q1), (self.q2, self.target_q2)):
            for param, target_param in zip(source.parameters(), target.parameters()):
                target_param.lerp_(param, self.target_rate)
            for buffer, target_buffer in zip(source.buffers(), target.buffers()):
                target_buffer.copy_(buffer)
