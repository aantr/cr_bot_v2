"""Train Object Transformer + GRU behavior cloning on unchanged trajectory JSONs.

From v2: python offline_rl/object_gru/train.py offline_rl/trajectories --output runs/offline_rl/object_gru
Default: T=8, max_units=64, 50/50 PLAY/WAIT batches, no reward conditioning.
Split, sampling, natural validation, checkpointing and resume use the shared trainer.
"""
from dataclasses import dataclass
from functools import partial
import math
from pathlib import Path
import sys

import torch.nn.functional as F

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from offline_rl import train as shared
from offline_rl.object_gru.model import ARCHITECTURE, ObjectGRUConfig, ObjectGRUPolicy


@dataclass(frozen=True)
class ObjectTrainConfig:
    seed: int = 42
    batch_size: int = 16
    learning_rate: float = 3e-4
    weight_decay: float = .01
    grad_clip: float = 1.
    validation_fraction: float = .2
    sequence_length: int = 8
    max_units: int = 64
    stride: int = 1
    supervise: str = "last"
    sampling: str = "balanced"
    class_embedding: int = 32
    d_model: int = 128
    n_heads: int = 4
    object_layers: int = 3
    ff_multiplier: int = 4
    state_dim: int = 512
    gru_hidden: int = 256
    gru_layers: int = 2
    dropout: float = .1
    play_weight: float = 1.
    patience: int = 10
    min_delta: float = 0.

    def __post_init__(self):
        for name in ("batch_size", "sequence_length", "max_units", "stride"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        if self.seed < 0 or self.patience < 0:
            raise ValueError("seed and patience must be nonnegative")
        for name in ("learning_rate", "grad_clip", "play_weight"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("weight_decay", "min_delta"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if not math.isfinite(self.validation_fraction) or not 0 <= self.validation_fraction < 1:
            raise ValueError("validation_fraction must be in [0,1)")
        if self.supervise != "last":
            raise ValueError("Object GRU supervises only the last real state; use supervise=last")
        if self.sampling not in {"balanced", "natural"}:
            raise ValueError("sampling must be balanced or natural")
        if self.sampling == "balanced" and (self.batch_size < 2 or self.batch_size % 2):
            raise ValueError("Balanced sampling requires an even batch_size >= 2")
        ObjectGRUConfig(num_cards=3, num_units=2, sequence_length=self.sequence_length, **self.model_options())

    def model_options(self):
        return {key: getattr(self, key) for key in ("class_embedding", "d_model", "n_heads", "object_layers",
                                                   "ff_multiplier", "state_dim", "gru_hidden", "gru_layers", "dropout")}


def loss_components(outputs, batch, play_weight=1.):
    """CE(type) + .5 CE(slot|play) + .5 CE(cell|play,known position)."""
    valid = batch["loss_mask"] & batch["attention_mask"]
    if not valid.any():
        return {}
    target = batch["targets"]
    logits, types = outputs["action_type"][valid], target["action_type"][valid]
    weights = logits.new_tensor([1., play_weight])
    parts = {"action_type": (F.cross_entropy(logits, types, weight=weights, reduction="sum"), weights[types].sum())}
    play = valid & batch["play_loss_mask"] & (target["action_type"] == 1)
    if play.any():
        parts["card_slot"] = (.5 * F.cross_entropy(outputs["card_slot"][play], target["card_slot"][play],
                                                   reduction="sum"), int(play.sum()))
    position = play & batch["position_loss_mask"]
    if position.any():
        cells = target["row"][position] * 18 + target["column"][position]
        parts["position"] = (.5 * F.cross_entropy(outputs["position"][position], cells, reduction="sum"),
                              int(position.sum()))
    return parts


def make_parser():
    return shared.make_parser(ObjectTrainConfig, __doc__)


def train(args):
    return shared.train(args, config_class=ObjectTrainConfig, model_class=ObjectGRUPolicy,
                         model_config_class=ObjectGRUConfig,
                         epoch_runner=partial(shared.run_epoch, loss_function=loss_components),
                         checkpoint_metadata={"architecture": ARCHITECTURE, "history_mode": "observations_only"})


def main(argv=None):
    parser = make_parser()
    args = parser.parse_args(argv)
    try:
        train(args)
    except (ValueError, OSError) as error:
        parser.exit(2, f"Error: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
