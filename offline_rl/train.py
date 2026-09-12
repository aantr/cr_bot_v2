"""Train the StARformer imitation policy on JSON battles, not on raw videos.

Run from v2: python offline_rl/train.py offline_rl/trajectories --epochs 20
One battle: add --validation-fraction 0 (no held-out performance estimate).
Resume: python offline_rl/train.py --resume runs/offline_rl/run/last.pt --epochs 40
Epochs is the TOTAL requested epoch count, including completed epochs.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, fields
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import tempfile
import time

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from offline_rl.dataset import (Normalization, TrajectoryDataset, Vocabulary,
                                    create_train_val_datasets, discover_trajectory_files)
    from offline_rl.starformer import StARformer, StARformerConfig
    from offline_rl.balanced_sampling import BalancedActionBatchSampler
else:
    from .dataset import (Normalization, TrajectoryDataset, Vocabulary,
                          create_train_val_datasets, discover_trajectory_files)
    from .starformer import StARformer, StARformerConfig
    from .balanced_sampling import BalancedActionBatchSampler


@dataclass(frozen=True)
class TrainConfig:
    seed: int = 42
    batch_size: int = 4
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    validation_fraction: float = 0.2
    sequence_length: int = 32
    max_units: int = 128
    stride: int = 1
    supervise: str = "last"
    sampling: str = "natural"
    d_model: int = 128
    n_heads: int = 4
    local_layers: int = 1
    temporal_layers: int = 3
    ff_multiplier: int = 4
    dropout: float = 0.1
    play_weight: float = 1.0
    patience: int = 10
    min_delta: float = 0.0

    def __post_init__(self):
        for name in ("batch_size", "sequence_length", "max_units", "stride"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        if self.seed < 0 or self.patience < 0:
            raise ValueError("seed and patience must be nonnegative")
        for name in ("learning_rate", "grad_clip", "play_weight"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("weight_decay", "min_delta"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if not math.isfinite(self.validation_fraction) or not 0 <= self.validation_fraction < 1:
            raise ValueError("validation_fraction must be in [0,1)")
        if self.supervise not in {"last", "all"}:
            raise ValueError("supervise must be last or all")
        if self.sampling not in {"natural", "balanced"}:
            raise ValueError("sampling must be natural or balanced")
        if self.sampling == "balanced" and (self.batch_size < 2 or self.batch_size % 2 or self.supervise != "last"):
            raise ValueError("Balanced sampling requires even batch_size >=2 and supervise=last")
        StARformerConfig(num_cards=3, num_units=2, **self.model_options())

    def model_options(self):
        return {key: getattr(self, key) for key in (
            "d_model", "n_heads", "local_layers", "temporal_layers", "ff_multiplier", "dropout")}


def to_device(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device, non_blocking=True)
    if isinstance(value, dict):
        return {key: to_device(item, device) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(to_device(item, device) for item in value)
    return value


def loss_components(outputs, batch, play_weight=1.0):
    """Differentiable sums and denominators; empty heads are omitted, not NaN.

    CE(type) + CE(slot|play) + CE(row/col|target slot,known position).
    Epoch reporting aggregates sums/counts, not averages of unequal batches.
    """
    valid = batch["loss_mask"] & batch["attention_mask"]
    if not valid.any():
        return {}
    target = batch["targets"]
    logits = outputs["action_type"][valid]
    weights = logits.new_tensor([1.0, play_weight])
    types = target["action_type"][valid]
    result = {"action_type": (F.cross_entropy(logits, types, weight=weights, reduction="sum"),
                               weights[types].sum())}
    play = batch["play_loss_mask"] & valid & (target["action_type"] == 1)
    if play.any():
        slots = target["card_slot"][play]
        count = slots.numel()
        result["card_slot"] = (F.cross_entropy(outputs["card_slot"][play], slots, reduction="sum"), count)
    position = batch.get("position_loss_mask", play) & play
    if position.any():
        slots = target["card_slot"][position]
        count = slots.numel()
        indices = torch.arange(count, device=slots.device)
        for name in ("row", "column"):
            logits = outputs[name + "_by_slot"][position][indices, slots]
            result[name] = (F.cross_entropy(logits, target[name][position], reduction="sum"), count)
    return result


class EpochMetrics:
    """Raw, unmasked greedy heads, without game legality rules.

    Play slot/card metrics cover all labeled plays. Cell/full-action metrics
    exclude unknown positions and use the PREDICTED slot, not teacher forcing.
    """
    def __init__(self):
        self.sums, self.denominators = {}, {}
        self.counts = dict(actions=0, plays=0, positions=0, complete_actions=0, predicted_plays=0, true_positive_plays=0,
                           correct_type=0, correct_slot=0, correct_card=0, correct_cell=0,
                           correct_play=0, correct_full=0)
        self.skipped_batches = 0

    @torch.no_grad()
    def update(self, outputs, batch, components):
        for key, (value, denominator) in components.items():
            self.sums[key] = self.sums.get(key, 0.0) + float(value.detach())
            self.denominators[key] = self.denominators.get(key, 0.0) + float(denominator)
        mask = batch["loss_mask"] & batch["attention_mask"]
        target = batch["targets"]
        expected_type = target["action_type"][mask]
        predicted_type = outputs["action_type"][mask].argmax(-1)
        is_play, predicted_play = expected_type == 1, predicted_type == 1
        self.counts["actions"] += expected_type.numel()
        self.counts["predicted_plays"] += predicted_play.sum().item()
        self.counts["true_positive_plays"] += (is_play & predicted_play).sum().item()
        self.counts["correct_type"] += (expected_type == predicted_type).sum().item()
        self.counts["correct_full"] += ((~is_play) & (~predicted_play)).sum().item()
        self.counts["complete_actions"] += (~is_play).sum().item()
        play = batch["play_loss_mask"] & mask & (target["action_type"] == 1)
        count = play.sum().item()
        self.counts["plays"] += count
        if not count:
            return
        slots = outputs["card_slot"][play].argmax(-1)
        indices = torch.arange(count, device=slots.device)
        slot_ok = slots == target["card_slot"][play]
        card_ok = batch["states"]["hand_ids"][play][indices, slots] == target["card_id"][play]
        self.counts["correct_slot"] += slot_ok.sum().item()
        self.counts["correct_card"] += card_ok.sum().item()
        position = batch.get("position_loss_mask", play) & play
        count = position.sum().item()
        self.counts["positions"] += count
        self.counts["complete_actions"] += count
        if not count:
            return
        slots = outputs["card_slot"][position].argmax(-1)
        indices = torch.arange(count, device=slots.device)
        slot_ok = slots == target["card_slot"][position]
        if "position" in outputs:
            cell = outputs["position"][position].argmax(-1)
            row, column = cell // 18, cell % 18
        else:
            row = outputs["row_by_slot"][position][indices, slots].argmax(-1)
            column = outputs["column_by_slot"][position][indices, slots].argmax(-1)
        cell_ok = (row == target["row"][position]) & (column == target["column"][position])
        full = slot_ok & cell_ok & (outputs["action_type"][position].argmax(-1) == 1)
        for key, value in (("correct_cell", cell_ok), ("correct_play", full), ("correct_full", full)):
            self.counts[key] += value.sum().item()

    def summary(self):
        c = self.counts
        losses = {key: self.sums[key] / self.denominators[key] for key in self.sums}
        def ratio(numerator, denominator):
            return c[numerator] / c[denominator] if c[denominator] else None
        return {"loss": sum(losses.values()) if losses else None, "losses": losses,
                "actions": c["actions"], "plays": c["plays"], "skipped_batches": self.skipped_batches,
                "positions": c["positions"], "complete_actions": c["complete_actions"],
                "action_type_accuracy": ratio("correct_type", "actions"),
                "full_action_accuracy": ratio("correct_full", "complete_actions"),
                "play_precision": ratio("true_positive_plays", "predicted_plays"),
                "play_recall": ratio("true_positive_plays", "plays"),
                "predicted_plays": c["predicted_plays"],
                "true_positive_plays": c["true_positive_plays"],
                "false_positive_plays": c["predicted_plays"] - c["true_positive_plays"],
                "missed_plays": c["plays"] - c["true_positive_plays"],
                "play_f1": (2 * c["true_positive_plays"] / (c["plays"] + c["predicted_plays"])
                            if c["plays"] + c["predicted_plays"] else None),
                "play_slot_accuracy": ratio("correct_slot", "plays"),
                "play_card_accuracy": ratio("correct_card", "plays"),
                "play_cell_accuracy": ratio("correct_cell", "positions"),
                "play_full_accuracy": ratio("correct_play", "positions")}


def run_epoch(model, loader, device, config, *, optimizer=None, log_every=50,
              loss_function=loss_components):
    training = optimizer is not None
    model.train(training)
    metrics = EpochMetrics()
    for index, batch in enumerate(loader, 1):
        if not (batch["loss_mask"] & batch["attention_mask"]).any():
            metrics.skipped_batches += 1
            continue
        batch = to_device(batch, device)
        with torch.set_grad_enabled(training):
            outputs = model(batch)
            components = loss_function(outputs, batch, config.play_weight)
            loss = sum(value / count for value, count in components.values())
            if not torch.isfinite(loss):
                raise ValueError(f"Nonfinite loss at batch {index}; no checkpoint written for this epoch")
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip, error_if_nonfinite=True)
                optimizer.step()
        metrics.update(outputs, batch, components)
        if log_every and index % log_every == 0:
            print(f"  {'train' if training else 'val'} batch {index}/{len(loader)} "
                  f"loss={metrics.summary()['loss']:.4f}", flush=True)
    result = metrics.summary()
    if not result["actions"]:
        raise ValueError(f"{'Training' if training else 'Validation'} split has no supervised actions; "
                         "check recognition labels, stride and the split")
    return result


def play_checkpoint_score(metrics):
    """F1 at fixed raw argmax (P(play)>0.5), then lower loss as tie-breaker.

    Require both classes so all-play/no-play validation cannot pick a policy.
    Evaluated on natural, unique windows, NOT the balanced optimization stream.
    F1 concerns exact labeled decision times, not card/cell correctness or wins.
    """
    if not metrics["plays"] or metrics["plays"] == metrics["actions"]:
        return None
    return (metrics["play_f1"], -metrics["loss"])


def file_manifest(dataset):
    return [{"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            for path in dataset.files] if dataset is not None else []


def restore_datasets(checkpoint, source=None):
    manifest = checkpoint["data_manifest"]
    entries = manifest["train"] + manifest["validation"]
    if source is not None:
        if set(discover_trajectory_files(source)) != {Path(entry["path"]) for entry in entries}:
            raise ValueError("Resume source differs from saved split; start a new run for changed data")
    for entry in entries:
        path = Path(entry["path"])
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
            raise ValueError(f"Trajectory changed or missing since checkpoint: {path}")
    encoding = checkpoint["encoding_config"]
    options = {key: encoding[key] for key in ("sequence_length", "max_units", "stride", "supervise")}
    options.update(vocabulary=Vocabulary.from_dict(encoding["vocabulary"]),
                   normalization=Normalization(**encoding["normalization"]))
    train = TrajectoryDataset([entry["path"] for entry in manifest["train"]], **options)
    val = (TrajectoryDataset([entry["path"] for entry in manifest["validation"]], **options)
           if manifest["validation"] else None)
    if train.encoding_config() != encoding:
        raise ValueError("Dataset encoding no longer matches checkpoint")
    return train, val


def atomic_save(path, value, *, json_format=False):
    """Replace only our named run artifact after a complete successful write."""
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            if json_format:
                stream.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False).encode("utf-8"))
            else:
                torch.save(value, stream)
        os.replace(temporary, path)
    finally:
        if os.path.isfile(temporary):
            os.unlink(temporary)


def make_parser(config_class=TrainConfig, description=__doc__):
    parser = argparse.ArgumentParser(description=description, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source", nargs="?", type=Path, help="Trajectory JSON or directory; default offline_rl/trajectories")
    parser.add_argument("--output", type=Path, help="New empty run directory; defaults to v2/runs/offline_rl/<timestamp>")
    parser.add_argument("--resume", type=Path, help="Resume last.pt with its saved configuration and split")
    parser.add_argument("--epochs", type=int, default=20, help="Total epochs, including completed ones (default 20)")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0 or 0")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers; Windows-safe default 0")
    parser.add_argument("--num-threads", type=int, help="Optional CPU thread limit")
    parser.add_argument("--log-every", type=int, default=50, help="Batch logging interval; 0 disables")
    for field in fields(config_class):
        parser.add_argument("--" + field.name.replace("_", "-"), type=type(field.default), default=None,
                            help=f"New-run default: {field.default}; restored on resume")
    return parser


def train(args, *, config_class=TrainConfig, model_class=StARformer,
          model_config_class=StARformerConfig, epoch_runner=run_epoch,
          checkpoint_metadata=None) -> Path:
    checkpoint_metadata = dict(checkpoint_metadata or {})
    if args.epochs < 1 or args.num_workers < 0 or args.log_every < 0:
        raise ValueError("epochs must be positive; num-workers and log-every must be nonnegative")
    if args.num_threads is not None:
        if args.num_threads < 1:
            raise ValueError("num-threads must be positive")
        torch.set_num_threads(args.num_threads)
    requested = {field.name: getattr(args, field.name) for field in fields(config_class)
                 if getattr(args, field.name) is not None}
    checkpoint = None
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=True)
        if checkpoint.get("checkpoint_version") != 1:
            raise ValueError("Unsupported training checkpoint version")
        if checkpoint.get("policy_kind") != "imitation":
            raise ValueError("train.py resumes imitation only; use train_iql.py for IQL checkpoints")
        if checkpoint.get("architecture", "starformer") != checkpoint_metadata.get("architecture", "starformer"):
            raise ValueError("Checkpoint architecture differs from this trainer; use the matching training script")
        # Old imitation runs predate sampling; their original mode is natural.
        saved = {**asdict(config_class()), **checkpoint["train_config"]}
        for key, value in requested.items():
            if saved[key] != value:
                raise ValueError(f"Cannot change {key} on resume; checkpoint uses {saved[key]}")
        config = config_class(**saved)
    else:
        config = config_class(**requested)
    device_name = args.device
    if device_name == "auto":
        device_name = "cuda:0" if torch.cuda.is_available() else "cpu"
    elif device_name.isdigit():
        device_name = "cuda:" + device_name
    device = torch.device(device_name)
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("Only CPU and CUDA devices are supported")
    if device.type == "cuda" and (not torch.cuda.is_available()
                                  or (device.index or 0) >= torch.cuda.device_count()):
        raise ValueError(f"CUDA device {device} unavailable; use --device cpu")
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if checkpoint:
        train_data, val_data = restore_datasets(checkpoint, args.source)
        model_config = model_config_class(**checkpoint["model_config"])
    else:
        source = args.source or Path(__file__).resolve().parent / "trajectories"
        train_data, val_data = create_train_val_datasets(
            source, validation_fraction=config.validation_fraction, seed=config.seed,
            sequence_length=config.sequence_length, max_units=config.max_units,
            stride=config.stride, supervise=config.supervise,
        )
        model_config = model_config_class.from_encoding_config(train_data.encoding_config(), **config.model_options())
    output = (args.output.resolve() if args.output else
              args.resume.resolve().parent if args.resume else
              Path(__file__).resolve().parents[1] / "runs" / "offline_rl" / datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    if checkpoint:
        if output != args.resume.resolve().parent:
            raise ValueError("Resume must use the checkpoint's existing run directory")
        if not (output / "best.pt").is_file():
            raise ValueError("Resume needs the run's best.pt beside last.pt")
        if (output / "last.pt").is_file():
            last = torch.load(output / "last.pt", map_location="cpu", weights_only=True)
            if last["epoch"] > checkpoint["epoch"]:
                raise ValueError("Cannot rewind an existing run; resume its latest last.pt")
            del last
    elif output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError(f"Output is not empty: {output}; choose a new directory or --resume")
    output.mkdir(parents=True, exist_ok=True)
    model = model_class(model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    generator = torch.Generator().manual_seed(config.seed)
    balanced_sampler = (BalancedActionBatchSampler(train_data, config.batch_size, generator=generator)
                        if config.sampling == "balanced" else None)
    start_epoch, best_loss, best_epoch, stale_epochs, history = 1, float("inf"), 0, 0, []
    stopping_loss = float("inf")
    best_play_score, best_play_epoch = None, 0
    if checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_loss, best_epoch = checkpoint["best_loss"], checkpoint["best_epoch"]
        stopping_loss = checkpoint["stopping_loss"]
        stale_epochs, history = checkpoint["stale_epochs"], checkpoint["history"]
        if checkpoint.get("best_play_score") is not None:
            if not (output / "best_play.pt").is_file():
                raise ValueError("Resume needs the saved best_play.pt beside last.pt")
            best_play_score = tuple(checkpoint["best_play_score"])
            best_play_epoch = checkpoint["best_play_epoch"]
        random.setstate(checkpoint["rng_state"]["python"])
        torch.set_rng_state(checkpoint["rng_state"]["torch"])
        generator.set_state(checkpoint["rng_state"]["loader"])
        cuda_rng = checkpoint["rng_state"]["cuda"]
        if device.type == "cuda" and len(cuda_rng) == torch.cuda.device_count():
            torch.cuda.set_rng_state_all(cuda_rng)
    loader_options = dict(batch_size=config.batch_size, num_workers=args.num_workers, pin_memory=device.type == "cuda")
    if balanced_sampler is None:
        train_loader = DataLoader(train_data, shuffle=True, generator=generator, **loader_options)
    else:
        train_loader = DataLoader(train_data, batch_sampler=balanced_sampler, generator=generator,
                                  num_workers=args.num_workers, pin_memory=device.type == "cuda")
    val_loader = (DataLoader(val_data, shuffle=False, generator=torch.Generator().manual_seed(config.seed + 1),
                             **loader_options) if val_data is not None else None)
    # For single-battle diagnostics, measure the final epoch weights in eval
    # mode on the ORIGINAL frequency, without oversampling or dropout noise.
    train_eval_loader = (DataLoader(train_data, shuffle=False,
                                   generator=torch.Generator().manual_seed(config.seed + 2), **loader_options)
                         if val_loader is None else None)
    manifest = {"train": file_manifest(train_data), "validation": file_manifest(val_data)}
    print(f"Device: {device}; train: {len(train_data.files)} battles / {len(train_data)} windows; "
          f"validation: {len(val_data.files) if val_data else 0} battles; output: {output}", flush=True)
    if val_loader is None:
        selection = "natural train_eval loss" if balanced_sampler is not None else "training loss"
        print(f"WARNING: no validation. best.pt uses {selection}; this does not measure generalization.", flush=True)
        print("best_play.pt uses natural train_eval F1; it measures memorization, NOT generalization.", flush=True)
    if balanced_sampler is not None:
        print(f"Balanced sampling: {balanced_sampler.stats}; exactly 50% play per batch.", flush=True)
        if config.play_weight != 1:
            print("WARNING: balanced sampling already oversamples play; start with --play-weight 1.", flush=True)
    if args.epochs < start_epoch:
        print(f"Already completed {start_epoch - 1} epochs; nothing to do.", flush=True)
        return output
    if val_loader is not None and config.patience and stale_epochs >= config.patience:
        print("Saved run already reached early stopping; nothing to do.", flush=True)
        return output
    for epoch in range(start_epoch, args.epochs + 1):
        begin = time.monotonic()
        training = epoch_runner(model, train_loader, device, config, optimizer=optimizer, log_every=args.log_every)
        validation = (epoch_runner(model, val_loader, device, config, log_every=args.log_every)
                      if val_loader is not None else None)
        train_evaluation = (epoch_runner(model, train_eval_loader, device, config, log_every=0)
                            if train_eval_loader is not None else None)
        # Preserve the old natural best.pt/early-stop contract; balanced runs
        # must never select best.pt using oversampled optimization metrics.
        monitored = (validation if validation is not None else
                     train_evaluation if balanced_sampler is not None else training)
        play_metrics = validation if validation is not None else train_evaluation
        play_score = play_checkpoint_score(play_metrics)
        play_improved = play_score is not None and (best_play_score is None or play_score > best_play_score)
        if play_improved:
            best_play_score, best_play_epoch = play_score, epoch
        improved = monitored["loss"] < best_loss
        if improved:
            best_loss, best_epoch = monitored["loss"], epoch
        if monitored["loss"] < stopping_loss - config.min_delta:
            stopping_loss, stale_epochs = monitored["loss"], 0
        else:
            stale_epochs += 1
        record = {"epoch": epoch, "seconds": time.monotonic() - begin,
                  "train": training, "validation": validation, "train_eval": train_evaluation}
        history.append(record)
        state = {**checkpoint_metadata, "checkpoint_version": 1, "policy_kind": "imitation", "epoch": epoch,
                 "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
                 "model_config": model_config.to_dict(), "encoding_config": train_data.encoding_config(),
                 "train_config": asdict(config), "data_manifest": manifest,
                 "best_loss": best_loss, "best_epoch": best_epoch, "stale_epochs": stale_epochs,
                 "stopping_loss": stopping_loss,
                 "monitor": "validation_loss" if val_loader is not None else
                            "train_eval_loss" if balanced_sampler is not None else "train_loss",
                 "best_play_score": best_play_score, "best_play_epoch": best_play_epoch,
                 "play_monitor": "validation_play_f1" if val_loader is not None else "train_eval_play_f1",
                 "play_evaluation_threshold": 0.5,
                 "sampling_stats": balanced_sampler.stats if balanced_sampler is not None else None,
                 "history": history, "rng_state": {
                     "python": random.getstate(), "torch": torch.get_rng_state(),
                     "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                     "loader": generator.get_state()}}
        if improved:
            atomic_save(output / "best.pt", state)
        if play_improved:
            atomic_save(output / "best_play.pt", state)
        atomic_save(output / "last.pt", state)
        atomic_save(output / "history.json", history, json_format=True)
        val_text = f" val_loss={validation['loss']:.4f}" if validation is not None else ""
        print(f"Epoch {epoch}/{args.epochs}: train_loss={training['loss']:.4f}{val_text} "
              f"full_acc={monitored['full_action_accuracy']} "
              f"plays={monitored['plays']} play_full_acc={monitored['play_full_accuracy']} "
              f"best_epoch={best_epoch} ({record['seconds']:.1f}s)", flush=True)
        print(f"  {'val' if validation is not None else 'train_eval'}: "
              f"play_f1={play_metrics['play_f1']} precision={play_metrics['play_precision']} "
              f"recall={play_metrics['play_recall']} "
              f"missed={play_metrics['missed_plays']} false_plays={play_metrics['false_positive_plays']} "
              f"best_play_epoch={best_play_epoch}", flush=True)
        if epoch == start_epoch and play_score is None:
            print("WARNING: best_play.pt needs both play and noop labels in the evaluation split.", flush=True)
        if epoch == start_epoch and not training["plays"]:
            print("WARNING: training has no supervised plays; card and coordinate heads cannot learn.", flush=True)
        if epoch == start_epoch and validation is not None and not validation["plays"]:
            print("WARNING: validation has no supervised plays; its loss cannot evaluate placement quality.", flush=True)
        if val_loader is not None and config.patience and stale_epochs >= config.patience:
            print(f"Early stopping after {stale_epochs} epochs without sufficient improvement.", flush=True)
            break
    return output


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
