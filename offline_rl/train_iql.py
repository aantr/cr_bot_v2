"""Train discrete IQL from battle JSONs; no simulator or live game required.

From v2: python offline_rl/train_iql.py offline_rl/trajectories --epochs 50
One battle: add --validation-fraction 0. Resume: --resume RUN/last.pt --epochs 100
best.pt is selected by unweighted policy NLL, NOT estimated win rate.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, fields
from datetime import datetime
import math
from pathlib import Path
import random
import sys
import time

import torch
from torch.utils.data import DataLoader

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from offline_rl.dataset import create_train_val_datasets
    from offline_rl.iql import IQL
    from offline_rl.iql_dataset import HISTORY_MODE, IQLDataset
    from offline_rl.starformer import StARformerConfig
    from offline_rl.train import TrainConfig, atomic_save, file_manifest, restore_datasets, to_device
else:
    from .dataset import create_train_val_datasets
    from .iql import IQL
    from .iql_dataset import HISTORY_MODE, IQLDataset
    from .starformer import StARformerConfig
    from .train import TrainConfig, atomic_save, file_manifest, restore_datasets, to_device


@dataclass(frozen=True)
class IQLTrainConfig:
    seed: int = 42
    batch_size: int = 2
    learning_rate: float = 3e-4
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    validation_fraction: float = 0.2
    sequence_length: int = 32
    max_units: int = 128
    d_model: int = 128
    n_heads: int = 4
    local_layers: int = 1
    temporal_layers: int = 3
    ff_multiplier: int = 4
    dropout: float = 0.0
    expectile: float = 0.7
    beta: float = 3.0
    max_weight: float = 100.0
    target_rate: float = 0.005
    patience: int = 0
    min_delta: float = 0.0

    def __post_init__(self):
        shared = {f.name for f in fields(TrainConfig)}
        TrainConfig(**{k: v for k, v in asdict(self).items() if k in shared})
        if not math.isfinite(self.expectile) or not .5 < self.expectile < 1:
            raise ValueError("expectile must be in (0.5, 1)")
        if not math.isfinite(self.beta) or self.beta <= 0:
            raise ValueError("beta must be finite and positive")
        if not math.isfinite(self.max_weight) or self.max_weight < 1:
            raise ValueError("max-weight must be finite and >=1")
        if not math.isfinite(self.target_rate) or not 0 < self.target_rate <= 1:
            raise ValueError("target-rate must be in (0, 1]")

    def model_options(self):
        return {key: getattr(self, key) for key in (
            "d_model", "n_heads", "local_layers", "temporal_layers", "ff_multiplier", "dropout")}


def make_optimizers(model, config):
    return {name: torch.optim.AdamW(getattr(model, name).parameters(),
                                    lr=config.learning_rate, weight_decay=config.weight_decay)
            for name in ("actor", "value", "q1", "q2")}


def run_epoch(model, loader, device, config, *, optimizers=None, log_every=50):
    training = optimizers is not None
    model.train(training)
    sums, count = {}, 0
    counters = {"plays", "predicted_plays", "true_positive_plays", "correct_plays"}
    for index, batch in enumerate(loader, 1):
        batch = to_device(batch, device)
        n = batch["action"].shape[0]
        with torch.set_grad_enabled(training):
            losses, metrics = model.losses(batch)
            if any(not torch.isfinite(loss) for loss in losses.values()):
                raise ValueError(f"Nonfinite IQL loss at batch {index}; epoch not checkpointed")
            if training:
                for optimizer in optimizers.values():
                    optimizer.zero_grad(set_to_none=True)
                # Graphs are independent: Bellman targets and advantage weights
                # are detached, so actor gradients cannot change Q or V.
                sum(losses.values()).backward()
                for name in optimizers:
                    torch.nn.utils.clip_grad_norm_(getattr(model, name).parameters(),
                                                   config.grad_clip, error_if_nonfinite=True)
                for optimizer in optimizers.values():
                    optimizer.step()
                model.update_targets()
        for key, value in {**{name + "_loss": loss for name, loss in losses.items()}, **metrics}.items():
            number = float(value.detach())
            if not math.isfinite(number):
                raise ValueError(f"Nonfinite IQL metric {key} at batch {index}")
            sums[key] = sums.get(key, 0.) + number * (1 if key in counters else n)
        count += n
        if log_every and index % log_every == 0:
            print(f"  {'train' if training else 'val'} {index}/{len(loader)} "
                  f"policy_nll={sums['policy_nll'] / count:.4f}", flush=True)
    if not count:
        raise ValueError("IQL split has no fully known actions")
    result = {key: int(value) if key in counters else value / count for key, value in sums.items()}
    result["transitions"] = count
    result["play_recall"] = result["true_positive_plays"] / result["plays"] if result["plays"] else None
    result["play_precision"] = (result["true_positive_plays"] / result["predicted_plays"]
                                if result["predicted_plays"] else None)
    result["play_full_accuracy"] = result["correct_plays"] / result["plays"] if result["plays"] else None
    return result


def make_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source", nargs="?", type=Path)
    parser.add_argument("--output", type=Path, help="New empty run directory")
    parser.add_argument("--resume", type=Path, help="IQL last.pt (restores all networks, optimizers and RNG)")
    parser.add_argument("--init-actor", type=Path, help="Optional imitation/IQL actor warm start; identical encoding required")
    parser.add_argument("--epochs", type=int, default=50, help="Total epochs including completed epochs")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0 or 0")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--num-threads", type=int)
    parser.add_argument("--log-every", type=int, default=50)
    for field in fields(IQLTrainConfig):
        parser.add_argument("--" + field.name.replace("_", "-"), type=type(field.default), default=None,
                            help=f"New-run default: {field.default}; restored on resume")
    return parser


def train(args):
    if args.epochs < 1 or args.num_workers < 0 or args.log_every < 0:
        raise ValueError("epochs must be positive; workers/log-every must be nonnegative")
    if args.resume and args.init_actor:
        raise ValueError("Use either --resume or --init-actor, not both")
    if args.num_threads is not None:
        if args.num_threads < 1:
            raise ValueError("num-threads must be positive")
        torch.set_num_threads(args.num_threads)
    requested = {f.name: getattr(args, f.name) for f in fields(IQLTrainConfig) if getattr(args, f.name) is not None}
    checkpoint = torch.load(args.resume, map_location="cpu", weights_only=True) if args.resume else None
    initial = torch.load(args.init_actor, map_location="cpu", weights_only=True) if args.init_actor else None
    if checkpoint:
        if (checkpoint.get("checkpoint_version") != 1 or checkpoint.get("policy_kind") != "iql"
                or checkpoint.get("iql_version") != 1 or checkpoint.get("history_mode") != HISTORY_MODE):
            raise ValueError("Resume requires a supported IQL checkpoint; imitation uses --init-actor")
        for key, value in requested.items():
            if checkpoint["train_config"][key] != value:
                raise ValueError(f"Cannot change {key} on resume; start a new run")
        config = IQLTrainConfig(**checkpoint["train_config"])
    else:
        if initial:
            if initial.get("checkpoint_version") != 1 or initial.get("policy_kind") not in {"imitation", "iql"}:
                raise ValueError("Unsupported initial actor checkpoint")
            inherited = {key: value for key, value in initial["model_config"].items()
                         if key not in {"num_cards", "num_units"}}
            inherited["max_units"] = initial["encoding_config"]["max_units"]
            for key in inherited:
                if key in requested and requested[key] != inherited[key]:
                    raise ValueError(f"--init-actor requires {key}={inherited[key]}")
            requested = {**inherited, **requested}
        config = IQLTrainConfig(**requested)
    device_name = args.device
    if device_name == "auto":
        device_name = "cuda:0" if torch.cuda.is_available() else "cpu"
    elif device_name.isdigit():
        device_name = "cuda:" + device_name
    device = torch.device(device_name)
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("Only CPU and CUDA are supported")
    if device.type == "cuda" and (not torch.cuda.is_available() or (device.index or 0) >= torch.cuda.device_count()):
        raise ValueError(f"CUDA device {device} unavailable; use --device cpu")
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if checkpoint:
        train_base, val_base = restore_datasets(checkpoint, args.source)
        model_config = StARformerConfig(**checkpoint["model_config"])
    else:
        train_base, val_base = create_train_val_datasets(
            args.source or Path(__file__).resolve().parent / "trajectories",
            validation_fraction=config.validation_fraction, seed=config.seed,
            sequence_length=config.sequence_length, max_units=config.max_units, stride=1, supervise="last")
        model_config = StARformerConfig.from_encoding_config(train_base.encoding_config(), **config.model_options())
    if initial:
        expected, actual = initial["encoding_config"], train_base.encoding_config()
        if any(expected[k] != actual[k] for k in actual if k not in {"stride", "supervise"}):
            raise ValueError("--init-actor vocabulary/normalization/encoding differs; train IQL from scratch")
        if initial.get("history_mode", "executed_feedback") != HISTORY_MODE:
            print("NOTE: actor warm start switches to observations_only history; adaptation is required.", flush=True)
    train_data = IQLDataset(train_base)
    val_data = IQLDataset(val_base) if val_base is not None else None
    for name, dataset in (("train", train_data), ("validation", val_data)):
        if dataset is None:
            continue
        print(f"IQL {name} data: {dataset.stats}", flush=True)
        if not len(dataset):
            raise ValueError(f"{name} has no fully known IQL transitions; repair recognition labels")
        if not dataset.stats["play"]:
            print(f"WARNING: {name} has no complete plays; card/placement learning cannot be evaluated.", flush=True)
        if not dataset.stats["nonzero_rewards_kept"]:
            print(f"WARNING: {name} has no retained nonzero rewards; this data supplies no reward preference.", flush=True)
        if dataset.stats["terminal_excluded"] or dataset.stats["nonzero_rewards_excluded"]:
            print(f"WARNING: {name} excludes {dataset.stats['terminal_excluded']} terminal and "
                  f"{dataset.stats['nonzero_rewards_excluded']} nonzero-reward transitions with unknown actions. "
                  "Their rewards are NOT reassigned to earlier actions.", flush=True)
    output = (args.output.resolve() if args.output else args.resume.resolve().parent if args.resume else
              Path(__file__).resolve().parents[1] / "runs" / "offline_rl" / ("iql_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f")))
    if checkpoint:
        if output != args.resume.resolve().parent or not (output / "best.pt").is_file():
            raise ValueError("Resume needs the original run directory and its best.pt")
        if (output / "last.pt").is_file():
            last = torch.load(output / "last.pt", map_location="cpu", weights_only=True)
            if last["epoch"] > checkpoint["epoch"]:
                raise ValueError("Cannot rewind a run; resume its latest last.pt")
            del last
    elif output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError(f"Output is not empty: {output}; choose a new directory or --resume")
    output.mkdir(parents=True, exist_ok=True)
    model = IQL(model_config, expectile=config.expectile, beta=config.beta,
                max_weight=config.max_weight, target_rate=config.target_rate).to(device)
    if initial:
        model.actor.load_state_dict(initial["model_state_dict"], strict=True)
        del initial
    optimizers = make_optimizers(model, config)
    generator = torch.Generator().manual_seed(config.seed)
    start, best_loss, best_epoch, stopping_loss, stale, history = 1, float("inf"), 0, float("inf"), 0, []
    if checkpoint:
        model.load_state_dict(checkpoint["iql_state_dict"], strict=True)
        for name, optimizer in optimizers.items():
            optimizer.load_state_dict(checkpoint["optimizer_state_dicts"][name])
        start = checkpoint["epoch"] + 1
        best_loss, best_epoch = checkpoint["best_loss"], checkpoint["best_epoch"]
        stopping_loss, stale, history = checkpoint["stopping_loss"], checkpoint["stale_epochs"], checkpoint["history"]
        rng = checkpoint["rng_state"]
        random.setstate(rng["python"])
        torch.set_rng_state(rng["torch"])
        generator.set_state(rng["loader"])
        if device.type == "cuda" and len(rng["cuda"]) == torch.cuda.device_count():
            torch.cuda.set_rng_state_all(rng["cuda"])
        del checkpoint
    options = dict(batch_size=config.batch_size, num_workers=args.num_workers, pin_memory=device.type == "cuda")
    train_loader = DataLoader(train_data, shuffle=True, generator=generator, **options)
    val_loader = (DataLoader(val_data, shuffle=False, generator=torch.Generator().manual_seed(config.seed + 1),
                             **options) if val_data is not None else None)
    manifest = {"train": file_manifest(train_base), "validation": file_manifest(val_base)}
    stats = {"train": train_data.stats, "validation": val_data.stats if val_data is not None else None}
    print(f"Device: {device}; IQL train: {len(train_base.files)} battles / {len(train_data)} transitions; output: {output}", flush=True)
    print("NOTE: best.pt uses unweighted policy NLL, not win rate or an offline policy-value estimate.", flush=True)
    if val_loader is None:
        print("WARNING: no validation; best.pt uses training NLL and does not measure generalization.", flush=True)
    if args.epochs < start or (val_loader is not None and config.patience and stale >= config.patience):
        print("Requested epochs or saved early-stopping limit already reached; nothing to do.", flush=True)
        return output
    for epoch in range(start, args.epochs + 1):
        begin = time.monotonic()
        training = run_epoch(model, train_loader, device, config, optimizers=optimizers, log_every=args.log_every)
        validation = run_epoch(model, val_loader, device, config, log_every=args.log_every) if val_loader is not None else None
        monitored = validation if validation is not None else training
        score = monitored["policy_nll"]
        improved = score < best_loss
        if improved:
            best_loss, best_epoch = score, epoch
        if score < stopping_loss - config.min_delta:
            stopping_loss, stale = score, 0
        else:
            stale += 1
        record = {"epoch": epoch, "seconds": time.monotonic() - begin, "train": training, "validation": validation}
        history.append(record)
        state = {"checkpoint_version": 1, "iql_version": 1, "policy_kind": "iql", "history_mode": HISTORY_MODE,
                 "epoch": epoch, "model_config": model_config.to_dict(), "encoding_config": train_base.encoding_config(),
                 "model_state_dict": model.actor.state_dict(), "iql_state_dict": model.state_dict(),
                 "optimizer_state_dicts": {key: opt.state_dict() for key, opt in optimizers.items()},
                 "train_config": asdict(config), "data_manifest": manifest, "data_stats": stats,
                 "best_loss": best_loss, "best_epoch": best_epoch, "stopping_loss": stopping_loss, "stale_epochs": stale,
                 "monitor": "validation_policy_nll" if val_loader is not None else "train_policy_nll",
                 "history": history, "rng_state": {"python": random.getstate(), "torch": torch.get_rng_state(),
                     "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [], "loader": generator.get_state()}}
        if improved:
            atomic_save(output / "best.pt", state)
        atomic_save(output / "last.pt", state)
        atomic_save(output / "history.json", history, json_format=True)
        print(f"Epoch {epoch}/{args.epochs}: actor={training['actor_loss']:.4f} "
              f"Q1={training['q1_loss']:.4f} Q2={training['q2_loss']:.4f} V={training['value_loss']:.4f} "
              f"{'val' if validation is not None else 'train'}_nll={score:.4f} "
              f"play_recall={monitored['play_recall']} best_epoch={best_epoch} ({record['seconds']:.1f}s)", flush=True)
        if val_loader is not None and config.patience and stale >= config.patience:
            print(f"Early stopping after {stale} epochs without sufficient NLL improvement.", flush=True)
            break
    return output


def main(argv=None):
    parser = make_parser()
    try:
        train(parser.parse_args(argv))
    except (ValueError, OSError) as error:
        parser.exit(2, f"Error: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
