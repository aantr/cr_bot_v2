"""PyTorch windows over schema-v1 JSON battles produced by build_trajectories.py.

Each sample ends at one transition, contains only that battle's past states,
and is RIGHT-padded. Default supervision covers the final real step only.
Boolean attention_mask means real timestep; padding_mask/causal_mask mean BLOCK.
Targets and return_to_go are separate from state inputs; the latter is an
optional return-conditioning signal, never a regular observed state feature.
"""
from __future__ import annotations

import argparse
from bisect import bisect_right
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import random
from typing import Iterable

import torch
from torch.utils.data import Dataset


PAD_ID, UNKNOWN_ID, EMPTY_CARD_ID = 0, 1, 2
IGNORE_INDEX = -100
PREV_PAD, PREV_BOS, PREV_UNKNOWN, PREV_NOOP, PREV_PLAY = range(5)
TOWERS = ("ally_1", "ally_2", "enemy_1", "enemy_2")
GRID_SIDES = ("ally", "enemy", "unknown")
SIDE_IDS = {"unknown": 1, "ally": 2, "enemy": 3}
PHASES = ("<pad>", "unknown", "normal", "double_elixir", "triple_elixir", "overtime", "end")
CARD_SPECIALS = ("<pad>", "<unknown>", "empty")
UNIT_SPECIALS = ("<pad>", "<unknown>")


def label(value: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"Class name must be a string, got {value!r}")
    value = value.strip().lower()
    return "<unknown>" if value in {"", "none", "unknown", "<unknown>", "<pad>"} else value


def discover_trajectory_files(source: str | Path | Iterable[str | Path]) -> list[Path]:
    sources = [source] if isinstance(source, (str, Path)) else list(source)
    files = set()
    for item in sources:
        path = Path(item).expanduser().resolve()
        if path.is_dir():
            files.update(path.glob("*.json"))
        elif path.is_file():
            files.add(path)
        else:
            raise FileNotFoundError(f"Trajectory path not found: {path}")
    if not files:
        raise ValueError("No trajectory JSON files found")
    return sorted(files)


def _number(value, where: str, low=None, high=None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{where}: expected a finite number")
    if low is not None and value < low or high is not None and value > high:
        raise ValueError(f"{where}: value {value} outside [{low}, {high}]")
    return float(value)


def _integer(value, where: str, low: int, high: int) -> int:
    _number(value, where, low, high)
    if not isinstance(value, int):
        raise ValueError(f"{where}: expected an integer")
    return value


def _boolean(value, where: str) -> None:
    if not isinstance(value, bool):
        raise ValueError(f"{where}: expected true or false")


def validate_observation(state: dict) -> None:
    """Validate one recognized state without requiring future observations."""
    timestamp = _number(state["timestamp_ms"], "timestamp_ms", 0)
    hand = state["hand"]
    if len(hand) != 4 or sorted(item["slot"] for item in hand) != [1, 2, 3, 4]:
        raise ValueError("Expected four unique hand slots 1..4")
    for card in hand:
        _integer(card["slot"], "hand slot", 1, 4)
        label(card["card"])
        _number(card["confidence"], "hand confidence", 0, 1)
    for unit in state["units"]:
        label(unit["unit"])
        _integer(unit["row"], "unit row", 1, 32)
        _integer(unit["column"], "unit column", 1, 18)
        if unit["side"] not in SIDE_IDS:
            raise ValueError(f"Unknown unit side: {unit['side']}")
        _number(unit["unit_confidence"], "unit confidence", 0, 1)
        _number(unit["detector_confidence"], "detector confidence", 0, 1)
        _boolean(unit["predicted_by_kalman"], "predicted_by_kalman")
    elixir = state["elixir"].get("value")
    if elixir is not None:
        _number(elixir, "elixir", 0, 10)
    if state.get("game_time_remaining_seconds") is not None:
        _number(state["game_time_remaining_seconds"], "remaining time", 0)
    for tower in TOWERS:
        hp = state["tower_hp"].get(tower, {})
        if hp.get("hp") is not None:
            _number(hp["hp"], f"{tower} HP", 0)
            _number(hp.get("confidence", 0), f"{tower} confidence", 0, 1)
            _boolean(hp.get("stale", True), f"{tower} stale")
            for key in ("confirmed_at_ms", "observed_at_ms", "last_seen_at_ms"):
                if hp.get(key) is not None:
                    _number(hp[key], f"{tower} {key}", 0, timestamp)


def validate_trajectory(data: dict) -> None:
    """Fail on structural errors instead of silently shifting state/action labels."""
    if data["schema_version"] != 1:
        raise ValueError("Only trajectory schema_version=1 is supported")
    coordinates = data["coordinates"]
    if (coordinates["rows"], coordinates["columns"], coordinates["index_base"],
            coordinates["origin"]) != (32, 18, 1, "top-left"):
        raise ValueError("Expected 32x18 top-left, 1-based source coordinates")
    observations, transitions = data["observations"], data["transitions"]
    if not transitions or len(observations) != len(transitions) + 1:
        raise ValueError("Expected N+1 observations and N nonempty transitions")
    terrain = data["metadata"]["field_layout"]
    if (len(terrain) != 32 or any(not isinstance(row, str) or len(row) != 18
                                or set(row) - {".", "#"} for row in terrain)):
        raise ValueError("metadata.field_layout must be 32 strings of 18 '.'/'#' cells")
    last_time = -1.0
    for index, state in enumerate(observations):
        timestamp = _number(state["timestamp_ms"], f"state {index} timestamp_ms", 0)
        if timestamp <= last_time:
            raise ValueError("Observation timestamps must be strictly increasing")
        last_time = timestamp
        validate_observation(state)
    for index, transition in enumerate(transitions):
        if (transition["state_index"], transition["next_state_index"]) != (index, index + 1):
            raise ValueError(f"transition {index}: indices must reference adjacent observations")
        _boolean(transition["action_valid"], "action_valid")
        _boolean(transition["terminated"], "terminated")
        _boolean(transition["truncated"], "truncated")
        done = transition["terminated"] or transition["truncated"]
        if done != (index == len(transitions) - 1):
            raise ValueError("Only the last transition may terminate/truncate an episode")
        expected_dt = observations[index + 1]["timestamp_ms"] - observations[index]["timestamp_ms"]
        if not math.isclose(_number(transition["dt_ms"], "dt_ms", 0), expected_dt,
                            rel_tol=1e-6, abs_tol=1e-3):
            raise ValueError(f"transition {index}: dt_ms does not match timestamps")
        for key in ("reward", "return_to_go", "discounted_return_to_go"):
            _number(transition[key], key)
        _number(transition["discount"], "discount", 0, 1)
        if transition["terminated"] and transition["discount"] != 0:
            raise ValueError("Terminated transition must have zero bootstrap discount")
        action = transition["action"]
        if transition["action_valid"]:
            if action["type"] == "play":
                _boolean(action.get("position_valid", True), "position_valid")
                _integer(action["slot"], "action slot", 1, 4)
                _integer(action["row"], "action row", 1, 32)
                _integer(action["column"], "action column", 1, 18)
                if label(action["card"]) in CARD_SPECIALS:
                    raise ValueError("Valid play must have a known, nonempty card")
                hand_card = next(c for c in observations[index]["hand"] if c["slot"] == action["slot"])
                if label(hand_card["card"]) != label(action["card"]):
                    raise ValueError("Valid play card differs from pre-action hand")
                if "timestamp_ms" in action:
                    _number(action["timestamp_ms"], "action timestamp")
                    if not (observations[index]["timestamp_ms"] < action["timestamp_ms"]
                            <= observations[index + 1]["timestamp_ms"]):
                        raise ValueError("Valid play must occur after its input state")
            elif action["type"] != "noop":
                raise ValueError("Only play/noop can have action_valid=true")


def load_trajectory(path: Path) -> dict:
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
        validate_trajectory(data)
        return data
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"Invalid trajectory {path}: {error}") from error


def battle_identity(data: dict) -> str:
    metadata = data["metadata"]
    identity = metadata.get("source_sha256") or metadata.get("battle_id")
    if identity:
        return str(identity)
    # Older files without source identity can still be grouped if copied verbatim.
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class Vocabulary:
    """Shared IDs fit on training files only; unseen validation names map to UNK."""
    cards: tuple[str, ...]
    units: tuple[str, ...]

    def __post_init__(self):
        for name, specials in (("cards", CARD_SPECIALS), ("units", UNIT_SPECIALS)):
            values = getattr(self, name)
            if (tuple(values[:len(specials)]) != specials or len(set(values)) != len(values)
                    or any(not isinstance(value, str) or not value for value in values)):
                raise ValueError(f"Invalid {name} vocabulary or reserved IDs")

    @classmethod
    def from_trajectories(cls, trajectories: Iterable[dict]) -> Vocabulary:
        cards, units = set(), set()
        for data in trajectories:
            model_vocab = data["metadata"].get("vocabulary", {})
            for key, destination in (("cards", cards), ("units", units)):
                names = model_vocab.get(key, {})
                destination.update(label(name) for name in
                                   (names.values() if isinstance(names, dict) else names))
            for state in data["observations"]:
                cards.update(label(c["card"]) for c in state["hand"])
                units.update(label(u["unit"]) for u in state["units"])
        return cls(CARD_SPECIALS + tuple(sorted(cards - set(CARD_SPECIALS))),
                   UNIT_SPECIALS + tuple(sorted(units - set(UNIT_SPECIALS))))

    def to_dict(self) -> dict:
        return {"cards": list(self.cards), "units": list(self.units)}

    @classmethod
    def from_dict(cls, data: dict) -> Vocabulary:
        return cls(tuple(data["cards"]), tuple(data["units"]))


@dataclass(frozen=True)
class Normalization:
    """Fixed scales shared by training/validation/inference; no future fitting."""
    elixir: float = 10.0
    tower_hp: float = 10000.0
    unit_count: float = 10.0
    time_seconds: float = 300.0
    hp_age_seconds: float = 10.0
    reward: float = 5.0
    return_to_go: float = 5.0

    def __post_init__(self):
        for name, value in asdict(self).items():
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"Normalization.{name} must be finite and positive")


class ObservationEncoder:
    """Shared offline/live feature encoding; never needs the next observation.

    previous_transitions[t] describes the ACTUAL transition into observation t;
    None means episode start. A reward of None means unavailable, not zero.
    Source coordinates are 1-based, exactly as in build_trajectories.py.
    Training-only fields are initialized empty and populated by the dataset.
    """

    def __init__(self, vocabulary, normalization, sequence_length=32, max_units=128):
        self.vocabulary, self.normalization = vocabulary, normalization
        self.sequence_length, self.max_units = sequence_length, max_units
        self.card_to_id = {name: i for i, name in enumerate(vocabulary.cards)}
        self.unit_to_id = {name: i for i, name in enumerate(vocabulary.units)}

    def _action(self, transition: dict) -> tuple[list[int], bool]:
        if not transition["action_valid"]:
            return [PREV_UNKNOWN, 0, 0, 0, 0], False
        action = transition["action"]
        if action["type"] == "noop":
            return [PREV_NOOP, 0, 0, 0, 0], True
        card_id = self.card_to_id.get(label(action["card"]), UNKNOWN_ID)
        if card_id <= EMPTY_CARD_ID:
            return [PREV_UNKNOWN, 0, 0, 0, 0], False
        row, column = (action["row"], action["column"]) if action.get("position_valid", True) else (0, 0)
        return [PREV_PLAY, card_id, action["slot"], row, column], True

    def encode(self, observations, previous_transitions, previous_timestamps,
               step_indices, field_layout, battle_id="live"):
        length = len(observations)
        if not 1 <= length <= self.sequence_length:
            raise ValueError("Observation window must have 1..sequence_length states")
        if any(len(items) != length for items in
               (previous_transitions, previous_timestamps, step_indices)):
            raise ValueError("Previous transition/time/step arrays must match observations")
        if any(len(state["units"]) > self.max_units for state in observations):
            raise ValueError("Observed units exceed checkpoint max_units; units are never dropped")
        T, M = self.sequence_length, self.max_units
        norm = self.normalization
        states = {
            "grid_counts": torch.zeros(T, 3, 32, 18),
            "unit_ids": torch.zeros(T, M, dtype=torch.long),
            "unit_sides": torch.zeros(T, M, dtype=torch.long),
            "unit_cells": torch.zeros(T, M, 2, dtype=torch.long),
            "unit_features": torch.zeros(T, M, 3),
            "unit_mask": torch.zeros(T, M, dtype=torch.bool),
            "hand_ids": torch.zeros(T, 4, dtype=torch.long),
            "hand_confidence": torch.zeros(T, 4),
            "hand_known_mask": torch.zeros(T, 4, dtype=torch.bool),
            "hand_nonempty_mask": torch.zeros(T, 4, dtype=torch.bool),
            "elixir": torch.zeros(T, 1),
            "elixir_mask": torch.zeros(T, 1, dtype=torch.bool),
            "tower_hp": torch.zeros(T, 4),
            "tower_hp_known_mask": torch.zeros(T, 4, dtype=torch.bool),
            "tower_hp_fresh_mask": torch.zeros(T, 4, dtype=torch.bool),
            "tower_hp_confidence": torch.zeros(T, 4),
            "tower_hp_age": torch.zeros(T, 4),
            "time_features": torch.zeros(T, 3),
            "remaining_time_mask": torch.zeros(T, dtype=torch.bool),
            "phase_ids": torch.zeros(T, dtype=torch.long),
        }
        sample = {
            "states": states,
            "terrain_mask": torch.tensor(
                [[cell == "#" for cell in row] for row in field_layout],
                dtype=torch.bool,
            ).unsqueeze(0),
            "previous_actions": torch.zeros(T, 5, dtype=torch.long),
            "previous_action_valid": torch.zeros(T, dtype=torch.bool),
            "previous_rewards": torch.zeros(T, 1),
            "previous_reward_mask": torch.zeros(T, dtype=torch.bool),
            "targets": {key: torch.full((T,), IGNORE_INDEX, dtype=torch.long)
                        for key in ("action_type", "card_id", "card_slot", "row", "column")},
            "action_valid": torch.zeros(T, dtype=torch.bool),
            "rewards": torch.zeros(T, 1), "return_to_go": torch.zeros(T, 1),
            "discounted_return_to_go": torch.zeros(T, 1),
            "transition_dt_ms": torch.zeros(T), "discounts": torch.zeros(T),
            "terminated": torch.zeros(T, dtype=torch.bool),
            "truncated": torch.zeros(T, dtype=torch.bool),
            "attention_mask": torch.arange(T) < length,
            "causal_mask": torch.ones(T, T, dtype=torch.bool).triu(1),
            "step_indices": torch.full((T,), -1, dtype=torch.long),
            "sequence_length": torch.tensor(length, dtype=torch.long),
            "battle_id": battle_id,
        }
        sample["padding_mask"] = ~sample["attention_mask"]
        for t, state in enumerate(observations):
            step = step_indices[t]
            sample["step_indices"][t] = step
            timestamp = state["timestamp_ms"]
            previous_time = previous_timestamps[t]
            states["time_features"][t, 0] = timestamp / 1000 / norm.time_seconds
            states["time_features"][t, 1] = (timestamp - previous_time) / 1000
            remaining = state.get("game_time_remaining_seconds")
            if remaining is not None:
                states["time_features"][t, 2] = remaining / norm.time_seconds
                states["remaining_time_mask"][t] = True
            phase = state.get("phase", "unknown")
            states["phase_ids"][t] = PHASES.index(phase) if phase in PHASES[1:] else 1
            # Spatial sorting deliberately excludes arbitrary track IDs.
            units = sorted(state["units"], key=lambda u: (
                u["row"], u["column"], u["side"], label(u["unit"]),
                u["unit_confidence"], u["detector_confidence"], u["predicted_by_kalman"],
            ))
            for j, unit in enumerate(units):
                row, column = unit["row"] - 1, unit["column"] - 1
                states["grid_counts"][t, GRID_SIDES.index(unit["side"]), row, column] += 1 / norm.unit_count
                states["unit_ids"][t, j] = self.unit_to_id.get(label(unit["unit"]), UNKNOWN_ID)
                states["unit_sides"][t, j] = SIDE_IDS[unit["side"]]
                states["unit_cells"][t, j] = torch.tensor([row, column])
                states["unit_features"][t, j] = torch.tensor([
                    unit["unit_confidence"], unit["detector_confidence"],
                    float(unit["predicted_by_kalman"]),
                ])
                states["unit_mask"][t, j] = True
            for card in state["hand"]:
                slot = card["slot"] - 1
                card_id = self.card_to_id.get(label(card["card"]), UNKNOWN_ID)
                states["hand_ids"][t, slot] = card_id
                states["hand_confidence"][t, slot] = card["confidence"]
                states["hand_known_mask"][t, slot] = card_id != UNKNOWN_ID
                states["hand_nonempty_mask"][t, slot] = card_id > EMPTY_CARD_ID
            elixir = state["elixir"].get("value")
            if elixir is not None:
                states["elixir"][t, 0] = elixir / norm.elixir
                states["elixir_mask"][t, 0] = True
            for j, tower in enumerate(TOWERS):
                hp = state["tower_hp"].get(tower, {})
                if hp.get("hp") is not None:
                    states["tower_hp"][t, j] = hp["hp"] / norm.tower_hp
                    states["tower_hp_known_mask"][t, j] = True
                    states["tower_hp_fresh_mask"][t, j] = not hp.get("stale", True)
                    states["tower_hp_confidence"][t, j] = hp.get("confidence", 0.0)
                    seen = hp.get("last_seen_at_ms")
                    states["tower_hp_age"][t, j] = (
                        (timestamp - seen) / 1000 / norm.hp_age_seconds if seen is not None else 1.0
                    )
            # Shift BEFORE slicing: the first step of an interior window still
            # receives the preceding transition from this same battle.
            if previous_transitions[t] is None:
                sample["previous_actions"][t, 0] = PREV_BOS
            else:
                previous, known = self._action(previous_transitions[t])
                sample["previous_actions"][t] = torch.tensor(previous)
                sample["previous_action_valid"][t] = known
                reward = previous_transitions[t].get("reward")
                if reward is not None:
                    sample["previous_rewards"][t, 0] = reward / norm.reward
                    sample["previous_reward_mask"][t] = True
        return sample


class TrajectoryDataset(Dataset):
    """One causal window per ending transition (stride=1), right padding to T.

    states.grid_counts: [T,3,32,18], channels ally/enemy/unknown.
    states.unit_ids/sides/mask: [T,M]; cells: [T,M,2] (0-based row,column);
      features: [T,M,3] (class confidence, detector confidence, Kalman flag).
    states.hand_ids/confidence/known_mask/nonempty_mask: [T,4].
    states.elixir/elixir_mask: [T,1]; tower_hp/masks/confidence/age: [T,4].
    states.time_features: [T,3] (elapsed/scale, previous dt seconds, remaining/scale).
    terrain_mask: [1,32,18]; '#' is terrain information, NOT an action restriction.
    previous_actions: [T,5] (type,card-ID,slot,row,column). Type IDs are PREV_*;
      previous coordinates/slots retain 1-based values with 0 for absent fields.
    targets: type 0=noop/1=play, card-ID, slot 0..3, row 0..31, column 0..17;
      unavailable targets use IGNORE_INDEX=-100, suitable for CrossEntropyLoss.
    position_loss_mask excludes ambiguous deployment cells but preserves the
      play/slot targets. Unknown prior play coordinates use reserved 0 tokens.
    All sample tensors are CPU tensors and support default DataLoader collation.
    """

    def __init__(
        self, source, sequence_length: int = 32, *, vocabulary: Vocabulary | None = None,
        normalization: Normalization | None = None, max_units: int = 128,
        stride: int = 1, supervise: str = "last",
    ):
        for name, value in (("sequence_length", sequence_length), ("max_units", max_units),
                            ("stride", stride)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if supervise not in {"last", "all"}:
            raise ValueError("supervise must be 'last' or 'all'")
        self.files = discover_trajectory_files(source)
        self.trajectories = [load_trajectory(path) for path in self.files]
        self.vocabulary = vocabulary or Vocabulary.from_trajectories(self.trajectories)
        self.normalization = normalization or Normalization()
        self.card_to_id = {name: i for i, name in enumerate(self.vocabulary.cards)}
        self.unit_to_id = {name: i for i, name in enumerate(self.vocabulary.units)}
        self.sequence_length, self.max_units = sequence_length, max_units
        self.stride, self.supervise = stride, supervise
        self.encoder = ObservationEncoder(self.vocabulary, self.normalization, sequence_length, max_units)
        self.battle_ids = [battle_identity(data) for data in self.trajectories]
        self._ends, self._cumulative = [], [0]
        for path, data in zip(self.files, self.trajectories):
            largest = max(len(state["units"]) for state in data["observations"])
            if largest > max_units:
                raise ValueError(f"{path}: {largest} units exceed max_units={max_units}; "
                                 "increase max_units (units are never silently dropped)")
            count = len(data["transitions"])
            ends = list(range(0, count, stride))
            if ends[-1] != count - 1:
                ends.append(count - 1)
            self._ends.append(ends)
            self._cumulative.append(self._cumulative[-1] + len(ends))

    def __len__(self) -> int:
        return self._cumulative[-1]

    def iter_window_endpoints(self):
        """Yield (sample index, battle index, final transition) without encoding."""
        for battle, ends in enumerate(self._ends):
            for offset, end in enumerate(ends):
                yield self._cumulative[battle] + offset, battle, end

    def encoding_config(self) -> dict:
        """Store this in a checkpoint for identical feature/ID mapping at inference."""
        return {
            "encoding_version": 1, "vocabulary": self.vocabulary.to_dict(),
            "normalization": asdict(self.normalization), "max_units": self.max_units,
            "sequence_length": self.sequence_length, "stride": self.stride,
            "supervise": self.supervise, "grid_sides": list(GRID_SIDES),
            "tower_order": list(TOWERS), "phases": list(PHASES),
        }

    def __getitem__(self, index: int) -> dict:
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        battle = bisect_right(self._cumulative, index) - 1
        end = self._ends[battle][index - self._cumulative[battle]]
        start = max(0, end + 1 - self.sequence_length)
        data = self.trajectories[battle]
        observations, transitions = data["observations"], data["transitions"]
        length = end - start + 1
        sample = self.encoder.encode(
            observations[start:end + 1],
            [transitions[i - 1] if i else None for i in range(start, end + 1)],
            [observations[i - 1]["timestamp_ms"] if i else observations[i]["timestamp_ms"]
             for i in range(start, end + 1)],
            list(range(start, end + 1)), data["metadata"]["field_layout"], self.battle_ids[battle],
        )
        norm = self.normalization
        for t, step in enumerate(range(start, end + 1)):
            transition = transitions[step]
            action, valid = self.encoder._action(transition)
            sample["action_valid"][t] = valid
            if valid:
                sample["targets"]["action_type"][t] = int(action[0] == PREV_PLAY)
                if action[0] == PREV_PLAY:
                    for key, value in zip(("card_id", "card_slot", "row", "column"),
                                          (action[1], action[2] - 1, action[3] - 1, action[4] - 1)):
                        if key not in {"row", "column"} or transition["action"].get("position_valid", True):
                            sample["targets"][key][t] = value
            for name, scale in (("reward", norm.reward), ("return_to_go", norm.return_to_go),
                                ("discounted_return_to_go", norm.return_to_go)):
                sample["rewards" if name == "reward" else name][t, 0] = transition[name] / scale
            sample["transition_dt_ms"][t] = transition["dt_ms"]
            sample["discounts"][t] = transition["discount"]
            sample["terminated"][t] = transition["terminated"]
            sample["truncated"][t] = transition["truncated"]
        sample["loss_mask"] = sample["action_valid"].clone()
        if self.supervise == "last":
            sample["loss_mask"][:length - 1] = False
        sample["play_loss_mask"] = sample["loss_mask"] & (sample["targets"]["action_type"] == 1)
        sample["position_loss_mask"] = sample["play_loss_mask"] & (sample["targets"]["row"] != IGNORE_INDEX)
        # CE(ignore_index=-100) also safely excludes context-only target positions.
        for target in sample["targets"].values():
            target[~sample["loss_mask"]] = IGNORE_INDEX
        return sample


def split_trajectory_files(source, validation_fraction: float = 0.2, seed: int = 42):
    """Group whole battles by source identity BEFORE vocabulary fitting/windowing."""
    if not math.isfinite(validation_fraction) or not 0 <= validation_fraction < 1:
        raise ValueError("validation_fraction must be in [0, 1)")
    files = discover_trajectory_files(source)
    groups: dict[str, list[Path]] = {}
    for path in files:
        groups.setdefault(battle_identity(load_trajectory(path)), []).append(path)
    if validation_fraction == 0:
        return files, []
    if len(groups) < 2:
        raise ValueError("Train/validation split requires at least two distinct battles; "
                         "use validation_fraction=0 for training only")
    identities = sorted(groups)
    random.Random(seed).shuffle(identities)
    validation_count = max(1, min(len(groups) - 1, round(len(groups) * validation_fraction)))
    validation_ids = set(identities[:validation_count])
    train = sorted(path for identity, paths in groups.items() if identity not in validation_ids for path in paths)
    validation = sorted(path for identity, paths in groups.items() if identity in validation_ids for path in paths)
    return train, validation


def create_train_val_datasets(source, *, validation_fraction=0.2, seed=42, **kwargs):
    train_files, validation_files = split_trajectory_files(source, validation_fraction, seed)
    train = TrajectoryDataset(train_files, **kwargs)
    validation_kwargs = {**kwargs, "vocabulary": train.vocabulary, "normalization": train.normalization}
    validation = (TrajectoryDataset(validation_files, **validation_kwargs)
                  if validation_files else None)
    return train, validation


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Inspect trajectory tensors without training")
    parser.add_argument("source", type=Path, help="A trajectory JSON or directory of battle JSONs")
    parser.add_argument("--sequence-length", type=int, default=32)
    parser.add_argument("--max-units", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=2)
    args = parser.parse_args(argv)
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    dataset = TrajectoryDataset(args.source, args.sequence_length, max_units=args.max_units)
    from torch.utils.data import DataLoader
    batch = next(iter(DataLoader(dataset, batch_size=args.batch_size, num_workers=0)))
    def shapes(value):
        if isinstance(value, torch.Tensor):
            return {"shape": list(value.shape), "dtype": str(value.dtype)}
        if isinstance(value, dict):
            return {key: shapes(item) for key, item in value.items()}
        return value
    print(json.dumps({
        "battles": len(dataset.files), "windows": len(dataset),
        "card_classes": len(dataset.vocabulary.cards), "unit_classes": len(dataset.vocabulary.units),
        "batch": shapes(batch),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
