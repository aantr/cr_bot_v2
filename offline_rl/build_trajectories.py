"""Build one offline-RL episode from one battle video.

Run from this directory:
    ../venv/Scripts/python.exe build_trajectories.py battle.mp4 --result win

The JSON stores observations once and transitions reference adjacent indices.
No models are loaded until main() starts extraction; assembly is testable without
GPU inference. Time always refers to source video, not processing wall time.
"""
from __future__ import annotations

import argparse
from bisect import bisect_left
from collections import Counter
from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import time


V2_DIR = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "trajectories"
TOWERS = ("ally_1", "ally_2", "enemy_1", "enemy_2")
UNKNOWN_CARDS = {"", "unknown", "none", "empty"}


@dataclass(frozen=True)
class BuildConfig:
    state_fps: float = 5.0
    detection_fps: float = 30.0
    min_card_confidence: float = 0.70
    damage_scale: float = 1000.0
    tower_reward: float = 1.0
    outcome_reward: float = 5.0
    max_hp_drop: int = 2500
    # Per-second discount: changing state_fps does not change the time horizon.
    gamma_per_second: float = 0.99
    uncertainty_ms: float = 300.0
    # Container frame counts can exceed the number OpenCV can decode.
    # Require BOTH bounds, so short clips and large truncations stay protected.
    eof_tolerance_seconds: float = 1.0
    eof_tolerance_fraction: float = 0.02

    def validate(self) -> None:
        for name in ("state_fps", "detection_fps", "damage_scale", "max_hp_drop"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.state_fps > self.detection_fps:
            raise ValueError("state_fps must not exceed detection_fps")
        for name in ("min_card_confidence", "gamma_per_second"):
            if not math.isfinite(getattr(self, name)) or not 0 < getattr(self, name) <= 1:
                raise ValueError(f"{name} must be in (0, 1]")
        for name in ("tower_reward", "outcome_reward", "uncertainty_ms",
                     "eof_tolerance_seconds"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        if (not math.isfinite(self.eof_tolerance_fraction)
                or not 0 <= self.eof_tolerance_fraction <= 1):
            raise ValueError("eof_tolerance_fraction must be in [0, 1]")


class HPRewardTracker:
    """Reward each confirmed HP decrease once, ignoring missing OCR and healing.

    A monotonically decreasing HP floor prevents high/low OCR flicker from
    repeatedly paying for the same damage. Large jumps are rejected and counted.
    With this conservative policy legitimate large hits/healing can be missed.
    """

    def __init__(self, config: BuildConfig):
        self.config = config
        self.floor: dict[str, int] = {}
        self.last_confirmation: dict[str, float] = {}
        self.events: list[dict] = []
        self.stats: Counter = Counter()

    def observe(self, states: dict, timestamp_ms: float) -> None:
        for tower in TOWERS:
            state = states.get(tower, {})
            hp = state.get("hp")
            confirmed = state.get("confirmed_at_ms")
            if (hp is None or state.get("stale", True) or confirmed is None
                    or confirmed > timestamp_ms or hp < 0):
                continue
            if self.last_confirmation.get(tower) == confirmed:
                continue
            self.last_confirmation[tower] = confirmed
            if tower not in self.floor:
                self.floor[tower] = int(hp)
                continue  # First observed HP is a baseline, never damage.
            previous = self.floor[tower]
            damage = previous - int(hp)
            if damage < 0:
                self.stats["ignored_hp_increases"] += 1
                continue
            if damage == 0:
                continue
            if damage > self.config.max_hp_drop:
                self.stats["rejected_hp_jumps"] += 1
                continue
            self.floor[tower] = int(hp)
            sign = 1.0 if tower.startswith("enemy") else -1.0
            destruction = hp == 0 and previous > 0
            self.events.append({
                "timestamp_ms": float(confirmed), "tower": tower,
                "hp_before": previous, "hp_after": int(hp), "damage": damage,
                "damage_reward": sign * damage / self.config.damage_scale,
                "tower_reward": sign * self.config.tower_reward if destruction else 0.0,
                "confidence": float(state.get("confidence", 0.0)),
            })


def is_known_card(name: str) -> bool:
    return name.lower() not in UNKNOWN_CARDS


def sparse_field(units: list[dict]) -> list[dict]:
    """Keep overlapping units as counts; elixir drops never occupy the grid."""
    counts = Counter(
        (int(unit["row"]), int(unit["column"]), unit["side"], unit["unit"])
        for unit in units
    )
    return [
        {"row": row, "column": column, "side": side, "unit": unit, "count": count}
        for (row, column, side, unit), count in sorted(counts.items())
    ]


def make_observation(frame: dict) -> dict:
    units = frame["units"]
    return {
        "timestamp_ms": float(frame["timestamp_ms"]),
        "frame_number": int(frame["frame_number"]),
        "units": units, "field_cells": sparse_field(units),
        "hand": frame["hand"], "elixir": frame["elixir_bar"],
        "tower_hp": frame["tower_hp"],
        # Current perception has no game-clock/phase recognizer. Do not invent it.
        "game_time_remaining_seconds": None, "phase": "unknown",
    }


class TrajectoryCollector:
    def __init__(self, config: BuildConfig):
        config.validate()
        self.config = config
        self.observations: list[dict] = []
        self.events: list[dict] = []
        self.empty_transitions: list[dict] = []
        self.elixir_drops: list[tuple[float, float]] = []
        self.hp_rewards = HPRewardTracker(config)
        self.last_frame: dict | None = None
        self.next_due_ms = 0.0
        self.frames_seen = 0
        self.last_progress_time = time.monotonic()

    def __call__(self, frame: dict) -> None:
        timestamp = float(frame["timestamp_ms"])
        if self.last_frame is not None and timestamp <= self.last_frame["timestamp_ms"]:
            raise ValueError("Observation timestamps must be strictly increasing")
        if frame["processing_fps"] < self.config.state_fps:
            raise ValueError("Source video FPS is lower than requested state_fps")
        self.frames_seen += 1
        self.hp_rewards.observe(frame["tower_hp"], timestamp)
        self.events.extend(dict(event) for event in frame["events"])
        if self.last_frame is not None:
            previous_time = self.last_frame["timestamp_ms"]
            for old, new in zip(self.last_frame["hand"], frame["hand"]):
                if is_known_card(old["card"]) and new["card"].lower() == "empty":
                    self.empty_transitions.append({
                        "slot": new["slot"], "frame_number": frame["frame_number"],
                        "timestamp_ms": timestamp, "previous_timestamp_ms": previous_time,
                        "previous_card": old["card"],
                    })
            previous_elixir = self.last_frame["elixir_bar"]["value"]
            current_elixir = frame["elixir_bar"]["value"]
            if (previous_elixir is not None and current_elixir is not None
                    and previous_elixir - current_elixir >= 0.8):
                self.elixir_drops.append((previous_time, timestamp))
        if timestamp + 1e-6 >= self.next_due_ms:
            self.observations.append(make_observation(frame))
            interval = 1000.0 / self.config.state_fps
            self.next_due_ms = (math.floor((timestamp + 1e-6) / interval) + 1) * interval
        self.last_frame = frame
        if time.monotonic() - self.last_progress_time >= 10:
            print(f"Video {timestamp / 1000:.1f}s | frames={self.frames_seen} "
                  f"states={len(self.observations)} | events={len(self.events)}",
                  flush=True)
            self.last_progress_time = time.monotonic()

    def finish(self, result: str, metadata: dict) -> dict:
        if self.last_frame is not None and (
            not self.observations
            or self.observations[-1]["timestamp_ms"] < self.last_frame["timestamp_ms"]
        ):
            self.observations.append(make_observation(self.last_frame))
        if len(self.observations) < 2:
            raise ValueError("At least two processed observations are required")
        if metadata.get("stopped_by_user"):
            raise ValueError("Interrupted video cannot be saved as a complete battle")
        expected = metadata.get("source_frame_count", 0)
        decoded = metadata.get("decoded_frame_count", expected)
        missing = max(0, expected - decoded)
        source_fps = metadata.get("source_fps", 0)
        fps_valid = math.isfinite(source_fps) and source_fps > 0
        allowed_missing = (
            math.floor(min(
                source_fps * self.config.eof_tolerance_seconds,
                expected * self.config.eof_tolerance_fraction,
            )) if fps_valid and expected > 0 else 0
        )
        if missing > allowed_missing:
            raise ValueError(
                f"Video decoding stopped early: {decoded}/{expected} frames; "
                f"missing={missing}, allowed={allowed_missing}. "
                "Check the video ending. EOF tolerance is controlled by "
                "--eof-tolerance-seconds and --eof-tolerance-fraction."
            )
        eof_info = {
            "reported_frames": expected, "decoded_frames": decoded,
            "missing_frames": missing,
            "estimated_missing_seconds": missing / source_fps if fps_valid else None,
            "allowed_missing_frames": allowed_missing,
            "status": "within_tolerance" if missing else "no_shortfall",
        }
        trajectory = assemble_trajectory(
            self.observations, self.events, self.empty_transitions,
            self.elixir_drops, self.hp_rewards, result, self.config,
        )
        trajectory["metadata"] = {**metadata, "video_eof": eof_info}
        if missing:
            warning = (
                f"Video ended at {decoded}/{expected} frames "
                f"(shortfall {missing / source_fps:.3f}s), within EOF tolerance. "
                "The supplied battle result is applied to the last processed state; "
                "missing frames are not reconstructed."
            )
            trajectory["metrics"]["warnings"].append(warning)
            print(f"Warning: {warning}", file=sys.stderr, flush=True)
        return trajectory


def invalidate(transition: dict, reason: str) -> None:
    transition["action_valid"] = False
    if reason not in transition["invalid_reasons"]:
        transition["invalid_reasons"].append(reason)


def consolidate_events(events: list[dict]) -> list[dict]:
    """One hand-empty onset labels at most one play; retain raw candidates.

    Only exact same card/slot/empty frame/action time can be duplicates.
    Conflicting cells do NOT get averaged or guessed: card/slot remain usable,
    while position_valid=False excludes coordinate supervision and history.
    """
    result = deepcopy(events)
    groups = {}
    for index, event in enumerate(result):
        for key in ("duplicate_of", "duplicate_indices", "position_candidates", "position_valid"):
            event.pop(key, None)
        if event.get("empty_frame") is None or event.get("slot") not in (1, 2, 3, 4) or not is_known_card(event["card"]):
            continue
        key = (event["slot"], event["empty_frame"], event["card"], event["action_timestamp_ms"])
        groups.setdefault(key, []).append(index)
    for indices in groups.values():
        canonical = min(indices, key=lambda i: (result[i]["confirmed_at_ms"], i))
        main = result[canonical]
        candidates = sorted({(result[i]["row"], result[i]["column"]) for i in indices})
        main["duplicate_indices"] = [i for i in indices if i != canonical]
        main["position_candidates"] = [{"row": row, "column": column} for row, column in candidates]
        main["position_valid"] = len(candidates) == 1
        for index in indices:
            if index != canonical:
                result[index]["duplicate_of"] = canonical
    return result


def align_event_onsets(observations, events, empty_transitions, config):
    """Correct a late empty sample only using an observed, unambiguous onset.

    Do not search arbitrary earlier states for a convenient card or change slots.
    When full-rate onset evidence is missing/conflicting, keep the label invalid.
    """
    result = deepcopy(events)
    times = [state["timestamp_ms"] for state in observations]
    def matches(index, event):
        if not 0 <= index < len(observations) - 1:
            return False
        hand = next((card for card in observations[index]["hand"] if card["slot"] == event["slot"]), None)
        return hand is not None and hand["card"] == event["card"] and hand["confidence"] >= config.min_card_confidence
    for event in result:
        timestamp = event["action_timestamp_ms"]
        if matches(bisect_left(times, timestamp) - 1, event):
            continue
        candidates = [empty for empty in empty_transitions
                      if empty.get("previous_card") == event["card"] and empty["slot"] == event["slot"]
                      and 0 <= timestamp - empty["timestamp_ms"] <= config.uncertainty_ms
                      and empty["timestamp_ms"] <= event["confirmed_at_ms"]
                      and matches(bisect_left(times, empty["timestamp_ms"]) - 1, event)]
        if len(candidates) == 1:
            onset = candidates[0]
            event["original_action_timestamp_ms"] = timestamp
            event["action_timestamp_ms"] = onset["timestamp_ms"]
            event["alignment_reason"] = "confirmed_empty_onset"
            if "frame_number" in onset:
                event["original_empty_frame"] = event.get("empty_frame")
                event["empty_frame"] = onset["frame_number"]
    return result


def assemble_trajectory(
    observations: list[dict], events: list[dict], empty_transitions: list[dict],
    elixir_drops: list[tuple[float, float]], hp_rewards: HPRewardTracker,
    result: str, config: BuildConfig,
) -> dict:
    if result not in {"win", "loss", "draw"}:
        raise ValueError("result must be win, loss or draw")
    events = consolidate_events(align_event_onsets(observations, events, empty_transitions, config))
    times = [observation["timestamp_ms"] for observation in observations]
    if len(times) < 2 or any(b <= a for a, b in zip(times, times[1:])):
        raise ValueError("Need at least two strictly increasing state timestamps")
    transitions = []
    for index, (before, after) in enumerate(zip(times, times[1:])):
        transitions.append({
            "state_index": index, "next_state_index": index + 1,
            "dt_ms": after - before, "action": {"type": "noop"},
            "action_valid": True, "invalid_reasons": [], "event_indices": [],
            "reward": 0.0,
            "reward_components": {"damage": 0.0, "tower": 0.0, "outcome": 0.0},
            "terminated": index == len(times) - 2, "truncated": False,
        })

    # Keep unresolved / out-of-context events in the file, never silently relabel
    # them as no-op. Coordinates and slots remain 1-based, including '#' cells.
    for event_index, event in enumerate(events):
        action_time = event["action_timestamp_ms"]
        index = bisect_left(times, action_time) - 1
        in_context = 0 <= index < len(transitions)
        index = min(len(transitions) - 1, max(0, index))
        event["transition_index"] = index
        if "duplicate_of" in event:
            continue
        transition = transitions[index]
        transition["event_indices"].append(event_index)
        if len(transition["event_indices"]) > 1:
            transition["action"] = {"type": "multiple"}
            invalidate(transition, "multiple_events_in_interval")
        else:
            transition["action"] = {
                "type": "play" if is_known_card(event["card"]) else "unknown",
                "card": event["card"], "slot": event["slot"],
                "column": event["column"], "row": event["row"],
                "timestamp_ms": action_time,
                "confidence": float(event["confidence"]),
                "position_valid": event.get("position_valid", True),
            }
        if not in_context:
            invalidate(transition, "no_pre_action_state")
        if not is_known_card(event["card"]) or event["slot"] not in (1, 2, 3, 4):
            invalidate(transition, "unresolved_card")
        elif event["confidence"] < config.min_card_confidence:
            invalidate(transition, "low_card_confidence")
        else:
            slot = observations[index]["hand"][event["slot"] - 1]
            if slot["card"] != event["card"] or slot["confidence"] < config.min_card_confidence:
                invalidate(transition, "pre_action_hand_mismatch")
        if not (1 <= event["column"] <= 18 and 1 <= event["row"] <= 32):
            invalidate(transition, "outside_battlefield")
        if action_time > event["confirmed_at_ms"]:
            invalidate(transition, "action_after_confirmation")

    # The same empty transition cannot label two drops / two actions.
    matches: dict[tuple, list[int]] = {}
    for event in events:
        if event.get("empty_frame") is not None and "duplicate_of" not in event:
            key = (event["slot"], event["empty_frame"])
            matches.setdefault(key, []).append(event["transition_index"])
    for indices in matches.values():
        if len(indices) > 1:
            for index in indices:
                invalidate(transitions[index], "reused_hand_transition")

    suspicious_intervals = []
    unmatched_empty = 0
    for empty in empty_transitions:
        # Resolver may select a later empty sample, not exactly its onset.
        matched = any(
            event["slot"] == empty["slot"]
            and event.get("empty_frame") is not None
            and 0 <= event["action_timestamp_ms"] - empty["timestamp_ms"] <= config.uncertainty_ms
            for event in events
        )
        if not matched:
            unmatched_empty += 1
            suspicious_intervals.append((
                empty["previous_timestamp_ms"],
                empty["timestamp_ms"] + config.uncertainty_ms, "unmatched_empty_slot",
            ))
    for before, after in elixir_drops:
        if not any(abs(event["action_timestamp_ms"] - after) <= config.uncertainty_ms
                   for event in events):
            suspicious_intervals.append((
                before, after + config.uncertainty_ms, "unmatched_elixir_spend",
            ))
    for transition in transitions:
        i = transition["state_index"]
        for before, after, reason in suspicious_intervals:
            if times[i] <= after and times[i + 1] > before:
                invalidate(transition, reason)
        # A confirmed play needs its OWN slot to be reliable (checked above).
        # Other uncertain hand slots are input uncertainty, not bad action labels.
        # Noop has no positive evidence: retain conservative all-hand validation.
        if transition["action"]["type"] == "noop" and any(slot["confidence"] < config.min_card_confidence
               or slot["card"].lower() in {"unknown", "none", ""}
               for slot in observations[i]["hand"]):
            invalidate(transition, "uncertain_hand")

    # Rewards use confirmed observations only. No HP from a later frame is
    # backfilled into an earlier state; reward quality is separate from action masks.
    for evidence in hp_rewards.events:
        index = bisect_left(times, evidence["timestamp_ms"]) - 1
        if 0 <= index < len(transitions):
            components = transitions[index]["reward_components"]
            components["damage"] += evidence["damage_reward"]
            components["tower"] += evidence["tower_reward"]
    outcome_sign = {"win": 1, "loss": -1, "draw": 0}[result]
    transitions[-1]["reward_components"]["outcome"] = outcome_sign * config.outcome_reward
    discounted_return = 0.0
    return_to_go = 0.0
    for transition in reversed(transitions):
        transition["reward"] = sum(transition["reward_components"].values())
        discount = config.gamma_per_second ** (transition["dt_ms"] / 1000.0)
        transition["discount"] = 0.0 if transition["terminated"] else discount
        return_to_go += transition["reward"]
        discounted_return = transition["reward"] + transition["discount"] * discounted_return
        transition["return_to_go"] = return_to_go
        transition["discounted_return_to_go"] = discounted_return

    all_units = [unit for state in observations for unit in state["units"]]
    hp_known = {
        tower: sum(
            state["tower_hp"].get(tower, {}).get("hp") is not None
            and not state["tower_hp"].get(tower, {}).get("stale", True)
            for state in observations
        ) / len(observations) for tower in TOWERS
    }
    reasons = Counter(reason for t in transitions for reason in t["invalid_reasons"])
    warnings = []
    if not events:
        warnings.append("No play events detected; this is not a verified no-op-only battle.")
    if not hp_rewards.floor:
        warnings.append("No confirmed HP baseline; rewards contain only the supplied outcome.")
    if reasons:
        warnings.append("Train action losses only where action_valid=true; retain elapsed time.")
    metrics = {
        "states": len(observations), "transitions": len(transitions),
        "video_duration_seconds": (times[-1] - times[0]) / 1000.0,
        "detected_events": len(events),
        "duplicate_events": sum("duplicate_of" in e for e in events),
        "unique_events": sum("duplicate_of" not in e for e in events),
        "time_aligned_events": sum(e.get("alignment_reason") == "confirmed_empty_onset" for e in events),
        "resolved_events": sum(is_known_card(e["card"]) and e["slot"] in (1, 2, 3, 4)
                               for e in events),
        "valid_play_transitions": sum(t["action_valid"] and t["action"]["type"] == "play"
                                      for t in transitions),
        "valid_position_transitions": sum(t["action_valid"] and t["action"]["type"] == "play"
                                           and t["action"].get("position_valid", True) for t in transitions),
        "valid_noop_transitions": sum(t["action_valid"] and t["action"]["type"] == "noop"
                                      for t in transitions),
        "action_valid_fraction": sum(t["action_valid"] for t in transitions) / len(transitions),
        "invalid_reasons": dict(reasons), "unmatched_empty_transitions": unmatched_empty,
        "mean_units": len(all_units) / len(observations),
        "unknown_unit_fraction": (sum(u["unit"].lower() in UNKNOWN_CARDS for u in all_units)
                                  / len(all_units) if all_units else 0.0),
        "kalman_predicted_fraction": (sum(u["predicted_by_kalman"] for u in all_units)
                                      / len(all_units) if all_units else 0.0),
        "tower_hp_valid_fraction": hp_known,
        "enemy_damage": sum(e["damage"] for e in hp_rewards.events if e["tower"].startswith("enemy")),
        "ally_damage": sum(e["damage"] for e in hp_rewards.events if e["tower"].startswith("ally")),
        "total_reward": sum(t["reward"] for t in transitions),
        "reward_filter": dict(hp_rewards.stats), "warnings": warnings,
    }
    return {
        "schema_version": 1, "action_label_version": 2, "result": result, "config": asdict(config),
        "coordinates": {"rows": 32, "columns": 18, "origin": "top-left",
                        "index_base": 1, "hash_cells_allowed": True},
        "observations": observations, "transitions": transitions,
        "events": events, "reward_evidence": hp_rewards.events,
        "empty_transitions": deepcopy(empty_transitions), "elixir_drops": list(elixir_drops),
        "uncertainty_intervals": [
            {"start_ms": before, "end_ms": after, "reason": reason}
            for before, after, reason in suspicious_intervals
        ],
        "metrics": metrics,
    }


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_trajectory(path: Path, trajectory: dict, overwrite: bool = False) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {path}. Use --overwrite to replace.")
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent,
            prefix=path.name + ".", suffix=".tmp", delete=False,
        ) as output:
            temp_path = Path(output.name)
            json.dump(trajectory, output, ensure_ascii=False, allow_nan=False,
                      separators=(",", ":"))
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        if path.exists() and not overwrite:
            raise FileExistsError(f"Output appeared during processing: {path}")
        os.replace(temp_path, path)
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()


def relabel_trajectory(path: Path) -> dict:
    """Rebuild labels from saved evidence without rerunning perception or editing states."""
    old = json.loads(path.read_text(encoding="utf-8-sig"))
    config = BuildConfig(**old["config"])
    config.validate()
    # Legacy JSONs kept only unmatched evidence as uncertainty intervals.
    # Preserve those exclusions exactly; don't invent missing cards/onsets.
    empty = old.get("empty_transitions")
    drops = old.get("elixir_drops")
    if empty is None:
        empty = [{"slot": None, "timestamp_ms": item["end_ms"] - config.uncertainty_ms,
                  "previous_timestamp_ms": item["start_ms"], "legacy_unmatched": True}
                 for item in old.get("uncertainty_intervals", []) if item["reason"] == "unmatched_empty_slot"]
    if drops is None:
        drops = [(item["start_ms"], item["end_ms"] - config.uncertainty_ms)
                 for item in old.get("uncertainty_intervals", []) if item["reason"] == "unmatched_elixir_spend"]
    rewards = HPRewardTracker(config)
    rewards.events = deepcopy(old["reward_evidence"])
    rewards.stats = Counter(old.get("metrics", {}).get("reward_filter", {}))
    for state in old["observations"]:
        for tower, hp in state["tower_hp"].items():
            if hp.get("hp") is not None and not hp.get("stale", True):
                rewards.floor[tower] = hp["hp"]
    new = assemble_trajectory(old["observations"], old["events"], empty, drops, rewards, old["result"], config)
    new["metadata"] = deepcopy(old["metadata"])
    new["metadata"]["relabeling"] = {
        "source_json": str(path.resolve()), "source_json_sha256": file_sha256(path),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "previous_valid_plays": old["metrics"]["valid_play_transitions"],
        "legacy_evidence": "empty_transitions" not in old,
    }
    for warning in old.get("metrics", {}).get("warnings", []):
        if "Video ended" in warning and warning not in new["metrics"]["warnings"]:
            new["metrics"]["warnings"].append(warning)
    if [t["reward"] for t in new["transitions"]] != [t["reward"] for t in old["transitions"]]:
        raise ValueError("Relabeling unexpectedly changed rewards; original JSON left untouched")
    return new


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", nargs="?", type=Path, help="One complete battle video")
    parser.add_argument("--relabel-json", type=Path, help="Rebuild labels from existing JSON; uses its saved config/result")
    parser.add_argument("--result", choices=("win", "loss", "draw"),
                        help="Outcome from the perspective of the bottom player's hand")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--state-fps", type=float, default=5.0)
    parser.add_argument("--detection-fps", type=float, default=30.0)
    parser.add_argument("--min-card-confidence", type=float, default=0.70)
    parser.add_argument("--damage-scale", type=float, default=1000.0)
    parser.add_argument("--tower-reward", type=float, default=1.0)
    parser.add_argument("--outcome-reward", type=float, default=5.0)
    parser.add_argument("--max-hp-drop", type=int, default=2500)
    parser.add_argument("--gamma-per-second", type=float, default=0.99)
    parser.add_argument("--eof-tolerance-seconds", type=float, default=1.0,
                        help="Maximum tolerated tail shortfall in seconds (0 = strict)")
    parser.add_argument("--eof-tolerance-fraction", type=float, default=0.02,
                        help="Also limit tail shortfall to this fraction of reported frames")
    parser.add_argument("--no-tower-hp", action="store_true",
                        help="Explicitly disable OCR and use terminal-only rewards")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.relabel_json:
        if args.video is not None or args.result is not None:
            raise ValueError("--relabel-json uses the saved observations/result; omit video and --result")
        output = args.output_dir.resolve() / args.relabel_json.name
        if output == args.relabel_json.resolve():
            raise ValueError("Choose a different --output-dir for relabeling; original JSON is preserved")
        if output.exists() and not args.overwrite:
            raise FileExistsError(f"Trajectory already exists: {output}")
        trajectory = relabel_trajectory(args.relabel_json)
        save_trajectory(output, trajectory, args.overwrite)
        print(f"Saved relabeled trajectory: {output}")
        print(json.dumps(trajectory["metrics"], ensure_ascii=False, indent=2))
        return 0
    if args.video is None or args.result is None:
        raise ValueError("Video extraction requires VIDEO --result win|loss|draw")
    config = BuildConfig(**{
        key: getattr(args, key) for key in asdict(BuildConfig()) if hasattr(args, key)
    })
    config.validate()
    video = args.video.expanduser().resolve()
    if not video.is_file():
        raise FileNotFoundError(f"Video not found: {video}")
    digest = file_sha256(video)
    output_path = args.output_dir.expanduser().resolve() / f"{video.stem}-{digest[:12]}.json"
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"Trajectory already exists: {output_path}; use --overwrite")
    if str(V2_DIR) not in sys.path:
        sys.path.insert(0, str(V2_DIR))
    from predict_video_kalman import FIELD, run_video_prediction

    print(f"Extracting {video.name} | result={args.result} | "
          f"detection={config.detection_fps:g} Hz, states={config.state_fps:g} Hz", flush=True)
    started = time.monotonic()
    collector = TrajectoryCollector(config)
    metadata = run_video_prediction(
        input_video=video, process_fps_limit=config.detection_fps,
        display=False, write_video=False, write_logs=False,
        observation_callback=collector, hp_enabled=not args.no_tower_hp,
        synchronous_hp=True,
    )
    metadata.update({
        "battle_id": digest, "source_video": str(video), "source_sha256": digest,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "processing_seconds": time.monotonic() - started,
        "tower_hp_enabled": not args.no_tower_hp, "field_layout": list(FIELD),
        "perspective": "bottom_player",
        "reward_formula": "(enemy_damage-ally_damage)/damage_scale + "
                          "tower_reward*(enemy_destroyed-ally_destroyed) + terminal_outcome",
        "assumptions": [
            "One complete battle per video, normal playback speed; trim menus/replays first.",
            "Battle clock, phase and king-tower HP are not recognized.",
            "Coordinates inferred from elixir markers approximate the deployment location.",
            "Reward is a heuristic; quality metrics are not recognition accuracy.",
            "Keep all frames from the same source_sha256 in the same train/validation split.",
        ],
    })
    trajectory = collector.finish(args.result, metadata)
    save_trajectory(output_path, trajectory, args.overwrite)
    print(f"Saved: {output_path}")
    print(json.dumps(trajectory["metrics"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Cancelled; no completed trajectory was written.", file=sys.stderr)
        raise SystemExit(130)
    except (OSError, ValueError, RuntimeError) as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(1)
