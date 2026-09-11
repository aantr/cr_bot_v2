"""Stateful checkpoint inference; recommendations only, never game controls.

From v2:
  python offline_rl/predict_action.py runs/offline_rl/first/best.pt --replay battle.json
  python offline_rl/predict_action.py runs/offline_rl/first/best.pt --input states.jsonl

JSONL messages contain observation, optional previous_action/previous_reward,
previous_action_valid and optional allowed_slots/allowed_cells/slot_costs.
Send {"reset": true, "battle_id": "new-battle"} between battles.
Observations match build_trajectories.make_observation, not raw video frames.
All public action slots/rows/columns are ONE-based; no-op fields are null.
"""
from __future__ import annotations

import argparse
from collections import deque
from contextlib import ExitStack
from copy import deepcopy
import json
import math
from pathlib import Path
import sys

import torch
from torch.utils.data import default_collate

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from offline_rl.dataset import (ObservationEncoder, Normalization, Vocabulary, label,
                                    validate_observation, load_trajectory, GRID_SIDES, TOWERS, PHASES)
    from offline_rl.starformer import StARformer, StARformerConfig
    from offline_rl.history_features import policy_inputs
else:
    from .dataset import (ObservationEncoder, Normalization, Vocabulary, label,
                          validate_observation, load_trajectory, GRID_SIDES, TOWERS, PHASES)
    from .starformer import StARformer, StARformerConfig
    from .history_features import policy_inputs


def default_field_layout():
    if __package__ and "." in __package__:
        from ..field import FIELD
    else:
        from field import FIELD
    return list(FIELD)


def _number(value, name, minimum=None, maximum=None):
    if (isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value)
            or (minimum is not None and value < minimum) or (maximum is not None and value > maximum)):
        raise ValueError(f"Invalid {name}: expected a finite number in [{minimum}, {maximum}]")
    return value


def _coordinate(value, name, maximum):
    _number(value, name, 1, maximum)
    if not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")


def _to_device(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {key: _to_device(item, device) for key, item in value.items()}
    return value


class ActionPredictor:
    """One predictor owns one battle's bounded history (not thread-safe).

    observe() records a measured state and the actual preceding transition.
    Omitted action means unknown, NOT noop; omitted reward means unavailable.
    predict() is read-only with respect to history, so repeated calls are safe.
    reset() must be called at each battle boundary. No detections run here.
    """

    def __init__(self, checkpoint, *, device="auto", field_layout=None,
                 card_costs=None, min_card_confidence=0.0, play_threshold=0.5):
        self.play_threshold = _number(play_threshold, "play_threshold", 0, 1)
        device_name = str(device)
        
        if device_name == "auto":
            device_name = "cuda:0" if torch.cuda.is_available() else "cpu"
        if device_name.isdigit():
            device_name = "cuda:" + device_name
        self.device = torch.device(device_name)
        if self.device.type not in {"cuda", "cpu"}:
            raise ValueError("Only CPU and CUDA are supported")
        if self.device.type == "cuda" and (not torch.cuda.is_available()
                                           or (self.device.index or 0) >= torch.cuda.device_count()):
            raise ValueError(f"CUDA device {self.device} unavailable; use --device cpu")
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if saved.get("checkpoint_version") != 1 or saved.get("policy_kind") not in {"imitation", "iql"}:
            raise ValueError("Expected a version-1 imitation or IQL checkpoint")
        self.policy_kind = saved["policy_kind"]
        self.history_mode = saved.get("history_mode", "executed_feedback")
        if self.policy_kind == "iql" and (saved.get("iql_version") != 1 or self.history_mode != "observations_only"):
            raise ValueError("Unsupported IQL checkpoint/history mode")
        if self.history_mode not in {"executed_feedback", "observations_only"}:
            raise ValueError("Unsupported checkpoint history mode")
        encoding = saved["encoding_config"]
        if (encoding.get("encoding_version") != 1 or encoding.get("grid_sides") != list(GRID_SIDES)
                or encoding.get("tower_order") != list(TOWERS) or encoding.get("phases") != list(PHASES)):
            raise ValueError("Unsupported checkpoint feature encoding")
        self.encoding_config = deepcopy(encoding)
        self.vocabulary = Vocabulary.from_dict(encoding["vocabulary"])
        self.normalization = Normalization(**encoding["normalization"])
        config = StARformerConfig(**saved["model_config"])
        if (config.num_cards != len(self.vocabulary.cards) or config.num_units != len(self.vocabulary.units)
                or config.sequence_length != encoding["sequence_length"]):
            raise ValueError("Model and feature encoding configuration disagree")
        if not isinstance(encoding["max_units"], int) or encoding["max_units"] < 1:
            raise ValueError("Invalid checkpoint max_units")
        self.encoder = ObservationEncoder(self.vocabulary, self.normalization,
                                          config.sequence_length, encoding["max_units"])
        self.model = StARformer(config).to(self.device)
        self.model.load_state_dict(saved["model_state_dict"], strict=True)
        self.model.eval()
        self.field_layout = list(default_field_layout() if field_layout is None else field_layout)
        if (len(self.field_layout) != 32 or any(not isinstance(row, str) or len(row) != 18
                                               or set(row) - {".", "#"} for row in self.field_layout)):
            raise ValueError("field_layout must be 32 strings of 18 '.'/'#' cells")
        self.min_card_confidence = _number(min_card_confidence, "min_card_confidence", 0, 1)
        self.card_costs = None
        if card_costs is not None:
            if not isinstance(card_costs, dict):
                raise ValueError("card_costs must map card names to costs")
            self.card_costs = {label(name): _number(cost, f"cost for {name}", 0, 10)
                               for name, cost in card_costs.items()}
        self._history = deque(maxlen=config.sequence_length)
        self.reset()

    def reset(self, battle_id="live"):
        self._history.clear()
        self.battle_id, self._step = str(battle_id), 0

    @property
    def history_length(self):
        return len(self._history)

    def observe(self, observation, *, previous_action=None, previous_reward=None,
                previous_action_valid=None):
        """Append one state at the same cadence/relative battle clock as training.

        Actual plays use type='play', card, slot, row, column and optional actual
        timestamp_ms. Confirmed idle is {'type': 'noop'}. Missing/uncertain action
        becomes UNK. A recommended action is never automatically committed.
        Invalid observations/feedback leave history unchanged.
        """
        state = deepcopy(observation)
        try:
            validate_observation(state)
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"Invalid observation: {error}") from error
        if len(state["units"]) > self.encoder.max_units:
            raise ValueError("Observed units exceed checkpoint max_units; no units were discarded")
        if previous_reward is not None:
            _number(previous_reward, "previous_reward")
        if previous_action_valid is not None and not isinstance(previous_action_valid, bool):
            raise ValueError("previous_action_valid must be boolean or null")
        previous = None
        previous_time = state["timestamp_ms"]
        if not self._history:
            if previous_action is not None or previous_reward is not None or previous_action_valid is not None:
                raise ValueError("First state is BOS: do not supply a preceding transition")
        else:
            last = self._history[-1]["observation"]
            previous_time = last["timestamp_ms"]
            if state["timestamp_ms"] <= previous_time:
                raise ValueError("Timestamps must increase; call reset() for a new battle")
            action = deepcopy(previous_action) if previous_action is not None else {"type": "unknown"}
            if not isinstance(action, dict) or action.get("type") not in {"play", "noop", "unknown", "multiple"}:
                raise ValueError("previous_action must describe play/noop/unknown/multiple")
            valid = (action["type"] in {"play", "noop"} if previous_action_valid is None else previous_action_valid)
            if valid and action["type"] not in {"play", "noop"}:
                raise ValueError("Only play/noop can have previous_action_valid=true")
            if valid and action["type"] == "play":
                try:
                    if not isinstance(action.get("position_valid", True), bool):
                        raise ValueError("position_valid must be boolean")
                    _coordinate(action["slot"], "action slot", 4)
                    _coordinate(action["row"], "action row", 32)
                    _coordinate(action["column"], "action column", 18)
                    card = label(action["card"])
                    before = next(c for c in last["hand"] if c["slot"] == action["slot"])
                    if card in {"<unknown>", "empty"} or card != label(before["card"]):
                        raise ValueError("Played card must match the preceding state's hand slot")
                    if "timestamp_ms" in action:
                        _number(action["timestamp_ms"], "action timestamp", previous_time, state["timestamp_ms"])
                        if action["timestamp_ms"] <= previous_time:
                            raise ValueError("Action must occur after the preceding observation")
                except (KeyError, TypeError) as error:
                    raise ValueError(f"Invalid previous play: {error}") from error
            previous = {"action": action, "action_valid": valid, "reward": previous_reward}
        self._history.append({"observation": state, "previous": previous,
                              "previous_time": previous_time, "step": self._step})
        self._step += 1

    def build_batch(self):
        """Expose the exact shared CPU feature batch for integration/tests."""
        if not self._history:
            raise ValueError("Call observe() before predict()")
        sample = self.encoder.encode(
            [entry["observation"] for entry in self._history],
            [entry["previous"] for entry in self._history],
            [entry["previous_time"] for entry in self._history],
            [entry["step"] for entry in self._history], self.field_layout, self.battle_id,
        )
        # No target placeholders or future-return fields enter online inference.
        inputs = policy_inputs(sample, self.history_mode)
        return default_collate([inputs])

    @torch.inference_mode()
    def predict(self, *, allowed_slots=None, allowed_cells=None, slot_costs=None):
        """Recommend a 1-based action, without modifying history.

        Masks: boolean [4] slots, [32,18] or [4,32,18] cells. # is allowed.
        Costs: optional four current costs (None blocks that slot). If a cost
        table is configured, missing cost/elixir blocks play conservatively.
        Without costs, affordability is NOT checked and the result says so.
        Confidence values are softmax scores, not calibrated success estimates.
        Play requires raw P(play) > play_threshold AND an allowed hand slot.
        The default 0.5 preserves argmax behavior, including wait on exact ties.
        """
        batch = _to_device(self.build_batch(), self.device)
        state = self._history[-1]["observation"]
        hand = sorted(state["hand"], key=lambda c: c["slot"])
        t = self.history_length - 1
        ids = batch["states"]["hand_ids"][0, t]
        slots = (ids > 2) & (batch["states"]["hand_confidence"][0, t] >= self.min_card_confidence)
        cells = torch.ones(4, 32, 18, dtype=torch.bool, device=self.device)
        if allowed_slots is not None:
            mask = torch.as_tensor(allowed_slots, device=self.device)
            if mask.dtype != torch.bool or tuple(mask.shape) != (4,):
                raise ValueError("allowed_slots must be boolean [4]")
            slots &= mask
        if allowed_cells is not None:
            mask = torch.as_tensor(allowed_cells, device=self.device)
            if mask.dtype != torch.bool or tuple(mask.shape) not in {(32, 18), (4, 32, 18)}:
                raise ValueError("allowed_cells must be boolean [32,18] or [4,32,18]")
            cells &= mask
        cost_check = slot_costs is not None or self.card_costs is not None
        if slot_costs is not None:
            if not isinstance(slot_costs, (tuple, list)) or len(slot_costs) != 4:
                raise ValueError("slot_costs must contain exactly four costs")
            costs = slot_costs
        elif self.card_costs is not None:
            costs = [self.card_costs.get(label(card["card"])) for card in hand]
        else:
            costs = None
        if costs is not None:
            available = state["elixir"].get("value")
            affordable = []
            for cost in costs:
                if cost is not None:
                    _number(cost, "slot cost", 0, 10)
                affordable.append(cost is not None and available is not None and cost <= available)
            slots &= torch.tensor(affordable, device=self.device)
        slots &= cells.flatten(1).any(-1)
        self.model.eval()
        outputs = self.model(batch)
        if any(not torch.isfinite(value[0, t]).all() for value in outputs.values()):
            raise ValueError("Model produced nonfinite logits")
        type_prob = outputs["action_type"][0, t].softmax(-1)
        play_probability = float(type_prob[1])
        play = play_probability > self.play_threshold and bool(slots.any())
        result = {"type": "play" if play else "noop", "slot": None, "card": None,
                  "row": None, "column": None, "index_base": 1,
                  "observation_timestamp_ms": state["timestamp_ms"], "battle_id": self.battle_id,
                  "history_length": self.history_length,
                  "play_probability": play_probability, "play_threshold": self.play_threshold,
                  "reason": "model_play" if play else "model_noop" if slots.any() else "no_allowed_play",
                  "constraints": {"elixir_checked": cost_check, "placement_mask_supplied": allowed_cells is not None},
                  "confidence": {"action_type": float(type_prob[int(play)]), "slot": None,
                                 "row": None, "column": None, "cell": None}}
        if not play:
            return result
        slot_prob = outputs["card_slot"][0, t].masked_fill(~slots, -torch.inf).softmax(-1)
        slot = slot_prob.argmax().item()
        joint = outputs["row_by_slot"][0, t, slot, :, None] + outputs["column_by_slot"][0, t, slot, None, :]
        cell_prob = joint.masked_fill(~cells[slot], -torch.inf).flatten().softmax(-1).reshape(32, 18)
        row, column = divmod(cell_prob.argmax().item(), 18)
        result.update(slot=slot + 1, card=self.vocabulary.cards[ids[slot].item()], row=row + 1, column=column + 1)
        result["confidence"].update(slot=float(slot_prob[slot]), row=float(cell_prob.sum(1)[row]),
                                    column=float(cell_prob.sum(0)[column]), cell=float(cell_prob[row, column]))
        return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint", type=Path)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--replay", type=Path, help="Replay recorded states and ACTUAL past actions; not a simulated rollout")
    mode.add_argument("--input", help="JSONL requests, or - for stdin")
    parser.add_argument("--output", type=Path, help="New JSONL output file; default stdout, existing files are not overwritten")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--card-costs", type=Path, help="JSON object mapping card names to elixir costs")
    parser.add_argument("--field-layout", type=Path, help="JSON list of 32 field strings; default field.py or replay layout")
    parser.add_argument("--min-card-confidence", type=float, default=0.0)
    parser.add_argument("--play-threshold", type=float, default=0.5,
                        help="Play if raw P(play) > this threshold in [0,1]; legality checks still apply (default 0.5)")
    parser.add_argument("--limit", type=int, help="Maximum predictions for a quick check")
    args = parser.parse_args(argv)
    try:
        if args.limit is not None and args.limit < 1:
            raise ValueError("limit must be positive")
        replay = load_trajectory(args.replay) if args.replay else None
        terrain = (json.loads(args.field_layout.read_text(encoding="utf-8-sig")) if args.field_layout else
                   replay["metadata"]["field_layout"] if replay else None)
        costs = json.loads(args.card_costs.read_text(encoding="utf-8-sig")) if args.card_costs else None
        predictor = ActionPredictor(args.checkpoint, device=args.device, field_layout=terrain,
                                    card_costs=costs, min_card_confidence=args.min_card_confidence,
                                    play_threshold=args.play_threshold)
        with ExitStack() as stack:
            output = stack.enter_context(args.output.open("x", encoding="utf-8")) if args.output else sys.stdout
            def emit(value):
                print(json.dumps(value, ensure_ascii=False, allow_nan=False), file=output, flush=True)
            if replay:
                predictor.reset(replay["metadata"].get("battle_id", args.replay.stem))
                for index in range(min(len(replay["transitions"]), args.limit or len(replay["transitions"]))):
                    previous = replay["transitions"][index - 1] if index else None
                    predictor.observe(replay["observations"][index], **({
                        "previous_action": previous["action"], "previous_reward": previous["reward"],
                        "previous_action_valid": previous["action_valid"]} if previous else {}))
                    emit(predictor.predict())
            else:
                stream = sys.stdin if args.input == "-" else stack.enter_context(Path(args.input).open(encoding="utf-8-sig"))
                count = 0
                for line_number, line in enumerate(stream, 1):
                    if not line.strip():
                        continue
                    try:
                        message = json.loads(line)
                        if not isinstance(message, dict):
                            raise ValueError("Request must be a JSON object")
                        if message.get("reset") is True:
                            predictor.reset(message.get("battle_id", "live"))
                            if "observation" not in message:
                                emit({"type": "reset", "battle_id": predictor.battle_id})
                                continue
                        predictor.observe(message["observation"], **{key: message[key] for key in
                            ("previous_action", "previous_reward", "previous_action_valid") if key in message})
                        emit(predictor.predict(**{key: message[key] for key in
                            ("allowed_slots", "allowed_cells", "slot_costs") if key in message}))
                    except (KeyError, TypeError, ValueError) as error:
                        raise ValueError(f"JSONL line {line_number}: {error}") from error
                    count += 1
                    if args.limit is not None and count >= args.limit:
                        break
    except (KeyError, TypeError, ValueError, OSError) as error:
        parser.exit(2, f"Error: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
