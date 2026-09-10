"""Trajectory invariants: causality, ambiguous labels, rewards and persistence."""
import importlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from contextlib import redirect_stderr, redirect_stdout

V2_DIR = Path(__file__).resolve().parents[1]
if str(V2_DIR) not in sys.path:
    sys.path.insert(0, str(V2_DIR))

from offline_rl.build_trajectories import (
    BuildConfig, HPRewardTracker, TrajectoryCollector, assemble_trajectory,
    main, save_trajectory, sparse_field,
)


def hp(value, timestamp, stale=False):
    return {"hp": value, "confirmed_at_ms": timestamp, "stale": stale, "confidence": .95}


def observation(timestamp, card="knight"):
    return {
        "timestamp_ms": timestamp, "frame_number": round(timestamp * .03),
        "source_fps": 30.0, "processing_fps": 30.0,
        "units": [], "events": [],
        "hand": [{"slot": i, "card": card if i == 1 else "arrows", "confidence": .95}
                 for i in range(1, 5)],
        "elixir_bar": {"value": 6.0, "fill_ratio": .6},
        "tower_hp": {}, "elixir_centers": [],
    }


def event(action_time=200, card="knight", column=9, empty_frame=6):
    return {
        "action_timestamp_ms": action_time, "confirmed_at_ms": 500,
        "timestamp_ms": 200, "card": card, "slot": 1 if card != "unknown" else None,
        "column": column, "row": 16, "confidence": .95,
        "empty_frame": empty_frame,
    }


def assemble(events, states=None, result="win", reward_tracker=None, **kwargs):
    config = BuildConfig()
    return assemble_trajectory(
        states or [observation(0), observation(200, "empty"), observation(400)],
        events, kwargs.get("empty_transitions", []), kwargs.get("elixir_drops", []),
        reward_tracker or HPRewardTracker(config), result, config,
    )


class TrajectoryTests(unittest.TestCase):
    def test_action_uses_strictly_pre_empty_state_and_hash_cell_is_allowed(self):
        data = assemble([event()])
        transition = data["transitions"][0]
        self.assertEqual(transition["state_index"], 0)
        self.assertEqual(transition["action"]["row"], 16)  # '#' river row is allowed.
        self.assertTrue(transition["action_valid"])
        self.assertEqual(data["observations"][0]["hand"][0]["card"], "knight")
        self.assertEqual(data["observations"][1]["hand"][0]["card"], "empty")

    def test_unknown_and_multiple_events_never_become_noop(self):
        data = assemble([event(card="unknown")])
        self.assertEqual(data["transitions"][0]["action"]["type"], "unknown")
        self.assertFalse(data["transitions"][0]["action_valid"])
        data = assemble([event(100), event(150)])
        self.assertEqual(data["transitions"][0]["action"]["type"], "multiple")
        self.assertFalse(data["transitions"][0]["action_valid"])

    def test_reused_empty_transition_masks_both_intervals(self):
        states = [observation(0), observation(200), observation(400), observation(600)]
        data = assemble([event(100), event(300)], states)
        for index in (0, 1):
            self.assertIn("reused_hand_transition", data["transitions"][index]["invalid_reasons"])

    def test_out_of_context_and_mismatched_hand(self):
        for e in (event(0), event(card="giant"), event(column=19)):
            self.assertFalse(assemble([e])["transitions"][0]["action_valid"])

    def test_empty_slot_without_event_masks_interval(self):
        data = assemble([], empty_transitions=[{
            "slot": 1, "timestamp_ms": 200, "previous_timestamp_ms": 180,
        }])
        self.assertIn("unmatched_empty_slot", data["transitions"][0]["invalid_reasons"])

    def test_unrelated_uncertain_card_does_not_discard_known_play(self):
        states = [observation(0), observation(200, "empty"), observation(400)]
        states[0]["hand"][3]["confidence"] = .2
        data = assemble([event()], states)
        self.assertTrue(data["transitions"][0]["action_valid"])
        # No-op still needs reliable hand evidence.
        self.assertFalse(assemble([], states)["transitions"][0]["action_valid"])
        states[0]["hand"][0]["confidence"] = .2
        self.assertFalse(assemble([event()], states)["transitions"][0]["action_valid"])

    def test_exact_duplicates_merge_without_changing_raw_evidence(self):
        from copy import deepcopy
        first, second = event(), event()
        second["confirmed_at_ms"] = 600
        events = [first, second]
        original = deepcopy(events)
        data = assemble(events)
        self.assertEqual(events, original)
        self.assertTrue(data["transitions"][0]["action_valid"])
        self.assertTrue(data["transitions"][0]["action"]["position_valid"])
        self.assertEqual(data["metrics"]["duplicate_events"], 1)
        self.assertEqual(data["events"][1]["duplicate_of"], 0)
        self.assertEqual(data["transitions"][0]["event_indices"], [0])

    def test_same_hand_event_conflicting_cells_keeps_card_but_masks_position(self):
        data = assemble([event(column=9), event(column=16)])
        action = data["transitions"][0]["action"]
        self.assertTrue(data["transitions"][0]["action_valid"])
        self.assertFalse(action["position_valid"])
        self.assertEqual(action["card"], "knight")
        self.assertEqual(data["metrics"]["valid_play_transitions"], 1)
        self.assertEqual(data["metrics"]["valid_position_transitions"], 0)
        self.assertEqual(len(data["events"][0]["position_candidates"]), 2)

    def test_conflicting_cards_are_not_duplicates(self):
        data = assemble([event(), event(card="giant")])
        self.assertFalse(data["transitions"][0]["action_valid"])
        self.assertEqual(data["transitions"][0]["action"]["type"], "multiple")

    def test_late_empty_time_uses_observed_onset_not_arbitrary_past_card(self):
        states = [observation(0), observation(200, "empty"), observation(400, "empty")]
        late = event(250, empty_frame=8)
        self.assertFalse(assemble([late], states)["transitions"][1]["action_valid"])
        onset = {"slot": 1, "timestamp_ms": 180, "previous_timestamp_ms": 150,
                 "previous_card": "knight", "frame_number": 6}
        data = assemble([late], states, empty_transitions=[onset])
        self.assertTrue(data["transitions"][0]["action_valid"])
        self.assertEqual(data["transitions"][0]["action"]["timestamp_ms"], 180)
        self.assertEqual(data["events"][0]["original_action_timestamp_ms"], 250)
        self.assertEqual(data["metrics"]["time_aligned_events"], 1)
        # An onset for another card cannot repair this event.
        onset["previous_card"] = "giant"
        self.assertFalse(assemble([late], states, empty_transitions=[onset])["transitions"][1]["action_valid"])

    def test_relabel_legacy_preserves_input_states_rewards_and_exclusions(self):
        from offline_rl.build_trajectories import relabel_trajectory
        data = assemble([event(), event(column=16)], empty_transitions=[{
            "slot": 2, "timestamp_ms": 300, "previous_timestamp_ms": 280}])
        data["metadata"] = {}
        del data["empty_transitions"], data["elixir_drops"]  # legacy evidence format
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "old.json"
            path.write_text(json.dumps(data), encoding="utf-8")
            before = path.read_bytes()
            repaired = relabel_trajectory(path)
            self.assertEqual(path.read_bytes(), before)
            self.assertEqual(repaired["observations"], data["observations"])
            self.assertEqual([t["reward"] for t in repaired["transitions"]],
                             [t["reward"] for t in data["transitions"]])
            self.assertIn("unmatched_empty_slot", repaired["transitions"][1]["invalid_reasons"])
            self.assertFalse(repaired["transitions"][0]["action"]["position_valid"])
            with self.assertRaisesRegex(ValueError, "original JSON is preserved"):
                main(["--relabel-json", str(path), "--output-dir", temporary])

    def test_hp_missing_and_increases_do_not_pay_twice(self):
        rewards = HPRewardTracker(BuildConfig())
        for value, timestamp, stale in [
            (2000, 0, False), (1500, 100, False), (None, 200, True),
            (2000, 300, False), (1500, 400, False), (1000, 500, False),
        ]:
            rewards.observe({"enemy_1": hp(value, timestamp, stale)}, timestamp)
        self.assertEqual(sum(e["damage"] for e in rewards.events), 1000)
        self.assertEqual(len(rewards.events), 2)

    def test_reward_formula_terminal_and_discount(self):
        rewards = HPRewardTracker(BuildConfig())
        rewards.observe({"enemy_1": hp(1000, 0), "ally_1": hp(1000, 0)}, 0)
        rewards.observe({"enemy_1": hp(0, 100), "ally_1": hp(500, 100)}, 100)
        data = assemble([], reward_tracker=rewards)
        steps = data["transitions"]
        self.assertAlmostEqual(steps[0]["reward"], 1.5)  # +1 damage +1 tower -0.5 damage
        self.assertEqual(steps[1]["reward"], 5)
        self.assertEqual(steps[0]["return_to_go"], 6.5)
        self.assertAlmostEqual(steps[0]["discounted_return_to_go"], 1.5 + .99 ** .2 * 5)
        self.assertEqual([t["terminated"] for t in steps], [False, True])
        self.assertEqual(steps[-1]["discount"], 0)
        self.assertEqual(assemble([], result="loss")["transitions"][-1]["reward"], -5)
        self.assertEqual(assemble([], result="draw")["transitions"][-1]["reward"], 0)

    def test_large_hp_jump_is_rejected(self):
        rewards = HPRewardTracker(BuildConfig())
        rewards.observe({"enemy_1": hp(5000, 0)}, 0)
        rewards.observe({"enemy_1": hp(50, 200)}, 200)
        self.assertEqual(rewards.events, [])
        self.assertEqual(rewards.stats["rejected_hp_jumps"], 1)

    def test_sampling_keeps_last_observation_and_timestamps(self):
        collector = TrajectoryCollector(BuildConfig())
        for i in range(11):
            collector(observation(i * 1000 / 30))
        data = collector.finish("win", {})
        self.assertEqual([s["frame_number"] for s in data["observations"]], [0, 6, 10])
        self.assertAlmostEqual(sum(t["dt_ms"] for t in data["transitions"]), 10000 / 30)

    def test_incomplete_video_is_not_given_winning_terminal_reward(self):
        collector = TrajectoryCollector(BuildConfig())
        collector(observation(0))
        collector(observation(200))
        with self.assertRaisesRegex(ValueError, "stopped early"):
            collector.finish("win", {"source_frame_count": 100, "decoded_frame_count": 7})

    def test_reported_3033_decoded_3001_is_accepted_with_warning(self):
        collector = TrajectoryCollector(BuildConfig())
        collector(observation(0))
        last = observation(3000 * 1000 / 59.76354679802956)
        collector(last)
        with redirect_stderr(io.StringIO()) as stderr:
            data = collector.finish("win", {
                "source_frame_count": 3033, "decoded_frame_count": 3001,
                "source_fps": 59.76354679802956,
            })
        eof = data["metadata"]["video_eof"]
        self.assertEqual(eof["missing_frames"], 32)
        self.assertAlmostEqual(eof["estimated_missing_seconds"], .5354434553)
        self.assertEqual(eof["status"], "within_tolerance")
        self.assertIn("3001/3033", stderr.getvalue())
        self.assertTrue(any("3001/3033" in w for w in data["metrics"]["warnings"]))
        self.assertEqual(data["observations"][-1]["timestamp_ms"], last["timestamp_ms"])
        self.assertEqual(data["transitions"][-1]["reward"], 5)

    def test_eof_requires_both_time_and_fraction_bounds(self):
        for expected, decoded, fps in ((3033, 2900, 60), (30, 20, 60)):
            collector = TrajectoryCollector(BuildConfig())
            collector(observation(0))
            collector(observation(200))
            with self.assertRaisesRegex(ValueError, "stopped early"):
                collector.finish("win", {
                    "source_frame_count": expected, "decoded_frame_count": decoded,
                    "source_fps": fps,
                })

    def test_strict_eof_and_user_interruption_still_rejected(self):
        collector = TrajectoryCollector(BuildConfig(eof_tolerance_seconds=0))
        collector(observation(0))
        collector(observation(200))
        with self.assertRaisesRegex(ValueError, "stopped early"):
            collector.finish("win", {
                "source_frame_count": 3033, "decoded_frame_count": 3001,
                "source_fps": 60,
            })
        with self.assertRaisesRegex(ValueError, "Interrupted"):
            collector.finish("win", {"stopped_by_user": True})

    def test_invalid_eof_tolerance_is_rejected(self):
        for config in (
            BuildConfig(eof_tolerance_seconds=-1),
            BuildConfig(eof_tolerance_seconds=float("nan")),
            BuildConfig(eof_tolerance_fraction=1.1),
        ):
            with self.assertRaises(ValueError):
                config.validate()

    def test_sparse_cells_preserve_overlapping_units(self):
        unit = {"row": 16, "column": 9, "unit": "skeleton", "side": "enemy"}
        self.assertEqual(sparse_field([unit, unit])[0]["count"], 2)

    def test_save_roundtrip_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "battle.json"
            trajectory = assemble([event()])
            save_trajectory(path, trajectory)
            self.assertEqual(json.loads(path.read_text("utf-8")), trajectory)
            with self.assertRaises(FileExistsError):
                save_trajectory(path, {})
            self.assertEqual(json.loads(path.read_text("utf-8")), trajectory)

    def test_predictor_import_does_not_start_models_or_video(self):
        import cv2
        import ultralytics
        with patch.object(cv2, "VideoCapture", side_effect=AssertionError("Video opened")), \
             patch.object(ultralytics, "YOLO", side_effect=AssertionError("Model loaded")):
            module = importlib.import_module("predict_video_kalman")
            importlib.reload(module)
            self.assertTrue(callable(module.run_video_prediction))
        importlib.reload(module)

    def test_resolver_returns_start_of_empty_run(self):
        from collections import deque
        import predict_video_kalman as prediction
        history = deque([
            prediction.CardObservation(i, "knight" if i < 3 else "empty", .95)
            for i in range(19)
        ])
        match = prediction.resolve_played_card(
            [history, deque(), deque(), deque()], 9, 18, 30,
            sampling_fps=30, return_details=True,
        )
        self.assertEqual(match["empty_frame"], 3)
        self.assertEqual(match["card"], "knight")

    def test_cli_writes_complete_episode(self):
        import predict_video_kalman as prediction

        def fake_prediction(**kwargs):
            self.assertFalse(kwargs["write_video"])
            self.assertFalse(kwargs["display"])
            self.assertTrue(kwargs["synchronous_hp"])
            for i in range(19):
                frame = observation(i * 1000 / 30, "knight" if i < 6 else "empty")
                if i == 15:
                    frame["events"] = [event()]
                kwargs["observation_callback"](frame)
            return {"source_frame_count": 19, "decoded_frame_count": 19}

        with tempfile.TemporaryDirectory() as temporary:
            video = Path(temporary) / "battle.mp4"
            video.write_bytes(b"Mock video decoded by fake_prediction")
            output = Path(temporary) / "trajectories"
            with patch.object(prediction, "run_video_prediction", fake_prediction), redirect_stdout(io.StringIO()):
                status = main([str(video), "--result", "win", "--output-dir", str(output)])
            self.assertEqual(status, 0)
            files = list(output.glob("*.json"))
            self.assertEqual(len(files), 1)
            data = json.loads(files[0].read_text("utf-8"))
            self.assertEqual(data["result"], "win")
            self.assertEqual(data["metrics"]["valid_play_transitions"], 1)
            self.assertEqual(data["metadata"]["perspective"], "bottom_player")
            self.assertEqual(data["transitions"][-1]["reward"], 5)
            self.assertTrue(data["transitions"][-1]["terminated"])


if __name__ == "__main__":
    unittest.main()
