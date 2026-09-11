"""Video adapter tests without YOLO, OCR or GUI windows."""
from contextlib import redirect_stdout
from copy import deepcopy
import io
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from test_build_trajectories import observation, event
from offline_rl.build_trajectories import BuildConfig
from predict_video_actions import VideoActionAdvisor


class FakePredictor:
    def __init__(self):
        self.encoder = SimpleNamespace(sequence_length=3)
        self.history = []

    def reset(self, battle_id):
        self.history = []

    def observe(self, state, **feedback):
        self.history.append((deepcopy(state), deepcopy(feedback)))

    def predict(self, **kwargs):
        return {"type": "play", "card": "knight", "slot": 1, "row": 16, "column": 9,
                "observation_timestamp_ms": self.history[-1][0]["timestamp_ms"],
                "history_length": min(3, len(self.history)),
                "play_probability": .8, "play_threshold": .5, "reason": "model_play",
                "constraints": {"elixir_checked": False, "placement_mask_supplied": False},
                "confidence": {"action_type": .8}}


class VideoActionTests(unittest.TestCase):
    def setUp(self):
        self.predictor = FakePredictor()
        self.output = io.StringIO()
        self.advisor = VideoActionAdvisor(self.predictor, BuildConfig(), output=self.output)

    def feed(self, frame):
        with redirect_stdout(io.StringIO()):
            self.advisor(frame)

    def test_delayed_actual_play_is_backfilled_not_moved_to_current_step(self):
        self.feed(observation(0))
        self.feed(observation(200, "empty"))
        self.feed(observation(400, "empty"))
        self.assertFalse(self.predictor.history[-1][1]["previous_action_valid"])
        frame = observation(600, "arrows")
        frame["events"] = [event(action_time=200)]
        frame["events"][0]["confirmed_at_ms"] = 600
        self.feed(frame)
        state, feedback = self.predictor.history[1]
        self.assertEqual(state["timestamp_ms"], 200)
        self.assertTrue(feedback["previous_action_valid"])
        self.assertEqual(feedback["previous_action"]["type"], "play")
        self.assertEqual(feedback["previous_action"]["timestamp_ms"], 200)
        self.assertFalse(self.predictor.history[-1][1]["previous_action_valid"])
        self.assertEqual(self.advisor.latest["observation_timestamp_ms"], 600)

    def test_sampling_bounded_memory_and_no_prediction_feedback(self):
        for time_ms in range(0, 3001, 100):
            self.feed(observation(time_ms))
        self.assertEqual(self.advisor.prediction_count, 16)
        self.assertEqual(len(self.advisor.collector.observations), 4)
        self.assertEqual(len(self.output.getvalue().splitlines()), 16)
        # Even though the fake policy always recommends PLAY, no actual play is recorded.
        for _, feedback in self.predictor.history[1:]:
            self.assertEqual(feedback["previous_action"]["type"], "noop")
            self.assertFalse(feedback["previous_action_valid"])

    def test_hp_rewards_missing_elixir_and_disabled_hp(self):
        first, second = observation(0), observation(200)
        first["tower_hp"] = {"enemy_1": {"hp": 2000, "confirmed_at_ms": 0, "stale": False}}
        second["tower_hp"] = {"enemy_1": {"hp": 1900, "confirmed_at_ms": 200, "stale": False}}
        first["elixir_bar"]["value"] = None
        self.feed(first)
        self.feed(second)
        self.assertAlmostEqual(self.predictor.history[-1][1]["previous_reward"], .1)
        other = VideoActionAdvisor(self.predictor, BuildConfig(), hp_enabled=False)
        with redirect_stdout(io.StringIO()):
            other(first)
            other(second)
        self.assertIsNone(self.predictor.history[-1][1]["previous_reward"])

    def test_overlay_marks_hash_cell_and_does_not_modify_observation(self):
        self.feed(observation(0))
        frame_data = observation(0)
        before = deepcopy(frame_data)
        canvas = np.zeros((960, 540, 3), dtype=np.uint8)
        with patch("model_paths.BATTLEFIELDS", {(540, 960): (0, 0, 540, 960)}):
            self.advisor.annotate(canvas, frame_data)
        self.assertEqual(frame_data, before)
        self.assertTrue(canvas[450:481, 240:271].any())
        self.assertTrue(canvas[84:200].any())

    def test_overlay_shows_raw_play_probability_and_block_reason_for_wait(self):
        import cv2
        self.feed(observation(0))
        self.advisor.latest.update(type="noop", play_probability=.8, play_threshold=.35,
                                   reason="no_allowed_play")
        canvas = np.zeros((960, 540, 3), dtype=np.uint8)
        with patch("cv2.putText", wraps=cv2.putText) as draw:
            self.advisor.annotate(canvas, observation(0))
        labels = [call.args[1] for call in draw.call_args_list]
        self.assertTrue(any("P(play) 0.800" in label and "threshold 0.350" in label for label in labels))
        self.assertTrue(any("PLAY blocked" in label for label in labels))


if __name__ == "__main__":
    unittest.main()
