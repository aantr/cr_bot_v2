"""Online/offline parity, bounded history, constrained decoding and CLI tests."""
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import torch
from torch.utils.data import default_collate

from test_offline_rl_dataset import battle, V2_DIR
from offline_rl.dataset import TrajectoryDataset, PREV_BOS, PREV_UNKNOWN
from offline_rl.starformer import StARformer, StARformerConfig
from offline_rl.predict_action import ActionPredictor


class PredictActionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.trajectory = battle()
        self.path = self.root / "battle.json"
        self.path.write_text(json.dumps(self.trajectory), encoding="utf-8")
        self.dataset = TrajectoryDataset(self.path, sequence_length=2, max_units=3)
        config = StARformerConfig.from_encoding_config(
            self.dataset.encoding_config(), d_model=16, n_heads=2, local_layers=1,
            temporal_layers=1, ff_multiplier=2, dropout=0,
        )
        torch.manual_seed(5)
        model = StARformer(config)
        self.saved = {"checkpoint_version": 1, "policy_kind": "imitation",
                      "encoding_config": self.dataset.encoding_config(), "model_config": config.to_dict(),
                      "model_state_dict": model.state_dict()}
        self.checkpoint = self.root / "best.pt"
        torch.save(self.saved, self.checkpoint)
        self.predictor = self.make_predictor()

    def make_predictor(self, **kwargs):
        return ActionPredictor(self.checkpoint, device="cpu",
                               field_layout=self.trajectory["metadata"]["field_layout"], **kwargs)

    def observed(self, index, predictor=None):
        predictor = predictor or self.predictor
        previous = self.trajectory["transitions"][index - 1] if index else None
        predictor.observe(self.trajectory["observations"][index], **({
            "previous_action": previous["action"], "previous_reward": previous["reward"],
            "previous_action_valid": previous["action_valid"]} if previous else {}))

    def force_play(self, predictor=None):
        predictor = predictor or self.predictor
        with torch.no_grad():
            predictor.model.action_head.weight.zero_()
            predictor.model.action_head.bias.copy_(torch.tensor([-10., 10.]))

    def test_online_offline_exact_feature_parity_after_window_rollover(self):
        for index in range(4):
            self.observed(index)
            online = self.predictor.build_batch()
            offline = default_collate([self.dataset[index]])
            for key, value in online.items():
                if key == "battle_id":
                    continue
                if isinstance(value, dict):
                    for name in value:
                        torch.testing.assert_close(value[name], offline[key][name], rtol=0, atol=0)
                else:
                    torch.testing.assert_close(value, offline[key], rtol=0, atol=0)
            self.assertLessEqual(self.predictor.history_length, 2)
        self.assertNotIn("targets", online)
        self.assertNotIn("return_to_go", online)
        self.assertNotEqual(online["previous_actions"][0, 0, 0].item(), PREV_BOS)

    def test_predictions_do_not_become_executed_actions(self):
        self.observed(0)
        self.force_play()
        a, b = self.predictor.predict(), self.predictor.predict()
        self.assertEqual(a, b)
        self.assertEqual(self.predictor.history_length, 1)
        self.predictor.observe(self.trajectory["observations"][1])
        batch = self.predictor.build_batch()
        self.assertEqual(batch["previous_actions"][0, 1].tolist(), [PREV_UNKNOWN, 0, 0, 0, 0])
        self.assertFalse(batch["previous_reward_mask"][0, 1])

    def test_reset_immutability_and_invalid_append(self):
        state = deepcopy(self.trajectory["observations"][0])
        self.predictor.observe(state)
        state["hand"][0]["card"] = "empty"
        self.assertGreater(self.predictor.build_batch()["states"]["hand_ids"][0, 0, 0], 2)
        with self.assertRaisesRegex(ValueError, "Timestamps"):
            self.observed(0)
        self.assertEqual(self.predictor.history_length, 1)
        action = {"type": "play", "card": "arrows", "slot": 1, "row": 1, "column": 1}
        with self.assertRaisesRegex(ValueError, "match"):
            self.predictor.observe(self.trajectory["observations"][1], previous_action=action)
        self.assertEqual(self.predictor.history_length, 1)
        self.predictor.reset("battle-two")
        with self.assertRaisesRegex(ValueError, "observe"):
            self.predictor.predict()
        self.observed(0)
        batch = self.predictor.build_batch()
        self.assertEqual(batch["previous_actions"][0, 0, 0], PREV_BOS)
        self.assertEqual(batch["step_indices"][0, 0], 0)
        self.assertEqual(batch["battle_id"], ["battle-two"])

    def test_costs_confidence_hash_cells_and_one_based_output(self):
        predictor = self.make_predictor(card_costs={"knight": 3, "arrows": 9})
        self.observed(0, predictor)
        self.force_play(predictor)
        cells = torch.zeros(32, 18, dtype=torch.bool)
        cells[15, 8] = True  # hash cell allowed
        result = predictor.predict(allowed_cells=cells)
        self.assertEqual((result["type"], result["slot"], result["card"], result["row"], result["column"]),
                         ("play", 1, "knight", 16, 9))
        self.assertTrue(result["constraints"]["elixir_checked"])
        self.assertEqual(result["confidence"]["cell"], 1)
        self.assertEqual(result["confidence"]["slot"], 1)
        result = predictor.predict(slot_costs=[9, 9, 9, 9])
        self.assertEqual(result["type"], "noop")
        self.assertEqual(result["reason"], "no_allowed_play")
        self.assertIsNone(result["row"])
        # Dynamic slot costs override the static table.
        result = predictor.predict(slot_costs=[None, 1, None, None])
        self.assertEqual(result["slot"], 2)

    def test_unknown_elixir_and_missing_cost_block_plays(self):
        for costs in ({}, {"knight": 3, "arrows": 3}):
            predictor = self.make_predictor(card_costs=costs)
            state = deepcopy(self.trajectory["observations"][0])
            state["elixir"]["value"] = None
            predictor.observe(state)
            self.force_play(predictor)
            self.assertEqual(predictor.predict()["reason"], "no_allowed_play")

    def test_empty_unknown_low_confidence_and_all_masks(self):
        predictor = self.make_predictor(min_card_confidence=.95)
        self.observed(0, predictor)
        self.force_play(predictor)
        self.assertEqual(predictor.predict()["type"], "noop")
        state = deepcopy(self.trajectory["observations"][0])
        for slot in state["hand"]:
            slot["card"] = "never_seen_before"
        self.predictor.observe(state)
        self.force_play()
        self.assertEqual(self.predictor.predict()["type"], "noop")
        self.predictor.reset()
        self.observed(0)
        self.assertEqual(self.predictor.predict(allowed_slots=[False] * 4)["type"], "noop")
        self.assertEqual(self.predictor.predict(allowed_cells=torch.zeros(4, 32, 18, dtype=torch.bool))["type"], "noop")
        with self.assertRaisesRegex(ValueError, "boolean"):
            self.predictor.predict(allowed_slots=[1, 1, 1, 1])
        with self.assertRaises(ValueError):
            self.predictor.predict(slot_costs=[float("nan")] * 4)

    def test_greedy_decode_matches_model_predict(self):
        self.observed(0)
        self.observed(1)
        self.force_play()
        actual = self.predictor.predict()
        expected = self.predictor.model.predict(self.predictor.build_batch())
        self.assertEqual(actual["slot"], expected["card_slot"].item() + 1)
        self.assertEqual(actual["row"], expected["row"].item() + 1)
        self.assertEqual(actual["column"], expected["column"].item() + 1)
        self.assertFalse(actual["constraints"]["elixir_checked"])

    def test_invalid_checkpoint_and_observation_overflow(self):
        bad = deepcopy(self.saved)
        bad["model_config"]["sequence_length"] = 9
        torch.save(bad, self.checkpoint)
        with self.assertRaisesRegex(ValueError, "disagree"):
            self.make_predictor()
        state = deepcopy(self.trajectory["observations"][0])
        state["units"] *= 2
        with self.assertRaisesRegex(ValueError, "max_units"):
            self.predictor.observe(state)
        self.assertEqual(self.predictor.history_length, 0)
        state = deepcopy(self.trajectory["observations"][0])
        state["units"][0]["row"] = 0
        with self.assertRaisesRegex(ValueError, "Invalid observation"):
            self.predictor.observe(state)

    def test_cli_replay_and_stream_reset(self):
        command = [sys.executable, str(V2_DIR / "offline_rl" / "predict_action.py"),
                   str(self.checkpoint), "--device", "cpu"]
        result = subprocess.run(command + ["--replay", str(self.path), "--limit", "2"],
                                capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stderr)
        rows = [json.loads(line) for line in result.stdout.splitlines()]
        self.assertEqual([row["history_length"] for row in rows], [1, 2])
        requests = [{"observation": self.trajectory["observations"][0]},
                    {"reset": True, "battle_id": "new"},
                    {"observation": self.trajectory["observations"][0]}]
        result = subprocess.run(command + ["--input", "-"],
                                input="\n".join(json.dumps(row) for row in requests),
                                capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stderr)
        rows = [json.loads(line) for line in result.stdout.splitlines()]
        self.assertEqual(rows[1], {"type": "reset", "battle_id": "new"})
        self.assertEqual(rows[2]["history_length"], 1)


if __name__ == "__main__":
    unittest.main()
