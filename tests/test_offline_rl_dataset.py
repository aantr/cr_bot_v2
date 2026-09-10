"""Dataset contract tests using small complete battle JSONs."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

import torch
from torch.utils.data import DataLoader

V2_DIR = Path(__file__).resolve().parents[1]
if str(V2_DIR) not in sys.path:
    sys.path.insert(0, str(V2_DIR))

from offline_rl.dataset import (
    EMPTY_CARD_ID, IGNORE_INDEX, PREV_BOS, PREV_NOOP, PREV_PLAY, PREV_UNKNOWN,
    UNKNOWN_ID, Normalization, TrajectoryDataset, Vocabulary,
    create_train_val_datasets, split_trajectory_files,
)


def battle(identity="battle-a", card="knight"):
    observations = []
    for step in range(5):
        observations.append({
            "timestamp_ms": step * 200.0, "frame_number": step * 6,
            "units": [
                {"track_id": tid, "unit": "skeleton", "side": "enemy",
                 "row": 16, "column": 9, "unit_confidence": .9,
                 "detector_confidence": .8, "predicted_by_kalman": False}
                for tid in (8, 99)
            ],
            "hand": [{"slot": slot, "card": (card if step < 2 else "empty") if slot == 1 else "arrows",
                      "confidence": .9} for slot in range(1, 5)],
            "elixir": {"value": 6.0},
            "tower_hp": {
                "ally_1": {"hp": None, "stale": True, "confidence": 0},
                "enemy_1": {"hp": 0, "stale": False, "confidence": .9,
                            "confirmed_at_ms": 0, "last_seen_at_ms": 0},
                "enemy_2": {"hp": 2000, "stale": True, "confidence": .8,
                            "confirmed_at_ms": 0, "last_seen_at_ms": 0},
            },
            "game_time_remaining_seconds": None, "phase": "unknown",
        })
    actions = [
        {"type": "noop"},
        {"type": "play", "card": card, "slot": 1, "row": 16, "column": 9,
         "timestamp_ms": 400, "confidence": .9},
        {"type": "unknown", "card": "secret-invalid-label", "slot": 4, "row": 9, "column": 7},
        {"type": "noop"},
    ]
    rewards = [1.0, 2.0, -3.0, 5.0]
    transitions = [
        {"state_index": i, "next_state_index": i + 1, "action": action,
         "action_valid": i != 2, "reward": rewards[i], "dt_ms": 200.0,
         "return_to_go": sum(rewards[i:]), "discounted_return_to_go": sum(rewards[i:]),
         "discount": .99 if i < 3 else 0.0, "terminated": i == 3, "truncated": False}
        for i, action in enumerate(actions)
    ]
    layout = ["." * 18 for _ in range(32)]
    layout[15] = "#" * 18
    return {
        "schema_version": 1, "coordinates": {"rows": 32, "columns": 18,
                                            "index_base": 1, "origin": "top-left"},
        "observations": observations, "transitions": transitions,
        "metadata": {"source_sha256": identity, "field_layout": layout,
                     "vocabulary": {"cards": {"0": card, "1": "arrows", "2": "empty"},
                                    "units": {"0": "skeleton"}}},
        "result": "win",
    }


class DatasetTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.path = self.write("a.json", battle())

    def tearDown(self):
        self.temporary.cleanup()

    def write(self, filename, data):
        path = self.root / filename
        path.write_text(json.dumps(data), encoding="utf-8")
        return path

    def test_windows_right_padding_and_target_head_masks(self):
        dataset = TrajectoryDataset(self.path, sequence_length=3)
        self.assertEqual(len(dataset), 4)
        first = dataset[0]
        self.assertEqual(first["attention_mask"].tolist(), [True, False, False])
        self.assertEqual(first["padding_mask"].tolist(), [False, True, True])
        self.assertEqual(first["step_indices"].tolist(), [0, -1, -1])
        self.assertEqual(first["previous_actions"][:, 0].tolist(), [PREV_BOS, 0, 0])
        for tensor in first["states"].values():
            self.assertTrue(torch.all(tensor[1:] == 0))
        self.assertEqual(first["targets"]["card_slot"].tolist(), [IGNORE_INDEX] * 3)
        play = dataset[1]
        self.assertEqual(play["loss_mask"].tolist(), [False, True, False])
        self.assertEqual(play["play_loss_mask"].tolist(), [False, True, False])
        self.assertEqual(play["targets"]["action_type"].tolist(), [IGNORE_INDEX, 1, IGNORE_INDEX])
        self.assertEqual(play["targets"]["card_slot"][1], 0)
        self.assertEqual(play["targets"]["row"][1], 15)
        self.assertEqual(play["targets"]["column"][1], 8)
        self.assertTrue(play["terrain_mask"][0, 15, 8])
        self.assertEqual(dataset[-1]["step_indices"].tolist(), [1, 2, 3])
        with self.assertRaises(IndexError):
            _ = dataset[4]

    def test_invalid_previous_action_is_unknown_not_original_label(self):
        dataset = TrajectoryDataset(self.path, sequence_length=4, supervise="all")
        sample = dataset[3]
        self.assertEqual(sample["previous_actions"][:, 0].tolist(),
                         [PREV_BOS, PREV_NOOP, PREV_PLAY, PREV_UNKNOWN])
        self.assertEqual(sample["previous_actions"][3].tolist(), [PREV_UNKNOWN, 0, 0, 0, 0])
        self.assertEqual(sample["action_valid"].tolist(), [True, True, False, True])
        self.assertEqual(sample["previous_action_valid"].tolist(), [False, True, True, False])
        self.assertTrue(torch.allclose(sample["previous_rewards"][:, 0],
                                       torch.tensor([0.0, .2, .4, -.6])))
        self.assertEqual(sample["previous_reward_mask"].tolist(), [False, True, True, True])
        self.assertEqual(sample["targets"]["action_type"][2], IGNORE_INDEX)
        self.assertNotIn("secret-invalid-label", dataset.vocabulary.cards)

    def test_shift_happens_before_window_slicing(self):
        dataset = TrajectoryDataset(self.path, sequence_length=1)
        self.assertEqual(dataset[2]["previous_actions"][0, 0], PREV_PLAY)
        self.assertAlmostEqual(dataset[2]["previous_rewards"][0, 0].item(), .4)
        self.assertEqual(dataset[3]["previous_actions"][0, 0], PREV_UNKNOWN)

    def test_future_and_current_labels_do_not_leak_into_state_or_previous_tokens(self):
        original = TrajectoryDataset(self.path, sequence_length=4)[1]
        edited = battle()
        edited["transitions"][1]["action"].update(card="arrows", slot=2, row=20)
        for transition in edited["transitions"][1:]:
            transition["reward"] = 70.0
            transition["return_to_go"] = 100.0
        edited["observations"][2]["elixir"]["value"] = 10.0
        other = self.write("future.json", edited)
        changed = TrajectoryDataset(other, sequence_length=4)[1]
        for key, tensor in original["states"].items():
            self.assertTrue(torch.equal(tensor, changed["states"][key]), key)
        self.assertTrue(torch.equal(original["previous_actions"], changed["previous_actions"]))
        self.assertTrue(torch.equal(original["previous_rewards"], changed["previous_rewards"]))
        self.assertNotEqual(original["targets"]["card_slot"][1], changed["targets"]["card_slot"][1])

    def test_units_overlap_and_track_ids_have_no_semantic_effect(self):
        dataset = TrajectoryDataset(self.path)
        state = dataset[0]["states"]
        self.assertAlmostEqual(state["grid_counts"][0, 1, 15, 8].item(), .2)
        self.assertEqual(state["unit_mask"][0].sum(), 2)
        self.assertTrue(torch.equal(state["unit_cells"][0, 0], torch.tensor([15, 8])))
        edited = battle()
        for observation in edited["observations"]:
            observation["units"].reverse()
            for unit in observation["units"]:
                unit["track_id"] += 100000
        other = TrajectoryDataset(self.write("ids.json", edited), vocabulary=dataset.vocabulary)
        for key, tensor in state.items():
            self.assertTrue(torch.equal(tensor, other[0]["states"][key]), key)
        with self.assertRaisesRegex(ValueError, "increase max_units"):
            TrajectoryDataset(self.path, max_units=1)

    def test_missing_hp_is_distinct_from_zero_and_stale_hp(self):
        states = TrajectoryDataset(self.path)[1]["states"]
        self.assertEqual(states["tower_hp"][1, 0], 0)
        self.assertFalse(states["tower_hp_known_mask"][1, 0])
        self.assertEqual(states["tower_hp"][1, 2], 0)
        self.assertTrue(states["tower_hp_known_mask"][1, 2])
        self.assertTrue(states["tower_hp_fresh_mask"][1, 2])
        self.assertTrue(states["tower_hp_known_mask"][1, 3])
        self.assertFalse(states["tower_hp_fresh_mask"][1, 3])
        self.assertAlmostEqual(states["tower_hp"][1, 3].item(), .2)
        self.assertAlmostEqual(states["tower_hp_age"][1, 3].item(), .02)
        self.assertFalse(states["remaining_time_mask"][1])
        self.assertAlmostEqual(states["elixir"][1, 0].item(), .6)
        self.assertEqual(TrajectoryDataset(self.path)[2]["states"]["hand_ids"][2, 0], EMPTY_CARD_ID)

    def test_validation_oov_uses_training_ids_and_masks_play_and_previous_token(self):
        train = TrajectoryDataset(self.path, sequence_length=4)
        edited = battle("battle-b", "golem")
        edited["observations"][0]["units"][0]["unit"] = "unseen-unit"
        validation = TrajectoryDataset(self.write("validation.json", edited),
                                       sequence_length=4, vocabulary=train.vocabulary)
        self.assertEqual(validation[0]["states"]["hand_ids"][0, 0], UNKNOWN_ID)
        self.assertIn(UNKNOWN_ID, validation[0]["states"]["unit_ids"][0].tolist())
        self.assertFalse(validation[1]["loss_mask"].any())
        self.assertEqual(validation[2]["previous_actions"][2, 0], PREV_UNKNOWN)
        self.assertEqual(train.vocabulary, validation.vocabulary)
        self.assertEqual(Vocabulary.from_dict(train.vocabulary.to_dict()), train.vocabulary)
        self.assertEqual(Normalization(**train.encoding_config()["normalization"]), train.normalization)

    def test_windows_do_not_cross_battles_and_dataloader_collates(self):
        self.write("b.json", battle("battle-b"))
        dataset = TrajectoryDataset(self.root, sequence_length=3)
        self.assertEqual(dataset[3]["battle_id"], "battle-a")
        self.assertEqual(dataset[4]["battle_id"], "battle-b")
        self.assertEqual(dataset[4]["step_indices"].tolist(), [0, -1, -1])
        self.assertEqual(dataset[4]["previous_actions"][0, 0], PREV_BOS)
        batch = next(iter(DataLoader(dataset, batch_size=2)))
        self.assertEqual(tuple(batch["states"]["grid_counts"].shape), (2, 3, 3, 32, 18))
        self.assertEqual(batch["targets"]["action_type"].dtype, torch.long)
        self.assertEqual(batch["padding_mask"].dtype, torch.bool)
        self.assertEqual(batch["states"]["elixir"].dtype, torch.float32)
        self.assertFalse(batch["causal_mask"][0, 2, 0])
        self.assertTrue(batch["causal_mask"][0, 0, 2])

    def test_split_groups_copies_and_is_reproducible_and_fits_train_only(self):
        self.write("copy-a.json", battle())
        self.write("b.json", battle("battle-b", "golem"))
        train_files, val_files = split_trajectory_files(self.root, .5, seed=19)
        self.assertEqual((train_files, val_files), split_trajectory_files(self.root, .5, seed=19))
        ids = lambda paths: {json.loads(p.read_text())["metadata"]["source_sha256"] for p in paths}
        self.assertFalse(ids(train_files) & ids(val_files))
        train, val = create_train_val_datasets(self.root, validation_fraction=.5, seed=19)
        self.assertIs(train.vocabulary, val.vocabulary)
        val_name = val.trajectories[0]["observations"][0]["hand"][0]["card"]
        self.assertNotIn(val_name, train.vocabulary.cards)

    def test_one_battle_requires_explicit_train_only_split(self):
        with self.assertRaisesRegex(ValueError, "at least two distinct"):
            split_trajectory_files(self.path)
        train, validation = create_train_val_datasets(self.path, validation_fraction=0)
        self.assertEqual(len(train), 4)
        self.assertIsNone(validation)

    def test_validation_rejects_corrupt_temporal_and_spatial_data(self):
        for modify in (
            lambda d: d.update(schema_version=9),
            lambda d: d["transitions"][0].update(next_state_index=2),
            lambda d: d["transitions"][0].update(dt_ms=10),
            lambda d: d["transitions"][0].update(terminated=True),
            lambda d: d["transitions"][-1].update(terminated=False),
            lambda d: d["observations"][1].update(timestamp_ms=0),
            lambda d: d["observations"][0]["units"][0].update(row=0),
            lambda d: d["observations"][0]["elixir"].update(value=float("nan")),
            lambda d: d["observations"][0]["tower_hp"]["enemy_1"].update(confirmed_at_ms=500),
            lambda d: d["observations"][0]["hand"][0].update(slot=2),
        ):
            data = battle()
            modify(data)
            path = self.write("bad.json", data)
            with self.assertRaisesRegex(ValueError, "Invalid trajectory"):
                TrajectoryDataset(path)

    def test_stride_keeps_final_transition(self):
        dataset = TrajectoryDataset(self.path, sequence_length=2, stride=2)
        self.assertEqual(len(dataset), 3)
        self.assertEqual(dataset[-1]["step_indices"].tolist(), [2, 3])
        self.assertTrue(dataset[-1]["terminated"][1])


if __name__ == "__main__":
    unittest.main()
