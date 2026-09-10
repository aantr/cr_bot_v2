"""Policy contract, causal isolation, backward and action selection tests."""
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest

import torch
from torch.nn import functional as F
from torch.utils.data import default_collate

from test_offline_rl_dataset import battle
from offline_rl.dataset import TrajectoryDataset
from offline_rl.starformer import StARformer, StARformerConfig


class StARformerTests(unittest.TestCase):
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
        path = Path(self.tmp.name) / "battle.json"
        path.write_text(json.dumps(battle()), encoding="utf-8")
        self.dataset = TrajectoryDataset(path, sequence_length=4, max_units=3, supervise="all")
        self.batch = default_collate([self.dataset[0], self.dataset[3]])
        self.config = StARformerConfig.from_encoding_config(
            self.dataset.encoding_config(), d_model=16, n_heads=2,
            local_layers=2, temporal_layers=2, ff_multiplier=2, dropout=0,
        )
        torch.manual_seed(7)
        self.model = StARformer(self.config).eval()

    def assert_outputs_close(self, a, b):
        for key in a:
            torch.testing.assert_close(a[key], b[key], atol=2e-6, rtol=2e-5)

    def test_shapes_padding_and_input_unchanged(self):
        before = copy.deepcopy(self.batch)
        out = self.model(self.batch)
        expected = {"action_type": (2, 4, 2), "card_slot": (2, 4, 4),
                    "row_by_slot": (2, 4, 4, 32), "column_by_slot": (2, 4, 4, 18)}
        for key, shape in expected.items():
            self.assertEqual(tuple(out[key].shape), shape)
            self.assertTrue(torch.isfinite(out[key]).all())
            self.assertEqual(out[key][0, 1:].count_nonzero(), 0)
        for key in before["states"]:
            torch.testing.assert_close(before["states"][key], self.batch["states"][key])
        torch.testing.assert_close(before["previous_actions"], self.batch["previous_actions"])

    def test_backward_all_heads_and_encoders(self):
        self.model.train()
        out = self.model(self.batch)
        target = self.batch["targets"]
        mask, play = self.batch["loss_mask"], self.batch["play_loss_mask"]
        loss = F.cross_entropy(out["action_type"][mask], target["action_type"][mask])
        loss = loss + F.cross_entropy(out["card_slot"][play], target["card_slot"][play])
        slots = target["card_slot"][play]
        indices = torch.arange(slots.numel())
        for name in ("row", "column"):
            logits = out[name + "_by_slot"][play][indices, slots]
            loss = loss + F.cross_entropy(logits, target[name][play])
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        for name, param in self.model.named_parameters():
            self.assertIsNotNone(param.grad, name)
            self.assertTrue(torch.isfinite(param.grad).all(), name)
        for module in (self.model.patch_projection, self.model.unit_embedding,
                       self.model.action_head, self.model.row_head, self.model.column_head):
            self.assertGreater(module.weight.grad.abs().sum().item(), 0)

    def test_causal_no_future_steps(self):
        changed = copy.deepcopy(self.batch)
        changed["states"]["grid_counts"][1, 2:] += 20
        changed["states"]["hand_ids"][1, 2:] = 1
        changed["previous_rewards"][1, 2:] += 12
        # Even a caller-supplied noncausal mask cannot disable model causality.
        changed["causal_mask"].zero_()
        with torch.no_grad():
            a, b = self.model(self.batch), self.model(changed)
        self.assert_outputs_close({k: v[1, :2] for k, v in a.items()},
                                  {k: v[1, :2] for k, v in b.items()})
        self.assertFalse(torch.allclose(a["action_type"][1, 2:], b["action_type"][1, 2:]))

    def test_training_only_fields_not_inputs(self):
        observed = {k: self.batch[k] for k in (
            "states", "terrain_mask", "attention_mask", "previous_actions",
            "previous_action_valid", "previous_rewards", "previous_reward_mask")}
        with torch.no_grad():
            self.assert_outputs_close(self.model(self.batch), self.model(observed))

    def test_masked_units_and_missing_values_do_not_leak(self):
        changed = copy.deepcopy(self.batch)
        s = changed["states"]
        missing = ~s["unit_mask"]
        for name in ("unit_ids", "unit_sides", "unit_cells", "unit_features"):
            s[name][missing] = 999
        s["tower_hp"][~s["tower_hp_known_mask"]] = float("nan")
        s["time_features"][..., 2][~s["remaining_time_mask"]] = 123
        changed["previous_rewards"][~changed["previous_reward_mask"]] = float("nan")
        with torch.no_grad():
            self.assert_outputs_close(self.model(self.batch), self.model(changed))

    def test_unit_order_invariance_and_empty_field(self):
        changed = copy.deepcopy(self.batch)
        for name in ("unit_ids", "unit_sides", "unit_cells", "unit_features", "unit_mask"):
            changed["states"][name] = changed["states"][name].flip(2)
        with torch.no_grad():
            self.assert_outputs_close(self.model(self.batch), self.model(changed))
        changed["states"]["unit_mask"].zero_()
        changed["states"]["grid_counts"].zero_()
        for value in self.model(changed).values():
            self.assertTrue(torch.isfinite(value).all())

    def test_predict_masks_hash_cell_last_real_and_noop(self):
        with torch.no_grad():
            self.model.action_head.weight.zero_()
            self.model.action_head.bias.copy_(torch.tensor([-10., 10.]))
        cells = torch.zeros(2, 4, 32, 18, dtype=torch.bool)
        cells[:, 0, 15, 8] = True  # # is allowed. Slot 0 is empty at last step of sample 1.
        result = self.model.predict(self.batch, allowed_cells=cells)
        self.assertEqual(result["action_type"].tolist(), [1, 0])
        self.assertEqual(result["card_slot"].tolist(), [0, -1])
        self.assertEqual(result["row"].tolist(), [15, -1])
        self.assertEqual(result["column"].tolist(), [8, -1])
        self.assertEqual(result["card_id"][0], self.batch["states"]["hand_ids"][0, 0, 0])
        result = self.model.predict(self.batch, allowed_slots=torch.zeros(2, 4, dtype=torch.bool))
        self.assertEqual(result["action_type"].tolist(), [0, 0])
        with self.assertRaises(ValueError):
            self.model.predict(self.batch, allowed_cells=torch.zeros(32, 18))
        self.model.train()
        with self.assertRaises(RuntimeError):
            self.model.predict(self.batch)

    def test_checkpoint_round_trip_and_independent_layers(self):
        a, b = self.model.temporal_transformer.layers
        self.assertFalse(torch.equal(a.self_attn.in_proj_weight, b.self_attn.in_proj_weight))
        buffer = io.BytesIO()
        torch.save({"config": self.config.to_dict(), "model": self.model.state_dict()}, buffer)
        buffer.seek(0)
        checkpoint = torch.load(buffer, weights_only=True)
        restored = StARformer(StARformerConfig(**checkpoint["config"])).eval()
        restored.load_state_dict(checkpoint["model"])
        with torch.no_grad():
            self.assert_outputs_close(self.model(self.batch), restored(self.batch))

    def test_invalid_config_and_sequence_masks(self):
        for override in ({"d_model": 15}, {"num_cards": 2}, {"dropout": 1}, {"n_heads": 0}):
            with self.assertRaises(ValueError):
                StARformerConfig(**(self.config.to_dict() | override))
        changed = copy.deepcopy(self.batch)
        changed["attention_mask"][0] = False
        with self.assertRaises(ValueError):
            self.model(changed)
        changed["attention_mask"][0] = torch.tensor([True, False, True, False])
        with self.assertRaises(ValueError):
            self.model(changed)
        small = StARformer(StARformerConfig(**(self.config.to_dict() | {"sequence_length": 2})))
        with self.assertRaises(ValueError):
            small(self.batch)


if __name__ == "__main__":
    unittest.main()
