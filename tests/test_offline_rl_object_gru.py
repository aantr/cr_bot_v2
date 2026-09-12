"""Object-set invariance, causal history, losses, unchanged JSON, resume and inference."""
from contextlib import redirect_stdout
from copy import deepcopy
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import torch
from torch.utils.data import default_collate

from test_offline_rl_dataset import V2_DIR, battle
from offline_rl.dataset import TrajectoryDataset
from offline_rl.predict_action import ActionPredictor
from offline_rl.train import EpochMetrics
from offline_rl.object_gru.features import auxiliary_features, object_features
from offline_rl.object_gru.model import ARCHITECTURE, ObjectGRUConfig, ObjectGRUPolicy
from offline_rl.object_gru.predict_action import ObjectGRUPredictor
from offline_rl.object_gru.train import ObjectTrainConfig, loss_components, make_parser, train


class ObjectGRUTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.path = self.root / "battle.json"
        self.raw = battle()
        self.path.write_text(json.dumps(self.raw), encoding="utf-8")
        self.dataset = TrajectoryDataset(self.path, sequence_length=4, max_units=3)
        self.config = ObjectGRUConfig.from_encoding_config(self.dataset.encoding_config(), d_model=16,
                            n_heads=2, object_layers=1, ff_multiplier=2, class_embedding=8,
                            state_dim=32, gru_hidden=16, gru_layers=2, dropout=0.)
        torch.manual_seed(42)
        self.model = ObjectGRUPolicy(self.config).eval()

    def checkpoint(self):
        path = self.root / "policy.pt"
        torch.save({"checkpoint_version": 1, "policy_kind": "imitation", "architecture": ARCHITECTURE,
                    "history_mode": "observations_only", "model_config": self.config.to_dict(),
                    "encoding_config": self.dataset.encoding_config(),
                    "model_state_dict": self.model.state_dict()}, path)
        return path

    def assert_outputs_close(self, actual, expected):
        for key in actual:
            torch.testing.assert_close(actual[key], expected[key], rtol=1e-5, atol=1e-6)

    def test_default_architecture_and_feature_semantics(self):
        batch = default_collate([self.dataset[1]])
        ids, features, mask = object_features(batch)
        self.assertEqual(features.shape, (1, 4, 3, 6))
        self.assertEqual(features[0, 0, 0, 0], -1)
        self.assertAlmostEqual(float(features[0, 0, 0, 1]), 8.5 / 18)
        self.assertAlmostEqual(float(features[0, 0, 0, 2]), 15.5 / 32)
        self.assertEqual(ids[~mask].count_nonzero(), 0)
        towers, hand, hand_extra, globals_ = auxiliary_features(batch)
        self.assertEqual(towers.shape, (1, 4, 20))
        # Missing ally HP != measured enemy HP=0 (availability differs).
        self.assertEqual(towers[0, 0, 1], 0)
        self.assertEqual(towers[0, 0, 11], 1)
        self.assertEqual(hand_extra.shape, (1, 4, 4, 3))
        self.assertEqual(globals_.shape, (1, 4, 4))
        default = ObjectGRUPolicy(ObjectGRUConfig(num_cards=5, num_units=3))
        self.assertEqual(default.gru.input_size, 512)
        self.assertEqual(default.gru.hidden_size, 256)
        self.assertEqual(default.gru.num_layers, 2)
        self.assertEqual(default.position_head.out_features, 576)
        self.assertEqual(len(default.object_transformer.layers), 3)
        self.assertEqual(ObjectTrainConfig().sampling, "balanced")

    @torch.no_grad()
    def test_object_permutation_padding_and_empty_arena(self):
        batch = default_collate([self.dataset[1]])
        expected = self.model(batch)
        altered = deepcopy(batch)
        for key in ("unit_ids", "unit_sides", "unit_cells", "unit_features", "unit_mask"):
            altered["states"][key] = altered["states"][key][:, :, [2, 1, 0]]
        self.assert_outputs_close(self.model(altered), expected)
        # Padding junk must not even reach an embedding lookup.
        mask = batch["states"]["unit_mask"]
        batch["states"]["unit_ids"][~mask] = 100000
        batch["states"]["unit_features"][~mask] = float("nan")
        self.assert_outputs_close(self.model(batch), expected)
        batch["states"]["unit_mask"].zero_()
        for value in self.model(batch).values():
            self.assertTrue(torch.isfinite(value).all())

    @torch.no_grad()
    def test_causality_and_no_label_reward_feedback_leakage(self):
        batch = default_collate([self.dataset[3]])
        expected = self.model(batch)
        modified = deepcopy(batch)
        for key in ("rewards", "return_to_go", "discounted_return_to_go", "previous_rewards", "previous_actions"):
            modified[key].fill_(99)
        for tensor in modified["targets"].values():
            tensor.fill_(99)
        self.assert_outputs_close(self.model(modified), expected)
        modified["states"]["elixir"][:, 2:] = .1
        modified["states"]["hand_ids"][:, 2:] = 1
        modified["states"]["unit_features"][:, 2:] = 0
        actual = self.model(modified)
        for key in actual:
            torch.testing.assert_close(actual[key][:, :2], expected[key][:, :2], rtol=1e-5, atol=1e-6)

    @torch.no_grad()
    def test_last_real_state_matches_short_unpadded_history(self):
        batch = default_collate([self.dataset[1]])
        short = {"states": {k: v[:, :2] for k, v in batch["states"].items()},
                 "attention_mask": batch["attention_mask"][:, :2]}
        padded, unpadded = self.model(batch), self.model(short)
        for key in padded:
            torch.testing.assert_close(padded[key][:, 1], unpadded[key][:, 1], rtol=1e-5, atol=1e-6)
        batch["attention_mask"][0] = torch.tensor([1, 0, 1, 0], dtype=torch.bool)
        with self.assertRaisesRegex(ValueError, "right padding"):
            self.model(batch)

    def test_losses_joint_position_partial_and_wait(self):
        batch = default_collate([self.dataset[1]])
        output = self.model(batch)
        parts = loss_components(output, batch)
        self.assertEqual(set(parts), {"action_type", "card_slot", "position"})
        expected = .5 * torch.nn.functional.cross_entropy(output["position"][0, 1].unsqueeze(0),
                                                          torch.tensor([15 * 18 + 8]))
        torch.testing.assert_close(parts["position"][0] / parts["position"][1], expected)
        batch["position_loss_mask"].zero_()
        parts = loss_components(output, batch)
        self.assertEqual(set(parts), {"action_type", "card_slot"})
        sum(value / count for value, count in parts.values()).backward()
        self.assertIsNone(self.model.position_head.weight.grad)
        self.assertGreater(float(self.model.card_head.weight.grad.abs().sum()), 0)
        wait = default_collate([self.dataset[0]])
        self.assertEqual(set(loss_components(self.model(wait), wait)), {"action_type"})
        wait["loss_mask"].zero_()
        self.assertEqual(loss_components(self.model(wait), wait), {})

    def test_metrics_use_joint_cell_not_independent_marginals(self):
        batch = default_collate([self.dataset[1]])
        outputs = {"action_type": torch.zeros(1, 4, 2), "card_slot": torch.zeros(1, 4, 4),
                   "position": torch.zeros(1, 4, 576)}
        outputs["action_type"][0, 1, 1] = 10
        outputs["card_slot"][0, 1, 0] = 10
        outputs["position"][0, 1, 15 * 18 + 8] = 10
        metrics = EpochMetrics()
        metrics.update(outputs, batch, loss_components(outputs, batch))
        self.assertEqual(metrics.summary()["play_full_accuracy"], 1)
        self.assertEqual(metrics.summary()["play_f1"], 1)

    @torch.no_grad()
    def test_online_offline_parity_sliding_history_and_reset(self):
        predictor = ObjectGRUPredictor(self.checkpoint(), device="cpu")
        for index, observation in enumerate(self.raw["observations"][:-1]):
            predictor.observe(observation)
            self.assert_outputs_close(self.model(default_collate([self.dataset[index]])),
                                      predictor.model(predictor.build_batch()))
        predictor.observe(self.raw["observations"][-1])
        self.assertEqual(predictor.history_length, 4)
        self.assertEqual(predictor.build_batch()["step_indices"].tolist(), [[1, 2, 3, 4]])
        prediction = predictor.predict()
        self.assertEqual(prediction["history_length"], 4)
        self.assertEqual(predictor.history_length, 4)
        predictor.reset("new")
        self.assertEqual(predictor.history_length, 0)
        predictor.observe(self.raw["observations"][0])
        self.assertEqual(predictor.build_batch()["step_indices"][0, 0], 0)

    def test_prediction_cell_masks_costs_and_architecture_guard(self):
        with torch.no_grad():
            for head in (self.model.action_head, self.model.card_head, self.model.position_head):
                head.weight.zero_()
                head.bias.zero_()
            self.model.action_head.bias[1] = 10
            self.model.card_head.bias[0] = 10
            self.model.position_head.bias[15 * 18 + 8] = 10
        checkpoint = self.checkpoint()
        with self.assertRaisesRegex(ValueError, "architecture"):
            ActionPredictor(checkpoint, device="cpu")
        predictor = ObjectGRUPredictor(checkpoint, device="cpu")
        predictor.observe(self.raw["observations"][0])
        result = predictor.predict()
        self.assertEqual((result["type"], result["slot"], result["row"], result["column"]), ("play", 1, 16, 9))
        cells = torch.zeros(4, 32, 18, dtype=torch.bool)
        cells[1, 31, 17] = True
        result = predictor.predict(allowed_cells=cells)
        self.assertEqual((result["slot"], result["row"], result["column"]), (2, 32, 18))
        self.assertEqual(predictor.predict(slot_costs=[7, 8, 9, None])["reason"], "no_allowed_play")
        self.assertEqual(predictor.predict(allowed_slots=[False] * 4)["type"], "noop")

    def training(self, output, epochs, resume=None):
        options = ["--device", "cpu", "--epochs", str(epochs), "--log-every", "0"]
        if resume:
            options += ["--resume", str(resume)]
        else:
            options += [str(self.path), "--output", str(output), "--validation-fraction", "0",
                        "--batch-size", "2", "--sequence-length", "4", "--max-units", "3",
                        "--d-model", "16", "--n-heads", "2", "--object-layers", "1",
                        "--ff-multiplier", "2", "--state-dim", "32", "--gru-hidden", "16",
                        "--class-embedding", "8", "--dropout", ".2"]
        with redirect_stdout(io.StringIO()):
            train(make_parser().parse_args(options))

    def test_train_checkpoint_resume_bit_exact_and_json_unchanged(self):
        original = self.path.read_bytes()
        direct, resumed = self.root / "direct", self.root / "resumed"
        self.training(direct, 2)
        self.training(resumed, 1)
        self.training(resumed, 2, resumed / "last.pt")
        a = torch.load(direct / "last.pt", weights_only=True)
        b = torch.load(resumed / "last.pt", weights_only=True)
        for name in a["model_state_dict"]:
            torch.testing.assert_close(a["model_state_dict"][name], b["model_state_dict"][name], rtol=0, atol=0)
        self.assertEqual(a["architecture"], ARCHITECTURE)
        self.assertEqual(a["history_mode"], "observations_only")
        self.assertEqual(a["history"][-1]["train_eval"]["actions"], 3)
        self.assertEqual(a["history"][-1]["train_eval"], b["history"][-1]["train_eval"])
        self.assertEqual(original, self.path.read_bytes())
        for name in ("best.pt", "last.pt", "best_play.pt"):
            predictor = ObjectGRUPredictor(direct / name, device="cpu")
            predictor.observe(self.raw["observations"][0])
            self.assertIn(predictor.predict()["type"], {"noop", "play"})

    def test_direct_cli_replay_and_video_help(self):
        checkpoint = self.checkpoint()
        result = subprocess.run([sys.executable, str(V2_DIR / "offline_rl/object_gru/predict_action.py"),
                                 str(checkpoint), "--replay", str(self.path), "--device", "cpu", "--limit", "2"],
                                capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len(result.stdout.splitlines()), 2)
        for filename in ("predict_video_object_gru.py", "offline_rl/object_gru/train.py"):
            result = subprocess.run([sys.executable, str(V2_DIR / filename), "--help"],
                                    capture_output=True, text=True, timeout=60)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("--device", result.stdout)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_cuda_backward_exits_cleanly_without_global_cudnn_change(self):
        # Run in a child: a cuDNN RNN native shutdown failure cannot be caught
        # as a Python exception inside this test runner.
        code = """
import sys
import torch
from torch.utils.data import default_collate
from offline_rl.dataset import TrajectoryDataset
from offline_rl.train import to_device
from offline_rl.object_gru.model import ObjectGRUConfig, ObjectGRUPolicy
from offline_rl.object_gru.train import loss_components
torch.set_num_threads(1)
data = TrajectoryDataset(sys.argv[1], sequence_length=4, max_units=3)
config = ObjectGRUConfig.from_encoding_config(data.encoding_config(), d_model=16,
    n_heads=2, object_layers=1, state_dim=32, gru_hidden=16, dropout=.1)
model = ObjectGRUPolicy(config).cuda().train()
batch = to_device(default_collate([data[1]]), torch.device('cuda'))
before = torch.backends.cudnn.enabled
parts = loss_components(model(batch), batch)
sum(value / count for value, count in parts.values()).backward()
torch.cuda.synchronize()
assert torch.backends.cudnn.enabled == before
assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
print('CUDA backward OK', flush=True)
"""
        result = subprocess.run([sys.executable, "-c", code, str(self.path)], cwd=V2_DIR,
                                capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("CUDA backward OK", result.stdout)


if __name__ == "__main__":
    unittest.main()
