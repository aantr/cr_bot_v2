"""Masked loss, metrics, checkpoint/resume and CLI integration tests."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, default_collate

from test_offline_rl_dataset import battle, V2_DIR
from offline_rl.dataset import TrajectoryDataset
from offline_rl.starformer import StARformer, StARformerConfig
from offline_rl.train import EpochMetrics, TrainConfig, loss_components, make_parser, run_epoch, train


class TrainTests(unittest.TestCase):
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
        self.data = self.root / "data"
        self.data.mkdir()
        for identity in ("a", "b"):
            (self.data / f"{identity}.json").write_text(json.dumps(battle(identity)), encoding="utf-8")
        self.options = ["--device", "cpu", "--sequence-length", "4", "--max-units", "3",
                        "--d-model", "16", "--n-heads", "2", "--local-layers", "1",
                        "--temporal-layers", "1", "--ff-multiplier", "2",
                        "--batch-size", "2", "--dropout", "0.2", "--log-every", "0"]

    def run_training(self, output, *, epochs=1, extra=()):
        args = make_parser().parse_args([str(self.data), "--output", str(output),
                                        "--epochs", str(epochs), *self.options, *extra])
        with redirect_stdout(io.StringIO()):
            return train(args)

    def load(self, output, name="last.pt"):
        return torch.load(output / name, map_location="cpu", weights_only=True)

    def test_masked_loss_weighting_and_target_slot_coordinates(self):
        dataset = TrajectoryDataset(self.data / "a.json", sequence_length=4, max_units=3, supervise="all")
        batch = default_collate([dataset[3]])
        model = StARformer(StARformerConfig.from_encoding_config(
            dataset.encoding_config(), d_model=16, n_heads=2, temporal_layers=1))
        outputs = model(batch)
        parts = loss_components(outputs, batch, play_weight=3)
        self.assertEqual(set(parts), {"action_type", "card_slot", "row", "column"})
        self.assertEqual(parts["action_type"][1].item(), 5)  # 2 noops + weighted play.
        play, target = batch["play_loss_mask"], batch["targets"]
        expected = F.cross_entropy(outputs["row_by_slot"][play][:, 0], target["row"][play], reduction="sum")
        torch.testing.assert_close(parts["row"][0], expected)
        outputs["action_type"] = outputs["action_type"].clone()
        outputs["action_type"][~batch["loss_mask"]] = float("nan")
        for value, count in loss_components(outputs, batch).values():
            self.assertTrue(torch.isfinite(value / count))
        batch["play_loss_mask"].zero_()
        self.assertEqual(set(loss_components(outputs, batch)), {"action_type"})
        batch["loss_mask"].zero_()
        self.assertEqual(loss_components(outputs, batch), {})

    def test_metrics_use_predicted_slot_not_teacher_forced_slot(self):
        dataset = TrajectoryDataset(self.data / "a.json", sequence_length=4, max_units=3)
        batch = default_collate([dataset[1]])  # Only step 1 is supervised play.
        outputs = {"action_type": torch.zeros(1, 4, 2), "card_slot": torch.zeros(1, 4, 4),
                   "row_by_slot": torch.zeros(1, 4, 4, 32), "column_by_slot": torch.zeros(1, 4, 4, 18)}
        outputs["action_type"][0, 1, 1] = 10
        outputs["card_slot"][0, 1, 1] = 10  # Wrong slot, despite correct teacher-forced coords.
        outputs["row_by_slot"][0, 1, 0, 15] = 10
        outputs["column_by_slot"][0, 1, 0, 8] = 10
        metrics = EpochMetrics()
        metrics.update(outputs, batch, loss_components(outputs, batch))
        summary = metrics.summary()
        self.assertEqual(summary["play_recall"], 1)
        self.assertEqual(summary["play_slot_accuracy"], 0)
        self.assertEqual(summary["play_cell_accuracy"], 0)
        self.assertEqual(summary["full_action_accuracy"], 0)

    def test_empty_batches_skipped_and_empty_split_rejected(self):
        dataset = TrajectoryDataset(self.data / "a.json", sequence_length=4, max_units=3)
        model = StARformer(StARformerConfig.from_encoding_config(
            dataset.encoding_config(), d_model=16, n_heads=2, temporal_layers=1))
        result = run_epoch(model, DataLoader(dataset, batch_size=1), torch.device("cpu"),
                           TrainConfig(), log_every=0)
        self.assertEqual(result["skipped_batches"], 1)
        self.assertEqual(result["actions"], 3)
        with self.assertRaisesRegex(ValueError, "no supervised actions"):
            run_epoch(model, [default_collate([dataset[2]])], torch.device("cpu"), TrainConfig(), log_every=0)

    def test_checkpoints_split_and_safe_resume_match_uninterrupted(self):
        full, resumed = self.root / "full", self.root / "resumed"
        self.run_training(full, epochs=2)
        self.run_training(resumed)
        first = self.load(resumed)
        self.assertEqual(first["epoch"], 1)
        self.assertEqual(first["monitor"], "validation_loss")
        train_paths = {item["path"] for item in first["data_manifest"]["train"]}
        val_paths = {item["path"] for item in first["data_manifest"]["validation"]}
        self.assertTrue(train_paths and val_paths and train_paths.isdisjoint(val_paths))
        args = make_parser().parse_args(["--resume", str(resumed / "last.pt"), "--epochs", "2", "--device", "cpu"])
        with redirect_stdout(io.StringIO()):
            train(args)
        a, b = self.load(full), self.load(resumed)
        self.assertEqual(b["epoch"], 2)
        self.assertEqual(len(b["history"]), 2)
        for name in a["model_state_dict"]:
            torch.testing.assert_close(a["model_state_dict"][name], b["model_state_dict"][name], rtol=0, atol=0)
        self.assertEqual(a["history"][-1]["validation"], b["history"][-1]["validation"])
        self.assertEqual(self.load(resumed, "best.pt")["epoch"], b["best_epoch"])
        self.assertEqual(len(json.loads((resumed / "history.json").read_text())), 2)
        restored = StARformer(StARformerConfig(**b["model_config"])).eval()
        restored.load_state_dict(b["model_state_dict"])

    def test_resume_rejects_changed_data_and_hyperparameters_and_overwrite(self):
        output = self.root / "run"
        self.run_training(output)
        with self.assertRaisesRegex(ValueError, "not empty"):
            self.run_training(output)
        args = make_parser().parse_args(["--resume", str(output / "last.pt"), "--learning-rate", "0.01"])
        with self.assertRaisesRegex(ValueError, "Cannot change learning_rate"):
            train(args)
        path = self.data / "a.json"
        path.write_text(path.read_text() + "\n", encoding="utf-8")
        args = make_parser().parse_args(["--resume", str(output / "last.pt")])
        with self.assertRaisesRegex(ValueError, "changed or missing"):
            train(args)

    def test_early_stopping_preserves_best(self):
        output = self.root / "stop"
        self.run_training(output, epochs=5, extra=["--patience", "1", "--min-delta", "1000000"])
        checkpoint = self.load(output)
        self.assertEqual(checkpoint["epoch"], 2)
        self.assertEqual(checkpoint["stale_epochs"], 1)
        self.assertEqual(self.load(output, "best.pt")["epoch"], checkpoint["best_epoch"])

    def test_single_battle_cli_no_validation(self):
        output = self.root / "cli"
        result = subprocess.run([
            sys.executable, str(V2_DIR / "offline_rl" / "train.py"), str(self.data / "a.json"),
            "--output", str(output), "--epochs", "1", "--validation-fraction", "0",
            "--num-threads", "1", *self.options,
        ], capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("no validation", result.stdout)
        checkpoint = self.load(output)
        self.assertEqual(checkpoint["monitor"], "train_loss")
        self.assertIsNone(checkpoint["history"][0]["validation"])


if __name__ == "__main__":
    unittest.main()
