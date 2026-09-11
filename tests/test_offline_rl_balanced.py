"""Balanced imitation batches, natural-distribution scoring and resume."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest

import torch
from torch.utils.data import DataLoader

from test_offline_rl_dataset import battle
from offline_rl.balanced_sampling import BalancedActionBatchSampler
from offline_rl.dataset import TrajectoryDataset
from offline_rl.predict_action import ActionPredictor
from offline_rl.train import EpochMetrics, TrainConfig, make_parser, play_checkpoint_score, train


class BalancedTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.data = self.root / "data"
        self.data.mkdir()
        for name in ("a", "b"):
            (self.data / f"{name}.json").write_text(json.dumps(battle(name)), encoding="utf-8")
        self.dataset = TrajectoryDataset(self.data, sequence_length=3, max_units=3)
        self.options = ["--sampling", "balanced", "--device", "cpu", "--sequence-length", "3",
                        "--max-units", "3", "--d-model", "16", "--n-heads", "2", "--temporal-layers", "1",
                        "--ff-multiplier", "2", "--batch-size", "4", "--dropout", ".2", "--log-every", "0"]

    def sampler(self, dataset=None, batch_size=4, generator=None):
        return BalancedActionBatchSampler(dataset or self.dataset, batch_size,
                                           generator=generator or torch.Generator().manual_seed(17))

    def test_exact_class_balance_coverage_no_invalid_and_natural_history(self):
        sampler = self.sampler()
        self.assertEqual(sampler.stats["unique_plays"], 2)
        self.assertEqual(sampler.stats["unique_noops"], 4)
        self.assertEqual(sampler.stats["excluded_windows"], 2)
        indices = list(sampler)
        self.assertEqual(len(indices), 2)
        self.assertTrue(set(sampler.play_indices + sampler.noop_indices).issubset(set(sum(indices, []))))
        for batch in DataLoader(self.dataset, batch_sampler=self.sampler()):
            target = batch["targets"]["action_type"][batch["loss_mask"]]
            self.assertEqual(sorted(target.tolist()), [0, 0, 1, 1])
        # A later sample retains the unknown transition's observed state.
        sample = self.dataset[3]
        self.assertEqual(sample["step_indices"].tolist(), [1, 2, 3])

    def test_partial_positions_still_enter_play_pool(self):
        self.dataset.trajectories[0]["transitions"][1]["action"]["position_valid"] = False
        sampler = self.sampler()
        self.assertIn(1, sampler.play_indices)
        sample = self.dataset[1]
        self.assertTrue(sample["play_loss_mask"].any())
        self.assertFalse(sample["position_loss_mask"].any())

    def test_strided_window_endpoint_mapping(self):
        ds = TrajectoryDataset(self.data / "a.json", sequence_length=3, stride=2)
        self.assertEqual(list(ds.iter_window_endpoints()), [(0, 0, 0), (1, 0, 2), (2, 0, 3)])
        with self.assertRaisesRegex(ValueError, "both valid"):
            self.sampler(ds)

    def test_rng_resume_and_full_last_batch(self):
        generator = torch.Generator().manual_seed(42)
        sampler = self.sampler(batch_size=6, generator=generator)
        first = list(sampler)
        self.assertTrue(all(len(batch) == 6 for batch in first))
        self.assertEqual(len(first), 2)
        state = generator.get_state()
        second = list(sampler)
        restored = torch.Generator()
        restored.set_state(state)
        self.assertEqual(second, list(self.sampler(batch_size=6, generator=restored)))

    def test_invalid_configuration_and_missing_classes(self):
        for size in (0, 1, 3):
            with self.assertRaisesRegex(ValueError, "even"):
                self.sampler(batch_size=size)
        with self.assertRaisesRegex(ValueError, "supervise"):
            TrainConfig(sampling="balanced", supervise="all")
        with self.assertRaisesRegex(ValueError, "sampling"):
            TrainConfig(sampling="randomish")
        for trajectory in self.dataset.trajectories:
            trajectory["transitions"][1]["action_valid"] = False
        with self.assertRaisesRegex(ValueError, "both valid"):
            self.sampler()

    def test_f1_penalizes_both_misses_and_false_plays(self):
        metrics = EpochMetrics()
        metrics.counts.update(actions=100, plays=10, predicted_plays=20, true_positive_plays=5)
        summary = metrics.summary()
        self.assertEqual(summary["missed_plays"], 5)
        self.assertEqual(summary["false_positive_plays"], 15)
        self.assertAlmostEqual(summary["play_f1"], 1 / 3)
        good = dict(summary, loss=3.)
        all_wait = dict(good, play_f1=0., loss=.01)
        self.assertGreater(play_checkpoint_score(good), play_checkpoint_score(all_wait))
        self.assertGreater(play_checkpoint_score(dict(good, loss=2.)), play_checkpoint_score(good))
        self.assertIsNone(play_checkpoint_score(dict(good, plays=0)))
        self.assertIsNone(play_checkpoint_score(dict(good, plays=100)))

    def training(self, output, epochs=1, extra=()):
        args = make_parser().parse_args([str(self.data), "--output", str(output), "--epochs", str(epochs),
                                        *self.options, *extra])
        with redirect_stdout(io.StringIO()):
            return train(args)

    def test_no_validation_uses_unique_natural_eval_and_checkpoint_runs_in_predictor(self):
        output = self.root / "natural_eval"
        self.training(output, extra=["--validation-fraction", "0"])
        saved = torch.load(output / "last.pt", weights_only=True)
        record = saved["history"][-1]
        self.assertEqual(record["train"]["plays"], 4)  # Oversampled stream.
        self.assertEqual(record["train_eval"]["plays"], 2)
        self.assertEqual(record["train_eval"]["actions"], 6)
        self.assertEqual(saved["monitor"], "train_eval_loss")
        self.assertEqual(saved["play_monitor"], "train_eval_play_f1")
        predictor = ActionPredictor(output / "best_play.pt", device="cpu")
        predictor.observe(battle()["observations"][0])
        self.assertIn(predictor.predict()["type"], {"noop", "play"})

    def test_balanced_cpu_resume_is_bit_exact_and_preserves_best_play(self):
        full, resumed = self.root / "full", self.root / "resumed"
        self.training(full, epochs=2)
        self.training(resumed)
        args = make_parser().parse_args(["--resume", str(resumed / "last.pt"), "--epochs", "2", "--device", "cpu"])
        with redirect_stdout(io.StringIO()):
            train(args)
        a, b = [torch.load(path / "last.pt", weights_only=True) for path in (full, resumed)]
        for name in a["model_state_dict"]:
            torch.testing.assert_close(a["model_state_dict"][name], b["model_state_dict"][name], rtol=0, atol=0)
        self.assertEqual(a["history"][-1]["validation"], b["history"][-1]["validation"])
        self.assertEqual(b["history"][-1]["validation"]["actions"], 3)
        self.assertEqual(b["history"][-1]["validation"]["plays"], 1)
        self.assertEqual(a["best_play_score"], b["best_play_score"])
        best = torch.load(resumed / "best_play.pt", weights_only=True)
        self.assertEqual(best["epoch"], b["best_play_epoch"])
        self.assertEqual(best["best_play_score"], b["best_play_score"])
        self.assertEqual(best["play_monitor"], "validation_play_f1")
        args = make_parser().parse_args(["--resume", str(resumed / "last.pt"), "--sampling", "natural"])
        with self.assertRaisesRegex(ValueError, "Cannot change sampling"):
            train(args)

    def test_legacy_resume_defaults_to_natural_and_rejects_switch(self):
        output = self.root / "legacy"
        self.training(output, extra=["--sampling", "natural"])
        saved = torch.load(output / "last.pt", weights_only=True)
        saved["train_config"].pop("sampling")
        saved.pop("best_play_score")
        saved.pop("best_play_epoch")
        torch.save(saved, output / "last.pt")
        args = make_parser().parse_args(["--resume", str(output / "last.pt"), "--sampling", "balanced"])
        with self.assertRaisesRegex(ValueError, "Cannot change sampling"):
            train(args)
        args = make_parser().parse_args(["--resume", str(output / "last.pt"), "--epochs", "2", "--device", "cpu"])
        with redirect_stdout(io.StringIO()):
            train(args)
        self.assertEqual(torch.load(output / "last.pt", weights_only=True)["train_config"]["sampling"], "natural")


if __name__ == "__main__":
    unittest.main()
