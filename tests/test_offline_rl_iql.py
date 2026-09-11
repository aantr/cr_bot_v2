"""IQL math, temporal alignment, gradient isolation and checkpoint integration."""
from contextlib import redirect_stdout
from copy import deepcopy
import io
import json
import math
from pathlib import Path
import tempfile
import unittest

import torch
from torch.utils.data import DataLoader, default_collate

from test_offline_rl_dataset import battle
from offline_rl.dataset import PREV_BOS, PREV_UNKNOWN, TrajectoryDataset, Vocabulary
from offline_rl.history_features import policy_inputs
from offline_rl.iql_dataset import IQLDataset
from offline_rl.iql import (IQL, action_log_probability, advantage_weights,
                            bellman_target, expectile_loss)
from offline_rl.predict_action import ActionPredictor
from offline_rl.starformer import StARformerConfig
from offline_rl.train import to_device
from offline_rl.train_iql import IQLTrainConfig, make_optimizers, make_parser, run_epoch, train


class IQLTests(unittest.TestCase):
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
        for identity in ("a", "b"):
            (self.data / f"{identity}.json").write_text(json.dumps(battle(identity)), encoding="utf-8")
        self.base = TrajectoryDataset(self.data / "a.json", sequence_length=3, max_units=3)
        self.dataset = IQLDataset(self.base)
        self.config = StARformerConfig.from_encoding_config(
            self.base.encoding_config(), d_model=16, n_heads=2, temporal_layers=1,
            ff_multiplier=2, dropout=0.)
        self.options = ["--device", "cpu", "--sequence-length", "3", "--max-units", "3",
                        "--d-model", "16", "--n-heads", "2", "--temporal-layers", "1",
                        "--ff-multiplier", "2", "--batch-size", "2", "--dropout", ".2", "--log-every", "0"]

    def test_pairs_include_terminal_observation_without_skipping_unknown_steps(self):
        self.assertEqual(self.dataset.indices, [(0, 0), (0, 1), (0, 3)])
        self.assertEqual(self.dataset.stats["invalid"], 1)
        self.assertEqual(self.dataset.stats["nonzero_rewards_excluded"], 1)
        play = self.dataset[1]
        self.assertEqual(play["history"]["step_indices"].tolist(), [0, 1, -1])
        self.assertEqual(play["next_history"]["step_indices"].tolist(), [0, 1, 2])
        self.assertEqual(play["action"].tolist()[2:], [1, 16, 9])
        last = self.dataset[2]
        self.assertEqual(last["history"]["step_indices"].tolist(), [1, 2, 3])
        self.assertEqual(last["next_history"]["step_indices"].tolist(), [2, 3, 4])
        self.assertTrue(last["terminated"])
        self.assertEqual(last["discount"], 0)
        self.assertEqual(last["reward"], 1.)  # Raw terminal reward 5 / scale 5.

    def test_partial_play_and_excluded_terminal_are_reported(self):
        raw = self.base.trajectories[0]
        raw["transitions"][1]["action"]["position_valid"] = False
        raw["transitions"][3]["action_valid"] = False
        dataset = IQLDataset(self.base)
        self.assertEqual(len(dataset), 1)
        self.assertEqual(dataset.stats["partial_position"], 1)
        self.assertEqual(dataset.stats["terminal_excluded"], 1)
        self.assertEqual(dataset.stats["nonzero_rewards_excluded"], 3)

    def test_no_future_fields_in_history(self):
        before = self.dataset[1]["history"]
        raw = self.base.trajectories[0]
        raw["observations"][2]["elixir"]["value"] = 0
        raw["transitions"][1]["reward"] = 99
        after = self.dataset[1]["history"]
        for name in before["states"]:
            torch.testing.assert_close(before["states"][name], after["states"][name])
        self.assertNotIn("targets", after)
        self.assertNotIn("return_to_go", after)
        self.assertFalse(after["previous_reward_mask"].any())
        self.assertEqual(after["previous_actions"][:, 0].tolist(), [PREV_BOS, PREV_UNKNOWN, 0])

    def test_observation_mask_is_nonmutating(self):
        sample = self.base[2]
        old = sample["previous_actions"].clone()
        new = policy_inputs(sample, "observations_only")
        torch.testing.assert_close(sample["previous_actions"], old)
        self.assertFalse(new["previous_action_valid"].any())
        self.assertFalse(new["previous_rewards"].any())

    def test_reject_duplicate_battle_and_subsampling(self):
        (self.data / "copy.json").write_text(json.dumps(battle("a")), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            IQLDataset(TrajectoryDataset(self.data, sequence_length=3, max_units=3))
        with self.assertRaisesRegex(ValueError, "stride"):
            IQLDataset(TrajectoryDataset(self.data / "a.json", stride=2))

    def test_unknown_vocabulary_play_is_not_noop(self):
        vocab = Vocabulary(("<pad>", "<unknown>", "empty", "arrows"), self.base.vocabulary.units)
        base = TrajectoryDataset(self.data / "a.json", vocabulary=vocab, sequence_length=3, max_units=3)
        dataset = IQLDataset(base)
        self.assertEqual(dataset.stats["unknown_card"], 1)
        self.assertEqual(dataset.stats["noop"], 2)

    def test_iql_scalar_formulas_terminal_and_time_discount(self):
        torch.testing.assert_close(expectile_loss(torch.tensor([-2., 2.]), .7), torch.tensor([1.2, 2.8]))
        torch.testing.assert_close(bellman_target(torch.tensor([1., 1., 1.]),
                                                  torch.tensor([0., .5, .9]), torch.tensor([999., 4., 4.])),
                                   torch.tensor([1., 3., 4.6]))
        adv = torch.tensor([0., 1., 10000.], requires_grad=True)
        weights = advantage_weights(adv, 3., 100.)
        torch.testing.assert_close(weights, torch.tensor([1., math.exp(3), 100.]))
        self.assertFalse(weights.requires_grad)

    def test_truncation_keeps_stored_bootstrap_discount(self):
        last = self.base.trajectories[0]["transitions"][-1]
        last.update(terminated=False, truncated=True, discount=.8)
        sample = IQLDataset(self.base)[-1]
        self.assertFalse(sample["terminated"])
        self.assertTrue(sample["truncated"])
        torch.testing.assert_close(bellman_target(sample["reward"], sample["discount"], torch.tensor(2.)),
                                   torch.tensor(2.6))

    def test_joint_log_probability_noop_and_target_slot(self):
        history = default_collate([self.dataset[0]["history"], self.dataset[1]["history"]])
        outputs = {"action_type": torch.zeros(2, 3, 2, requires_grad=True),
                   "card_slot": torch.zeros(2, 3, 4, requires_grad=True),
                   "row_by_slot": torch.zeros(2, 3, 4, 32, requires_grad=True),
                   "column_by_slot": torch.zeros(2, 3, 4, 18, requires_grad=True)}
        action = torch.tensor([[0, -100, -100, -100, -100], [1, 4, 3, 16, 9]])
        logp = action_log_probability(outputs, history, action)
        torch.testing.assert_close(logp, torch.tensor([-math.log(2), -math.log(2 * 4 * 32 * 18)]))
        (-logp.sum()).backward()
        grad = outputs["row_by_slot"].grad
        self.assertFalse(grad[0].any())
        self.assertTrue(grad[1, 1, 2].any())
        self.assertFalse(grad[1, 1, 0].any())

    def test_gradients_do_not_cross_actor_value_critics_or_targets(self):
        model = IQL(self.config)
        batch = default_collate([self.dataset[0], self.dataset[1]])
        networks = ("actor", "value", "q1", "q2", "target_q1", "target_q2")
        for loss_name in ("actor", "value", "q1", "q2"):
            model.zero_grad(set_to_none=True)
            losses, _ = model.losses(batch)
            losses[loss_name].backward()
            for name in networks:
                has_gradient = any(p.grad is not None for p in getattr(model, name).parameters())
                self.assertEqual(has_gradient, name == loss_name, (loss_name, name))
        parameter_sets = [{id(p) for p in getattr(model, name).parameters()} for name in networks]
        self.assertEqual(sum(map(len, parameter_sets)), len(set.union(*parameter_sets)))

    def test_target_ema_includes_history_encoder_and_stays_eval(self):
        model = IQL(self.config, target_rate=.1).train()
        self.assertFalse(model.target_q1.training)
        before = model.target_q1.backbone.summary_token.detach().clone()
        with torch.no_grad():
            model.q1.backbone.summary_token.add_(1.)
        model.update_targets()
        torch.testing.assert_close(model.target_q1.backbone.summary_token, before + .1)
        self.assertTrue(all(not p.requires_grad for p in model.target_q1.parameters()))

    def save_actor(self, path):
        model = IQL(self.config)
        torch.save({"checkpoint_version": 1, "iql_version": 1, "policy_kind": "iql",
                    "history_mode": "observations_only", "model_config": self.config.to_dict(),
                    "encoding_config": self.base.encoding_config(), "model_state_dict": model.actor.state_dict()}, path)

    def test_live_and_offline_iql_histories_match_without_delayed_feedback(self):
        path = self.root / "actor.pt"
        self.save_actor(path)
        raw = self.base.trajectories[0]
        predictor = ActionPredictor(path, device="cpu", field_layout=raw["metadata"]["field_layout"])
        for step, observation in enumerate(raw["observations"]):
            predictor.observe(observation)  # No retrospective action labels/rewards.
            online = predictor.build_batch()
            offline = default_collate([self.dataset.history(0, step)])
            for name in online["states"]:
                torch.testing.assert_close(online["states"][name], offline["states"][name])
            for name in ("previous_actions", "previous_rewards", "previous_reward_mask", "step_indices"):
                torch.testing.assert_close(online[name], offline[name])
        result = predictor.predict()
        self.assertIn(result["type"], {"play", "noop"})
        self.assertEqual(predictor.history_mode, "observations_only")

    def training(self, output, epochs=1, extra=()):
        args = make_parser().parse_args([str(self.data), "--output", str(output), "--epochs", str(epochs),
                                        *self.options, *extra])
        with redirect_stdout(io.StringIO()):
            return train(args)

    def test_video_iql_adapter_preserves_absolute_steps_and_observation_history(self):
        from offline_rl.build_trajectories import BuildConfig
        from predict_video_actions import VideoActionAdvisor
        path = self.root / "video_actor.pt"
        self.save_actor(path)
        raw = self.base.trajectories[0]
        predictor = ActionPredictor(path, device="cpu", field_layout=raw["metadata"]["field_layout"])
        advisor = VideoActionAdvisor(predictor, BuildConfig())
        for step, observation in enumerate(raw["observations"]):
            frame = deepcopy(observation)
            frame["elixir_bar"] = frame.pop("elixir")
            frame.update(processing_fps=30., events=[])
            with redirect_stdout(io.StringIO()):
                advisor(frame)
            online = predictor.build_batch()
            offline = default_collate([self.dataset.history(0, step)])
            torch.testing.assert_close(online["step_indices"], offline["step_indices"])
            for key in offline["states"]:
                torch.testing.assert_close(online["states"][key], offline["states"][key])
        self.assertEqual(advisor.prediction_count, 5)
        self.assertEqual(predictor.history_length, 3)

    def test_training_checkpoint_prediction_and_bit_exact_cpu_resume(self):
        continuous, resumed = self.root / "continuous", self.root / "resumed"
        self.training(continuous, epochs=2)
        self.training(resumed)
        args = make_parser().parse_args(["--resume", str(resumed / "last.pt"), "--epochs", "2",
                                        "--device", "cpu", "--log-every", "0"])
        with redirect_stdout(io.StringIO()):
            train(args)
        a = torch.load(continuous / "last.pt", weights_only=True)
        b = torch.load(resumed / "last.pt", weights_only=True)
        self.assertEqual(b["epoch"], 2)
        self.assertEqual(b["policy_kind"], "iql")
        self.assertEqual(b["monitor"], "validation_policy_nll")
        for key in a["iql_state_dict"]:
            torch.testing.assert_close(a["iql_state_dict"][key], b["iql_state_dict"][key], rtol=0, atol=0, msg=key)
        self.assertEqual(a["history"][1]["train"], b["history"][1]["train"])
        predictor = ActionPredictor(resumed / "best.pt", device="cpu")
        predictor.observe(battle()["observations"][0])
        self.assertIn(predictor.predict()["type"], {"play", "noop"})
        # A changed data file must not be silently accepted by resume.
        (self.data / "a.json").write_text(json.dumps(battle("changed")), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "changed or missing"):
            train(args)

    def test_warm_start_and_single_battle_no_validation(self):
        path = self.root / "initial.pt"
        self.save_actor(path)
        saved_actor = torch.load(path, weights_only=True)
        saved_actor["policy_kind"] = "imitation"
        saved_actor.pop("iql_version")
        saved_actor.pop("history_mode")
        torch.save(saved_actor, path)
        # Model config is inherited; dropout=0 in the saved actor.
        args = make_parser().parse_args([str(self.data / "a.json"), "--output", str(self.root / "warm"),
                                        "--init-actor", str(path), "--epochs", "1", "--device", "cpu",
                                        "--validation-fraction", "0", "--log-every", "0"])
        with redirect_stdout(io.StringIO()):
            train(args)
        saved = torch.load(self.root / "warm" / "last.pt", weights_only=True)
        self.assertEqual(saved["monitor"], "train_policy_nll")
        self.assertEqual(saved["data_stats"]["train"]["play"], 1)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_cuda_training_update_and_inference(self):
        model = IQL(self.config).cuda()
        config = IQLTrainConfig(batch_size=2)
        optimizers = make_optimizers(model, config)
        metrics = run_epoch(model, DataLoader(self.dataset, batch_size=2), torch.device("cuda"),
                            config, optimizers=optimizers, log_every=0)
        self.assertTrue(math.isfinite(metrics["actor_loss"]))
        batch = to_device(default_collate([self.dataset[1]]), torch.device("cuda"))
        prediction = model.actor.eval().predict(batch["history"])
        self.assertIn(prediction["action_type"].item(), (0, 1))


if __name__ == "__main__":
    unittest.main()
