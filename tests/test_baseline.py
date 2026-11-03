import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from kilter.baseline.config import Config, DataConfig, ModelConfig, TrainConfig, load_config
from kilter.baseline.data import BOS, EOS, FRAME, PAD, Tokenizer, collate, load_arrays, prepare, split_for_uuid, v_grade
from kilter.baseline.experiment import LengthBucketSampler, evaluate_run, sample, train
from kilter.baseline.model import AutoregressiveTransformer, parameter_count, resolve_model, token_loss


class RepresentationTests(unittest.TestCase):
    def test_roles_frames_and_numeric_grades(self):
        tokenizer = Tokenizer([100, 200])
        frames = [[{"hold_token": 200, "hold_use_type": 2}, {"hold_token": 100, "hold_use_type": 1}],
                  [{"hold_token": 200, "hold_use_type": 3}]]
        encoded = tokenizer.encode(frames)
        self.assertEqual(encoded, [BOS, 5, 10, FRAME, 11, EOS])
        self.assertEqual(tokenizer.decode(encoded), [sorted(frames[0], key=lambda h: h["hold_token"]), frames[1]])
        self.assertEqual(v_grade("V12"), 12.0)
        self.assertIsNone(v_grade(None))
        with self.assertRaises(ValueError):
            v_grade("6c/V5")

    def test_padding_does_not_change_loss(self):
        logits = torch.randn(1, 2, 12)
        targets = torch.tensor([[5, EOS]])
        padded = torch.cat([logits, torch.randn(1, 3, 12)], dim=1)
        padded_targets = torch.cat([targets, torch.full((1, 3), PAD)], dim=1)
        torch.testing.assert_close(token_loss(logits, targets), token_loss(padded, padded_targets))
        x, y, conditions = collate([(np.array([BOS, 5, EOS]), np.array([5, 40 / 90])),
                                    (np.array([BOS, EOS]), np.array([2, 1.0]))])
        self.assertEqual(y.tolist(), [[5, EOS], [EOS, PAD]])
        self.assertEqual(x.tolist(), [[BOS, 5], [BOS, PAD]])
        self.assertAlmostEqual(conditions[0, 1].item(), 40 / 90)

    def test_split_is_stable_under_row_order_and_angle_changes(self):
        config = DataConfig()
        keys = [f"climb-{i}" for i in range(1000)]
        original = {key: split_for_uuid(key, config) for key in keys}
        self.assertEqual(original, {key: split_for_uuid(key, config) for key in reversed(keys)})
        self.assertEqual(set(original.values()), {0, 1, 2})

    def test_config_overrides_and_typos(self):
        config = load_config("configs/baseline.toml", ["model.positional_embeddings=false", "train.seed=7"])
        self.assertFalse(config.model.positional_embeddings)
        self.assertEqual(config.train.seed, 7)
        with self.assertRaises(ValueError):
            load_config("configs/baseline.toml", ["model.position_embeddings=false"])


class ModelTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(1)

    def test_causality_and_both_conditions(self):
        for positional in (True, False):
            model = AutoregressiveTransformer(ModelConfig(d_model=16, n_heads=4, n_layers=2,
                                                          dropout=0, positional_embeddings=positional), 24, 16).eval()
            tokens = torch.tensor([[BOS, 5, 6, 7]])
            conditions = torch.tensor([[5., 40 / 90]])
            baseline = model(tokens, conditions)
            changed = tokens.clone()
            changed[0, -1] = 10
            torch.testing.assert_close(baseline[:, :-1], model(changed, conditions)[:, :-1])
            for c in (torch.tensor([[6., 40 / 90]]), torch.tensor([[5., 50 / 90]])):
                self.assertFalse(torch.allclose(baseline, model(tokens, c)))

    def test_sizing_counts_tied_weights_and_uses_training_rows(self):
        resolved, target = resolve_model(ModelConfig(), 1908, 128, 305000)
        model = AutoregressiveTransformer(resolved, 1908, 128)
        self.assertEqual(target, 152500)
        self.assertLess(abs(parameter_count(model) - target) / target, 0.08)
        self.assertIs(model.token_embedding.weight, model.lm_head.weight)
        no_pos, _ = resolve_model(ModelConfig(positional_embeddings=False), 1908, 128, 305000)
        self.assertEqual(no_pos.d_model, resolved.d_model)
        self.assertEqual(parameter_count(model) - parameter_count(AutoregressiveTransformer(no_pos, 1908, 128)),
                         128 * resolved.d_model)

    def test_masked_projection_matches_unpadded_loss_and_gradients(self):
        model = AutoregressiveTransformer(ModelConfig(d_model=16, n_layers=1, dropout=0), 12, 8)
        x = torch.tensor([[BOS, 5, 6], [BOS, 7, PAD]])
        y = torch.tensor([[5, 6, EOS], [7, EOS, PAD]])
        c = torch.tensor([[3., 0.5], [4., 0.2]])
        full_loss = token_loss(model(x, c), y)
        full_loss.backward()
        gradients = {key: p.grad.clone() for key, p in model.named_parameters()}
        model.zero_grad(set_to_none=True)
        mask = y != PAD
        masked_loss = token_loss(model(x, c, output_mask=mask), y[mask])
        masked_loss.backward()
        torch.testing.assert_close(full_loss, masked_loss)
        for key, parameter in model.named_parameters():
            torch.testing.assert_close(parameter.grad, gradients[key])

    def test_can_overfit_a_tiny_batch(self):
        model = AutoregressiveTransformer(ModelConfig(d_model=16, n_layers=1, dropout=0), 12, 8)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.02)
        x, y = torch.tensor([[BOS, 5, 6]] * 4), torch.tensor([[5, 6, EOS]] * 4)
        c = torch.tensor([[3., 0.5]] * 4)
        initial = token_loss(model(x, c), y).item()
        for _ in range(60):
            optimizer.zero_grad()
            loss = token_loss(model(x, c), y)
            loss.backward()
            optimizer.step()
        self.assertLess(token_loss(model(x, c), y).item(), initial / 10)


class LifecycleTests(unittest.TestCase):
    def test_prepare_train_resume_evaluate_and_sample(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            vocabulary = root / "holds.csv"
            vocabulary.write_text("hold_token\n100\n200\n")
            source = root / "climbs.jsonl"
            rows = []
            for i in range(60):
                for angle in (0, 45, 90):
                    rows.append({"climb_uuid": str(i), "climb_angle_id": f"{i}:{angle}", "angle": angle,
                                 "grade_hueco": "V5", "grade_status": "reported",
                                 "frames": [[{"hold_token": 100, "hold_use_type": 2},
                                             {"hold_token": 200, "hold_use_type": 3}]]})
            source.write_text("".join(json.dumps(row) + "\n" for row in rows))
            config = Config(output_dir=str(root / "runs"),
                            data=DataConfig(source=str(source), vocabulary=str(vocabulary), cache_dir=str(root / "cache"),
                                            train_fraction=0.7, validation_fraction=0.15, max_seq_len=8),
                            model=ModelConfig(d_model=8, n_layers=1, n_heads=2, dropout=0.1),
                            train=TrainConfig(epochs=2, batch_size=32, threads=1, log_every=100))
            directory, metadata = prepare(config.data)
            arrays = load_arrays(directory)
            for uuid in set(arrays["uuids"]):
                self.assertEqual(len(set(arrays["splits"][arrays["uuids"] == uuid])), 1)
            self.assertEqual(set(arrays["conditions"][:, 1]), {0, 0.5, 1})
            from kilter.baseline.data import ClimbDataset
            dataset = ClimbDataset(arrays, 0)
            sampler = LengthBucketSampler(dataset, 13, torch.Generator().manual_seed(7))
            first_epoch = list(sampler)
            self.assertEqual(sorted(i for batch in first_epoch for i in batch), list(range(len(dataset))))
            self.assertEqual(len(first_epoch), len(sampler))
            self.assertNotEqual(first_epoch, list(sampler))
            self.assertEqual(prepare(config.data)[1]["fingerprint"], metadata["fingerprint"])
            # Interrupt immediately after the first epoch's durable last checkpoint.
            from kilter.baseline import experiment
            original = experiment.checkpoint_save
            def interrupt(path, state):
                original(path, state)
                if path.name == "last.pt":
                    raise InterruptedError("simulated interruption")
            with patch("kilter.baseline.experiment.checkpoint_save", side_effect=interrupt):
                with self.assertRaises(InterruptedError):
                    train(config)
            interrupted = next((root / "runs").glob("*/last.pt"))
            resumed_dir = train(config, interrupted)
            uninterrupted_dir = train(config)
            resumed = torch.load(resumed_dir / "last.pt", weights_only=False)
            uninterrupted = torch.load(uninterrupted_dir / "last.pt", weights_only=False)
            for key in resumed["model"]:
                torch.testing.assert_close(resumed["model"][key], uninterrupted["model"][key], rtol=0, atol=0)
            self.assertEqual(resumed["global_step"], uninterrupted["global_step"])
            metrics = evaluate_run(resumed_dir / "best.pt")
            self.assertTrue(np.isfinite(metrics["loss"]))
            self.assertFalse(metrics["partial"])
            generated = sample(resumed_dir / "best.pt", 5, 90, seed=1)
            self.assertEqual(generated[0]["normalized_angle"], 1.0)
            self.assertEqual(generated, sample(resumed_dir / "best.pt", 5, 90, seed=1))
            # A changed source produces a new cache rather than stale data reuse.
            source.write_text(source.read_text() + json.dumps({**rows[0], "climb_uuid": "new"}) + "\n")
            self.assertNotEqual(prepare(config.data)[1]["fingerprint"], metadata["fingerprint"])


if __name__ == "__main__":
    unittest.main()
