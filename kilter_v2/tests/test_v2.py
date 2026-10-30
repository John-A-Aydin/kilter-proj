import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from kilter_v2.config import Config, DataConfig, ModelConfig, TrainConfig, load_config
from kilter_v2.data import (BOS, EOS, FINISH, FOOT, HAND, PAD, SEP, START, Board, ClimbDataset, Collator,
                            build_example, climb_problem, given_mask, load_arrays, prepare, split_for_uuid)
from kilter_v2.experiment import evaluate, load_run, train
from kilter_v2.model import ClimbTransformer, weighted_token_loss
from kilter_v2.sampling import allowed_tokens, climb_checks, generate, validate_given

# Six holds on a 2x3 grid: ids 10..15, y rising with id.
ROWS = [{"id": 10 + i, "x": (i % 2) / 2, "y": (i // 2) / 3, "kickboard": int(i < 2), "set": 1 + 19 * (i % 2)}
        for i in range(6)]


def hold(id_, role):
    return {"hold_token": id_, "hold_use_type": role}


class RepresentationTests(unittest.TestCase):
    def setUp(self):
        self.board = Board(ROWS)

    def test_canonical_order_is_role_then_bottom_to_top(self):
        climb = [hold(15, FINISH), hold(10, FOOT), hold(13, HAND), hold(11, START), hold(12, HAND)]
        tokens = self.board.encode(climb)
        self.assertEqual([self.board.decode([t])[0]["hold_token"] for t in tokens], [11, 12, 13, 15, 10])
        self.assertEqual(sorted(self.board.decode(tokens), key=str), sorted(climb, key=str))
        self.assertEqual(len(set(self.board.rank[4:])), 24)
        self.assertTrue((self.board.rank[:4] == -1).all())

    def test_given_holds_are_context_and_only_the_rest_is_predicted(self):
        tokens = self.board.encode([hold(11, START), hold(12, HAND), hold(15, FINISH), hold(10, FOOT)])
        mask = given_mask(np.array(tokens), "starts")
        x, y = build_example(tokens, mask)
        self.assertEqual(x, [BOS, tokens[0], SEP, tokens[1], tokens[2], tokens[3]])
        self.assertEqual(y, [PAD, PAD, tokens[1], tokens[2], tokens[3], EOS])
        x, y = build_example(tokens, given_mask(np.array(tokens), "scratch"))
        self.assertEqual(x, [BOS, SEP, *tokens])
        self.assertEqual(y, [PAD, *tokens, EOS])
        feet = given_mask(np.array(tokens), "starts_feet")
        self.assertEqual(feet.tolist(), [True, False, False, True])

    def test_training_mixture_covers_scratch_starts_and_random(self):
        tokens = np.array(self.board.encode([hold(11, START), hold(12, HAND), hold(15, FINISH), hold(10, FOOT)]))
        collator = Collator(probabilities=(0.3, 0.45, 0.25), seed=3)
        examples = [(tokens, np.array([5, 0.5], dtype=np.float32), 1)] * 400
        x, y, conditions, weights = collator(examples)
        given_counts = (x == SEP).float().argmax(1) - 1
        self.assertTrue((given_counts == 0).any() and (given_counts >= 1).any())
        self.assertTrue(((x == SEP).sum(1) == 1).all())
        self.assertEqual(weights.tolist(), [1.0] * 400)
        heavy = Collator(weighting="log").row_weight(1000)
        self.assertEqual(heavy, 6.0)

    def test_filters_and_split_match_baseline(self):
        from kilter.data import split_for_uuid as baseline_split
        config = DataConfig()
        for key in (f"climb-{i}" for i in range(500)):
            self.assertEqual(split_for_uuid(key, config), baseline_split(key, config))
        self.assertEqual(climb_problem([[hold(10, START)], [hold(11, FINISH)]], 40), "multi_frame")
        self.assertEqual(climb_problem([[hold(10, START), hold(10, FINISH)]], 40), "repeated_hold")
        self.assertEqual(climb_problem([[hold(10, HAND), hold(11, FINISH)]], 40), "start_count")
        self.assertEqual(climb_problem([[hold(10, START), hold(11, HAND)]], 40), "finish_count")
        self.assertEqual(climb_problem([[hold(10, START), hold(11, FINISH)]], 1), "too_many_holds")
        self.assertIsNone(climb_problem([[hold(10, START), hold(11, FINISH)]], 40))

    def test_config_overrides_and_typos(self):
        path = Path(__file__).parents[1] / "configs" / "default.toml"
        self.assertEqual(load_config(path).to_dict(), Config().to_dict())
        config = load_config(path, ["model.d_model=64", "train.ascent_weighting=log"])
        self.assertEqual((config.model.d_model, config.train.ascent_weighting), (64, "log"))
        for bad in (["model.width=3"], ["train.p_scratch=0.9"], ["train.ascent_weighting=sqrt"]):
            with self.assertRaises(ValueError):
                load_config(path, bad)


class ModelTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(1)
        self.board = Board(ROWS)
        self.model = ClimbTransformer(ModelConfig(d_model=16, n_heads=2, n_layers=2, dropout=0,
                                                  coordinate_frequencies=2),
                                      self.board.features(2), 10).eval()

    def test_causal_conditions_and_positions_restart_at_sep(self):
        tokens = torch.tensor([[BOS, 5, SEP, 9, 13]])
        c = torch.tensor([[5., 40 / 90]])
        out = self.model(tokens, c)
        self.assertEqual(out.shape, (1, 5, self.board.vocab_size))
        changed = tokens.clone()
        changed[0, -1] = 17
        torch.testing.assert_close(out[:, :-1], self.model(changed, c)[:, :-1])
        for other in (torch.tensor([[6., 40 / 90]]), torch.tensor([[5., 50 / 90]])):
            self.assertFalse(torch.allclose(out, self.model(tokens, other)))
        # A hold after SEP sees the same position embedding whatever the number of given holds.
        positions = []
        self.model.position.register_forward_hook(lambda _m, inputs, _o: positions.append(inputs[0]))
        self.model(torch.tensor([[BOS, 5, 9, SEP, 13, PAD]]), c)
        self.assertEqual(positions[-1].tolist(), [[0, 1, 2, 0, 1, 2]])

    def test_tied_table_shares_geometry_between_roles(self):
        table = self.model.token_table()
        self.assertEqual(table.shape, (self.board.vocab_size, 16))
        # Moving one hold changes all four of its role embeddings and nothing else.
        self.model.hold_features[3, 0] += 1
        changed = (self.model.token_table() - table).abs().sum(1) > 0
        self.assertEqual(torch.nonzero(changed).flatten().tolist(), [self.board.token(3, role) for role in range(4)])

    def test_weighted_loss_and_padding(self):
        x = torch.tensor([[BOS, SEP, 5], [BOS, SEP, PAD]])
        y = torch.tensor([[PAD, 5, EOS], [PAD, EOS, PAD]])
        c = torch.tensor([[3., 0.5], [4., 0.2]])
        logits = self.model(x, c, output_mask=y != PAD)
        equal, losses = weighted_token_loss(logits, y, torch.tensor([1., 1.]))
        torch.testing.assert_close(equal, losses.mean())
        heavy, _ = weighted_token_loss(logits, y, torch.tensor([3., 1.]))
        torch.testing.assert_close(heavy, (3 * losses[0] + 3 * losses[1] + losses[2]) / 7)

    def test_can_overfit_a_conditional_example(self):
        model = ClimbTransformer(ModelConfig(d_model=16, n_heads=2, n_layers=1, dropout=0, coordinate_frequencies=2),
                                 self.board.features(2), 10)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.02)
        tokens = self.board.encode([hold(11, START), hold(13, HAND), hold(15, FINISH)])
        x, y = build_example(tokens, given_mask(np.array(tokens), "starts"))
        x, y, c, w = torch.tensor([x] * 4), torch.tensor([y] * 4), torch.tensor([[3., 0.5]] * 4), torch.ones(4)
        initial = weighted_token_loss(model(x, c, output_mask=y != PAD), y, w)[0].item()
        for _ in range(80):
            optimizer.zero_grad()
            loss = weighted_token_loss(model(x, c, output_mask=y != PAD), y, w)[0]
            loss.backward()
            optimizer.step()
        self.assertLess(loss.item(), initial / 10)


class SamplingTests(unittest.TestCase):
    def setUp(self):
        self.board = Board(ROWS)

    def allowed(self, **state):
        mask = allowed_tokens(self.board, **{"used": set(), "last_rank": -1, "starts": 0, "finishes": 0,
                                             "count": 0, "max_holds": 10, **state})
        return {(self.board.holds[self.board.split(t)[0]], self.board.split(t)[1]) if t >= 4 else t
                for t in np.flatnonzero(mask)}

    def test_rules_require_a_start_then_a_finish_and_respect_order(self):
        self.assertEqual(self.allowed(), {(i, START) for i in range(10, 16)})
        after_start = self.allowed(used={1}, last_rank=self.board.rank[self.board.token(1, START)], starts=1, count=1)
        self.assertNotIn(EOS, after_start)
        self.assertNotIn((11, HAND), after_start)
        self.assertFalse(any(role == FOOT for _, role in after_start))
        self.assertIn((12, START), after_start)
        self.assertNotIn((10, START), after_start)  # Same height, further left: earlier in order.
        done = self.allowed(used={1, 2, 5}, last_rank=self.board.rank[self.board.token(2, HAND)], starts=1,
                            finishes=1, count=3)
        self.assertIn(EOS, done)
        self.assertEqual({h[1] for h in done if h != EOS}, {HAND, FINISH, FOOT})
        self.assertNotIn((12, FINISH), done)
        top = self.allowed(used={1, 5}, last_rank=self.board.rank[self.board.token(5, FINISH)], starts=1,
                           finishes=1, count=2)
        self.assertEqual({h[1] for h in top if h != EOS}, {FOOT})
        full = self.allowed(used={1, 5}, last_rank=0, starts=2, finishes=2, count=10)
        self.assertEqual(full, {EOS})
        reserve = self.allowed(used={1}, last_rank=self.board.rank[self.board.token(1, START)], starts=1, count=9)
        self.assertTrue(all(role == FINISH for _, role in reserve))
        self.assertTrue(PAD not in done and BOS not in done and SEP not in done)

    def test_generation_keeps_given_holds_and_never_repeats_them(self):
        class Adversary(torch.nn.Module):
            """Always prefers EOS, then the given holds again, then start tokens."""
            max_len = 12

            def __init__(self, board):
                super().__init__()
                self.p = torch.nn.Parameter(torch.zeros(1))
                self.board = board

            def forward(self, tokens, conditions):
                logits = torch.zeros(1, tokens.shape[1], self.board.vocab_size)
                logits[..., EOS] = 50
                logits[..., self.board.token(1, START)] = 40
                logits[..., self.board.token(0, FOOT)] = 30
                return logits

        given = [(11, START), (10, FOOT)]
        result = generate(Adversary(self.board), self.board, 5, 40, given, generator=torch.Generator().manual_seed(0))
        ids = [h["hold_token"] for h in result["holds"]]
        self.assertEqual(result["holds"][:2], [{"hold_token": 11, "hold_use_type": START, "given": True},
                                               {"hold_token": 10, "hold_use_type": FOOT, "given": True}])
        self.assertEqual(len(ids), len(set(ids)))
        self.assertTrue(result["ended_with_eos"])
        checks, _ = climb_checks(result["holds"], {r["id"]: r["y"] for r in ROWS})
        self.assertTrue(checks["finishes_1_2"] and checks["starts_1_2"] and checks["no_repeated_hold"])
        with self.assertRaises(ValueError):
            validate_given(self.board, [(10, START), (11, START), (12, START)], 10)
        with self.assertRaises(ValueError):
            validate_given(self.board, [(10, START), (10, FOOT)], 10)


class LifecycleTests(unittest.TestCase):
    def test_prepare_train_evaluate_sample_on_tiny_data(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            vocabulary = root / "holds.csv"
            vocabulary.write_text("hold_x,hold_y,on_kickboard,hold_token,set_id\n"
                                  + "".join(f"{r['x']},{r['y']},{r['kickboard']},{r['id']},{r['set']}\n" for r in ROWS))
            rows = []
            # 28 distinct hold sets, plus one set repeated under 20 more UUIDs to exercise leakage removal.
            sets = [[hold(start, START), *[hold(h, HAND) for h in (12, 13, 14) if mask >> (h - 12) & 1],
                     hold(15, FINISH), *([hold(21 - start, FOOT)] if foot else [])]
                    for start in (10, 11) for mask in range(1, 8) for foot in (0, 1)]
            sets += [sets[0]] * 20
            for i, frames in enumerate(sets):
                for angle in (20, 40):
                    rows.append({"climb_uuid": str(i), "climb_angle_id": f"{i}:{angle}", "angle": angle,
                                 "grade_hueco": f"V{i % 8}", "grade_status": "reported",
                                 "ascensionist_count": i, "frames": [frames]})
            rows.append({**rows[0], "climb_uuid": "multi", "frames": [sets[0], sets[1]]})
            source = root / "climbs.jsonl"
            source.write_text("".join(json.dumps(r) + "\n" for r in rows))
            config = Config(output_dir=str(root / "runs"),
                            data=DataConfig(source=str(source), vocabulary=str(vocabulary),
                                            cache_dir=str(root / "cache"), train_fraction=0.5,
                                            validation_fraction=0.25, max_holds=8),
                            model=ModelConfig(d_model=16, n_layers=1, n_heads=2, coordinate_frequencies=2),
                            train=TrainConfig(epochs=2, batch_size=16, threads=1, ascent_weighting="log"))
            directory, metadata = prepare(config.data)
            self.assertEqual(metadata["excluded"], {"multi_frame": 1})
            self.assertGreater(metadata["training_rows_removed_for_leakage"], 0)
            arrays = load_arrays(directory)
            held_out = set(arrays["signatures"][(arrays["splits"] == 1) | (arrays["splits"] == 2)])
            self.assertFalse(held_out & set(arrays["signatures"][arrays["splits"] == 0]))
            self.assertEqual(prepare(config.data)[1]["fingerprint"], metadata["fingerprint"])
            run = train(config)
            self.assertTrue((run / "summary.json").exists())
            model, board, loaded, _ = load_run(run / "best.pt")
            result = evaluate(model, arrays, 1, "cpu")
            for mode in ("scratch", "starts", "starts_feet"):
                self.assertTrue(np.isfinite(result[mode]["nll_per_climb"]))
            self.assertLess(result["starts"]["nll_per_climb"], result["scratch"]["nll_per_climb"])
            sample = generate(model, board, 3, 40, [(10, START)], generator=torch.Generator().manual_seed(1),
                              max_holds=loaded.data.max_holds)
            self.assertEqual(sample["holds"][0], {"hold_token": 10, "hold_use_type": START, "given": True})
            self.assertLessEqual(len(sample["holds"]), 8)


if __name__ == "__main__":
    unittest.main()
