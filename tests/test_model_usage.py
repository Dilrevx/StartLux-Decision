"""Batch usage accounting without loading checkpoint weights or requiring a GPU."""
import unittest
from unittest.mock import Mock

try:
    import torch
except ImportError:
    torch = None

if torch is not None:
    from startlux_decision import jevfmt as J
    from startlux_decision.model import StartLuxDecision

    class FakeTokenizer:
        def __init__(self):
            self.encode_calls = 0

        def apply_chat_template(self, messages, **kwargs):
            return "\n".join(f'{m["role"]}: {m["content"]}' for m in messages) + "\nassistant:"

        def encode(self, text, **kwargs):
            self.encode_calls += 1
            return list(range(len(text)))

    class StubDecision(StartLuxDecision):
        def __init__(self):
            self.tok = FakeTokenizer()
            self.max_length = 65536
            self.temperature = {"choice": 1.0, "noul": 1.0, "score": 1.0}
            self.group, self.keep, self.residual = 25, 3, 0.001
            self.graphs = {(1, 128): object()}
            self.batches = []

        def _logits(self, rows):
            self.batches.append(list(rows))
            tokens = sum(len(J.render_ids(r, self.tok, max_length=self.max_length)[0]) for r in rows)
            return [torch.zeros(len(r["options"])) for r in rows], tokens


@unittest.skipIf(torch is None, "torch is not installed")
class BatchUsageTests(unittest.TestCase):
    def setUp(self):
        self.requests = [
            ("short evidence", {
                "normal": {"type": "noul", "instructions": "Is action needed?"},
                "single": {"type": "choice", "criteria": {"only": "The only option"}},
            }),
            ({"ticket": "longer evidence " * 20}, {
                "normal": {"type": "score", "instructions": "How severe?", "criteria": ["low", "high"]},
                "wide": {"type": "choice", "instructions": "Pick an item.",
                         "criteria": {f"item{i}": f"Description {i}" for i in range(27)}},
            }),
            ("unused evidence", {"single": {"type": "choice", "criteria": {"only": "Only"}}}),
        ]

    def test_usage_matches_individual_requests_including_wide_rounds(self):
        baseline = StubDecision()
        expected = [baseline.decide(state, questions) for state, questions in self.requests]
        engine = StubDecision()
        original_graphs = engine.graphs

        actual = engine.decide_batch(self.requests, return_usage=True)

        self.assertEqual(actual, expected)
        self.assertEqual(actual[2][1], {"input_tokens": 0, "output_tokens": 0})
        self.assertGreater(actual[1][1]["input_tokens"], actual[0][1]["input_tokens"])
        self.assertIs(engine.graphs, original_graphs)
        self.assertEqual([len(batch) for batch in engine.batches], [2, 1, 2])

    def test_default_returns_answers_without_extra_tokenization(self):
        baseline = StubDecision()
        expected = [baseline.decide(state, questions)[0] for state, questions in self.requests]
        engine = StubDecision()

        actual = engine.decide_batch(self.requests)

        self.assertEqual(actual, expected)
        self.assertTrue(all(isinstance(answer, dict) for answer in actual))
        self.assertEqual(engine.tok.encode_calls, sum(len(batch) for batch in engine.batches))

    def test_graphs_are_restored_after_forward_failure(self):
        for return_usage in (False, True):
            with self.subTest(return_usage=return_usage):
                engine = StubDecision()
                original_graphs = engine.graphs
                engine._logits = Mock(side_effect=RuntimeError("forward failed"))

                with self.assertRaisesRegex(RuntimeError, "forward failed"):
                    engine.decide_batch([self.requests[0]], return_usage=return_usage)

                self.assertIs(engine.graphs, original_graphs)

    def test_usage_render_failure_keeps_graphs_and_skips_forward(self):
        engine = StubDecision()
        original_graphs = engine.graphs
        engine.max_length = 1

        with self.assertRaisesRegex(ValueError, "length"):
            engine.decide_batch([self.requests[0]], return_usage=True)

        self.assertIs(engine.graphs, original_graphs)
        self.assertEqual(engine.batches, [])

    def test_empty_batch(self):
        engine = StubDecision()
        original_graphs = engine.graphs
        self.assertEqual(engine.decide_batch([]), [])
        self.assertEqual(engine.decide_batch([], return_usage=True), [])
        self.assertIs(engine.graphs, original_graphs)
        self.assertEqual(engine.batches, [])


if __name__ == "__main__":
    unittest.main()
