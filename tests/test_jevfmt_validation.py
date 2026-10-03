"""Real question parsing and batch isolation without importing torch or checkpoint weights."""
import importlib.util
from pathlib import Path
import threading
import unittest


def load_module(name, filename):
    path = Path(__file__).resolve().parents[1] / "startlux_decision" / filename
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


J = load_module("_startlux_jevfmt_validation_tests", "jevfmt.py")
BatchDispatcher = load_module("_startlux_real_parser_batch_tests", "server.py").BatchDispatcher


class QuestionValidationTests(unittest.TestCase):
    def test_choice_accepts_object_list_tuple_and_options_alias(self):
        cases = [
            ({"criteria": {"first": "One", "second": "Two"}}, [
                {"id": "first", "criterion": "One"}, {"id": "second", "criterion": "Two"},
            ]),
            ({"criteria": ["first", "second"]}, [
                {"id": "first", "criterion": None}, {"id": "second", "criterion": None},
            ]),
            ({"criteria": ("first", "second")}, [
                {"id": "first", "criterion": None}, {"id": "second", "criterion": None},
            ]),
            ({"options": {"first": "One", "second": "Two"}}, [
                {"id": "first", "criterion": "One"}, {"id": "second", "criterion": "Two"},
            ]),
        ]
        for fields, expected in cases:
            with self.subTest(fields=fields):
                row = J.from_systemone("evidence", dict(type="choice", instructions="Choose", **fields))
                self.assertEqual(row["options"], expected)
                self.assertEqual(J.validate(row), [option["id"] for option in expected])

    def test_score_accepts_list_tuple_alias_and_sorts_numeric_legend(self):
        cases = [
            {"criteria": ["Low", "High"]},
            {"criteria": ("Low", "High")},
            {"options": ["Low", "High"]},
            {"criteria": {"10": "High", "2": "Low"}},
        ]
        expected = [{"id": "0", "criterion": "Low"}, {"id": "1", "criterion": "High"}]
        for fields in cases:
            with self.subTest(fields=fields):
                row = J.from_systemone("evidence", dict(type="score", instructions="Rate", **fields))
                self.assertEqual(row["options"], expected)
                self.assertTrue(row["ordered"])
                self.assertEqual(J.validate(row), ["0", "1"])

    def test_noul_and_bool_accept_omitted_criteria(self):
        for question_type in ("noul", "bool"):
            with self.subTest(question_type=question_type):
                row = J.from_systemone("evidence", {"type": question_type})
                self.assertEqual(row["type"], "noul")
                self.assertEqual(row["instructions"], "Which answer fits the evidence?")
                self.assertEqual(J.validate(row), ["true", "false"])

    def test_invalid_choice_and_score_criteria_raise_value_error(self):
        for question_type in ("choice", "score"):
            for invalid in (None, 7, True, "first", 1.5):
                with self.subTest(question_type=question_type, invalid=invalid):
                    with self.assertRaisesRegex(ValueError, f"{question_type} criteria"):
                        J.from_systemone("evidence", {
                            "type": question_type, "instructions": "Evaluate", "criteria": invalid,
                        })
            with self.subTest(question_type=question_type, missing=True):
                with self.assertRaisesRegex(ValueError, f"{question_type} criteria"):
                    J.from_systemone("evidence", {"type": question_type, "instructions": "Evaluate"})


class ParsingEngine:
    """Use the real parser in both paths; block one call to build a deterministic waiting batch."""

    def __init__(self):
        self.started = threading.Event()
        self.release = threading.Event()
        self.batch_states = []

    def decide(self, state, questions):
        if state == "blocking":
            self.started.set()
            if not self.release.wait(5):
                raise RuntimeError("test did not release the first call")
        answers = {}
        for key, question in questions.items():
            row = J.from_systemone(state, question, key)
            J.validate(row)
            answers[key] = {"type": row["type"], "state": state}
        return answers, {"input_tokens": len(str(state)), "output_tokens": 0}

    def decide_batch(self, requests, return_usage=False):
        if not return_usage:
            raise AssertionError("dispatcher must request individual usage")
        self.batch_states.append([state for state, _ in requests])
        return [self.decide(state, questions) for state, questions in requests]


class RealParserBatchIsolationTests(unittest.TestCase):
    def test_bad_choice_and_score_do_not_fail_good_batch_members(self):
        engine = ParsingEngine()
        dispatcher = BatchDispatcher(engine, batch_size=8, batch_wait_ms=20)
        self.addCleanup(dispatcher.close)
        self.addCleanup(engine.release.set)
        good_questions = {"q": {"type": "noul", "instructions": "Is action needed?"}}
        first = dispatcher.submit("blocking", good_questions)
        self.assertTrue(engine.started.wait(2))
        good_before = dispatcher.submit("before", good_questions)
        bad_choice = dispatcher.submit("bad choice", {"q": {"type": "choice", "instructions": "Choose"}})
        bad_score = dispatcher.submit("bad score", {"q": {"type": "score", "instructions": "Rate", "criteria": 7}})
        good_after = dispatcher.submit("after", good_questions)
        engine.release.set()
        first.result(timeout=2)

        for state, future in (("before", good_before), ("after", good_after)):
            with self.subTest(state=state):
                self.assertEqual(future.result(timeout=2), engine.decide(state, good_questions))
        for question_type, future in (("choice", bad_choice), ("score", bad_score)):
            with self.subTest(question_type=question_type):
                with self.assertRaisesRegex(ValueError, f"{question_type} criteria"):
                    future.result(timeout=2)
        self.assertIn(["before", "bad choice", "bad score", "after"], engine.batch_states)
        self.assertEqual(dispatcher.submit("recovered", good_questions).result(timeout=2),
                         engine.decide("recovered", good_questions))


if __name__ == "__main__":
    unittest.main()
