"""Dispatcher behavior tests using a fake engine; no torch, model, or GPU needed."""
import importlib.util
from pathlib import Path
import queue
import sys
import threading
import unittest
from concurrent.futures import CancelledError, TimeoutError


# Import only the serving module: the package's __init__ imports torch eagerly.
_spec = importlib.util.spec_from_file_location(
    "_startlux_server_batching_tests", Path(__file__).resolve().parents[1] / "startlux_decision" / "server.py"
)
_server = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _server
_spec.loader.exec_module(_server)
BatchDispatcher = _server.BatchDispatcher


class FakeEngine:
    def __init__(self):
        self.started = threading.Event()
        self.release = threading.Event()
        self.calls = []
        self.active = 0
        self.max_active = 0
        self.lock = threading.Lock()

    @staticmethod
    def result(state, questions):
        return {"decision": state}, {"input_tokens": questions["tokens"], "output_tokens": 0}

    def _call(self, method, requests):
        with self.lock:
            self.calls.append((method, [state for state, _ in requests]))
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        try:
            if any(state == "blocking" for state, _ in requests):
                self.started.set()
                if not self.release.wait(5):
                    raise RuntimeError("test did not release the blocked engine")
            if any(state == "bad" for state, _ in requests):
                raise ValueError("invalid question")
            if any(state == "failure" for state, _ in requests):
                raise RuntimeError("model failure")
            return [self.result(state, questions) for state, questions in requests]
        finally:
            with self.lock:
                self.active -= 1

    def decide(self, state, questions):
        return self._call("decide", [(state, questions)])[0]

    def decide_batch(self, requests, return_usage=False):
        if not return_usage:
            raise AssertionError("serving must request usage for each request")
        return self._call("decide_batch", requests)


class BatchDispatcherTests(unittest.TestCase):
    def make_dispatcher(self, **kwargs):
        engine = FakeEngine()
        dispatcher = BatchDispatcher(engine, **kwargs)
        # Release before close even when an assertion fails, so cleanup cannot hang.
        self.addCleanup(dispatcher.close)
        self.addCleanup(engine.release.set)
        return engine, dispatcher

    def block_worker(self, engine, dispatcher):
        future = dispatcher.submit("blocking", {"tokens": 1})
        self.assertTrue(engine.started.wait(2), "worker did not start the first request")
        return future

    def test_single_request_uses_decide(self):
        engine, dispatcher = self.make_dispatcher(batch_size=1)
        result = dispatcher.submit("single", {"tokens": 17}).result(timeout=2)
        self.assertEqual(result, engine.result("single", {"tokens": 17}))
        self.assertEqual(engine.calls, [("decide", ["single"])])

    def test_batches_preserve_request_order_and_individual_usage(self):
        engine, dispatcher = self.make_dispatcher(batch_size=8)
        first = self.block_worker(engine, dispatcher)
        states = ["third", "first", "second"]
        futures = [dispatcher.submit(state, {"tokens": tokens}) for state, tokens in zip(states, [3, 41, 9])]
        engine.release.set()
        first.result(timeout=2)
        results = [future.result(timeout=2) for future in futures]
        self.assertEqual(results, [engine.result(state, {"tokens": n}) for state, n in zip(states, [3, 41, 9])])
        self.assertEqual(engine.calls, [("decide", ["blocking"]), ("decide_batch", states)])

    def test_only_one_model_call_runs_at_a_time(self):
        engine, dispatcher = self.make_dispatcher(batch_size=4)
        first = self.block_worker(engine, dispatcher)
        futures = [dispatcher.submit(str(i), {"tokens": i}) for i in range(12)]
        # While one call holds the engine, every other request must remain pending.
        for future in futures:
            with self.assertRaises(TimeoutError):
                future.result(timeout=0.01)
        self.assertEqual(engine.max_active, 1)
        engine.release.set()
        first.result(timeout=2)
        for future in futures:
            future.result(timeout=2)
        self.assertEqual(engine.max_active, 1)
        self.assertTrue(all(len(states) <= 4 for _, states in engine.calls))

    def test_bad_input_is_isolated_from_other_batch_members(self):
        engine, dispatcher = self.make_dispatcher(batch_size=8)
        first = self.block_worker(engine, dispatcher)
        good_before = dispatcher.submit("before", {"tokens": 7})
        bad = dispatcher.submit("bad", {"tokens": 2})
        good_after = dispatcher.submit("after", {"tokens": 19})
        engine.release.set()
        first.result(timeout=2)
        self.assertEqual(good_before.result(timeout=2), engine.result("before", {"tokens": 7}))
        with self.assertRaisesRegex(ValueError, "invalid question"):
            bad.result(timeout=2)
        self.assertEqual(good_after.result(timeout=2), engine.result("after", {"tokens": 19}))
        self.assertIn(("decide_batch", ["before", "bad", "after"]), engine.calls)

    def test_model_failure_fails_batch_and_worker_recovers(self):
        engine, dispatcher = self.make_dispatcher(batch_size=8)
        first = self.block_worker(engine, dispatcher)
        failure = dispatcher.submit("failure", {"tokens": 3})
        affected = dispatcher.submit("affected", {"tokens": 11})
        engine.release.set()
        first.result(timeout=2)
        for future in (failure, affected):
            with self.assertRaisesRegex(RuntimeError, "model failure"):
                future.result(timeout=2)
        recovered = dispatcher.submit("recovered", {"tokens": 13})
        self.assertEqual(recovered.result(timeout=2), engine.result("recovered", {"tokens": 13}))

    def test_queue_capacity_is_bounded(self):
        engine, dispatcher = self.make_dispatcher(batch_size=1, queue_size=1)
        first = self.block_worker(engine, dispatcher)
        waiting = dispatcher.submit("waiting", {"tokens": 4})
        with self.assertRaises(queue.Full):
            dispatcher.submit("overflow", {"tokens": 9})
        engine.release.set()
        first.result(timeout=2)
        waiting.result(timeout=2)
        self.assertNotIn("overflow", [state for _, states in engine.calls for state in states])

    def test_timed_out_pending_request_can_be_cancelled(self):
        engine, dispatcher = self.make_dispatcher(batch_size=8)
        first = self.block_worker(engine, dispatcher)
        expired = dispatcher.submit("expired", {"tokens": 4})
        with self.assertRaises(TimeoutError):
            expired.result(timeout=0.01)
        self.assertTrue(expired.cancel())
        live = dispatcher.submit("live", {"tokens": 8})
        engine.release.set()
        first.result(timeout=2)
        self.assertEqual(live.result(timeout=2), engine.result("live", {"tokens": 8}))
        with self.assertRaises(CancelledError):
            expired.result()
        self.assertNotIn("expired", [state for _, states in engine.calls for state in states])

    def test_close_rejects_new_requests(self):
        _, dispatcher = self.make_dispatcher()
        dispatcher.close()
        with self.assertRaises(RuntimeError):
            dispatcher.submit("late", {"tokens": 1})

    def test_close_cancels_queued_requests_and_waits_for_active_call(self):
        engine, dispatcher = self.make_dispatcher(batch_size=1)
        first = self.block_worker(engine, dispatcher)
        waiting = dispatcher.submit("waiting", {"tokens": 4})
        cancelled = threading.Event()
        waiting.add_done_callback(lambda _: cancelled.set())
        closer = threading.Thread(target=dispatcher.close)
        closer.start()
        try:
            self.assertTrue(cancelled.wait(2), "close did not cancel the queued request")
            self.assertTrue(waiting.cancelled())
            self.assertTrue(closer.is_alive(), "close must wait for active inference")
        finally:
            engine.release.set()
            closer.join(timeout=2)
        self.assertFalse(closer.is_alive())
        self.assertEqual(first.result(timeout=2), engine.result("blocking", {"tokens": 1}))


if __name__ == "__main__":
    unittest.main()
