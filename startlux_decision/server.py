"""TypeSafe-compatible decision server.

    python -m startlux_decision.server --model /path/to/StartLux-Decision-4B --port 8090

POST /v1/systemone  {"state": ..., "questions": {key: {"type", "instructions", "criteria"}}}
  -> {"answers": {key: answer}, "usage": {"input_tokens", "output_tokens"}, "model": ...}
GET  /health        {"status", "model", "backend", "fast_kernels", "cuda_graphs"}
GET  /v1/models
Concurrent requests are queued and batched by one inference worker.  At start-up one request runs through both the CUDA-graph path and the
eager path; if their probabilities differ by more than 0.02 the graphs are dropped.

--backend auto (the default) runs the model with MLX on Apple Silicon when mlx-lm is installed (mlx_model.py) and with
PyTorch everywhere else; --int8 adds int8 matmuls on M5 and later Macs (mlx_int8.py).
"""
import argparse
import json
import math
import os
import queue
import threading
import time
from concurrent.futures import CancelledError, Future, TimeoutError
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class BatchDispatcher:
    """One model worker, a bounded waiting queue, and a fixed batch collection window."""

    def __init__(self, engine, batch_size=8, batch_wait_ms=5, queue_size=64):
        if batch_size < 1 or queue_size < 1 or not math.isfinite(batch_wait_ms) or batch_wait_ms < 0:
            raise ValueError("batch/queue sizes must be positive and batch_wait_ms finite and non-negative")
        self.engine, self.batch_size, self.batch_wait_ms = engine, batch_size, batch_wait_ms
        self.queue = queue.Queue(maxsize=queue_size)
        self.max_observed_batch_size = 0
        self._closed, self._admission = threading.Event(), threading.Lock()
        self._worker = threading.Thread(target=self._run, name="startlux-inference", daemon=True)
        self._worker.start()

    def submit(self, state, questions):
        future = Future()
        with self._admission:
            if self._closed.is_set():
                raise RuntimeError("inference worker is closed")
            self.queue.put_nowait((future, state, questions))
        return future

    def close(self):
        with self._admission:
            self._closed.set()
        while True:
            try:
                self.queue.get_nowait()[0].cancel()
            except queue.Empty:
                break
        self._worker.join()

    def _decide(self, item):
        future, state, questions = item
        try:
            result = self.engine.decide(state, questions)
        except Exception as error:
            future.set_exception(error)
        else:
            future.set_result(result)

    def _run(self):
        try:
            while not self._closed.is_set():
                try:
                    batch = [self.queue.get(timeout=0.1)]
                except queue.Empty:
                    continue
                deadline = time.monotonic() + self.batch_wait_ms / 1000
                while len(batch) < self.batch_size:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        break
                    try:
                        batch.append(self.queue.get(timeout=remaining))
                    except queue.Empty:
                        break
                if self._closed.is_set():
                    for future, _, _ in batch:
                        future.cancel()
                    break
                active = [item for item in batch if item[0].set_running_or_notify_cancel()]
                if not active:
                    continue
                self.max_observed_batch_size = max(self.max_observed_batch_size, len(active))
                if len(active) == 1:
                    self._decide(active[0])
                    continue
                try:
                    results = self.engine.decide_batch([(s, q) for _, s, q in active], return_usage=True)
                    if len(results) != len(active):
                        raise RuntimeError("model returned an unexpected number of batch results")
                except ValueError:  # isolate invalid inputs without failing the valid requests beside them
                    for item in active:
                        self._decide(item)
                except Exception as error:
                    for future, _, _ in active:
                        future.set_exception(error)
                else:
                    for (future, _, _), result in zip(active, results):
                        future.set_result(result)
        finally:
            while True:
                try:
                    self.queue.get_nowait()[0].cancel()
                except queue.Empty:
                    break


def mlx_available():
    """True on Apple Silicon with mlx-lm installed (requirements.txt installs it there)."""
    try:
        import mlx.core as mx
        import mlx_lm  # noqa: F401
        return mx.metal.is_available()
    except ImportError:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="local StartLux-Decision directory")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8090)
    ap.add_argument("--device", help="cuda or cpu for the torch backend (default: cuda when available)")
    ap.add_argument("--backend", choices=("auto", "torch", "mlx"), default="auto",
                    help="auto: MLX on Apple Silicon when mlx-lm is installed, else torch")
    ap.add_argument("--int8", action="store_true", help="MLX on M5 and later: int8 matmuls on the neural accelerators")
    ap.add_argument("--name", help="model name reported in responses (default: the directory name)")
    ap.add_argument("--batch-size", type=int, default=8, help="maximum torch HTTP requests per inference batch; 1 disables batching")
    ap.add_argument("--batch-wait-ms", type=float, default=5, help="fixed collection window after the first request")
    ap.add_argument("--queue-size", type=int, default=64, help="maximum requests waiting for inference")
    ap.add_argument("--request-timeout", type=float, default=120, help="queue plus inference timeout in seconds")
    ap.add_argument("--max-length", type=int, default=65536, help="maximum tokens per question prompt")
    ap.add_argument("--max-batch-tokens", type=int, default=8192, help="torch padded-token budget per forward pass")
    a = ap.parse_args()
    if (a.batch_size < 1 or a.queue_size < 1 or a.max_length < 1 or a.max_batch_tokens < 1 or
            not math.isfinite(a.batch_wait_ms) or a.batch_wait_ms < 0 or
            not math.isfinite(a.request_timeout) or a.request_timeout <= 0):
        ap.error("sizes/limits must be positive, batch-wait-ms non-negative, and timeouts finite")
    backend = a.backend
    if backend == "auto":
        backend = "mlx" if a.device is None and mlx_available() else "torch"
    if backend == "mlx":
        from .mlx_model import MLXDecision
        engine = MLXDecision(a.model, int8=a.int8, max_length=a.max_length)
        engine.warm_up()
    else:
        from .model import StartLuxDecision
        engine = StartLuxDecision(a.model, device=a.device, max_length=a.max_length, max_batch_tokens=a.max_batch_tokens)
    demo_state = {"ticket": "I was charged twice for order #4411 and the app still shows it as unpaid."}
    demo_questions = {
        "team": {"type": "choice", "instructions": "Which team should handle this ticket?",
                 "criteria": {"billing": "Payments, refunds and invoices", "shipping": "Delivery and tracking",
                              "technical": "App, login and account problems"}},
        "urgent": {"type": "noul", "instructions": "Should this ticket be answered today?"},
        "severity": {"type": "score", "instructions": "How severe is the impact?",
                     "criteria": ["cosmetic", "annoying", "blocks the customer"]},
    }
    engine.decide(demo_state, demo_questions)                     # warm-up
    if engine.graphs:
        diff = engine.self_test(demo_state, demo_questions)
        print(f"graph self-test: max |p_graph - p_eager| = {diff:.2e}", flush=True)
        if diff > 0.02:
            print("graph readout disagrees with the eager path, graphs disabled", flush=True)
            engine.graphs = {}
    # MLX's shared-prefix path does not bound cross-request batches; keep its existing singleton HTTP path.
    batch_size = a.batch_size if backend == "torch" else 1
    batch_wait_ms = a.batch_wait_ms if batch_size > 1 else 0
    dispatcher = BatchDispatcher(engine, batch_size, batch_wait_ms, a.queue_size)
    name = a.name or os.path.basename(os.path.normpath(a.model))

    class Handler(BaseHTTPRequestHandler):
        def _send(self, code, obj):
            data = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            try:
                self.wfile.write(data)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def do_GET(self):
            path = self.path.rstrip("/")
            if path in ("/health", "/v1/health"):
                return self._send(200, {"status": "ok", "model": name, "backend": backend, "fast_kernels": engine.fast_kernels,
                                        "cuda_graphs": len(engine.graphs), "batching": {
                                            "max_requests": batch_size, "wait_ms": batch_wait_ms,
                                            "queue_capacity": a.queue_size, "queued_requests": dispatcher.queue.qsize(),
                                            "max_observed_requests": dispatcher.max_observed_batch_size}})
            if path == "/v1/models":
                return self._send(200, {"models": [{"name": name, "description": "StartLux-Decision typed decision model"}]})
            self._send(404, {"error": "not found"})

        def do_POST(self):
            if self.path.rstrip("/") != "/v1/systemone":
                return self._send(404, {"error": "not found"})
            try:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
                if not isinstance(body, dict):
                    raise ValueError("request body must be an object")
                questions = body.get("questions")
                if not isinstance(questions, dict) or not questions:
                    raise ValueError("questions must be a non-empty object")
                if any(not isinstance(q, dict) for q in questions.values()):
                    raise ValueError("each question must be an object")
            except ValueError as e:
                return self._send(400, {"error": str(e)})
            t = time.perf_counter()
            try:
                future = dispatcher.submit(body.get("state"), questions)
            except queue.Full:
                return self._send(503, {"error": "inference queue is full; retry later"})
            except RuntimeError as e:
                return self._send(503, {"error": str(e)})
            try:
                answers, usage = future.result(timeout=a.request_timeout)
            except TimeoutError:
                future.cancel()  # pending work is skipped; an in-progress GPU call finishes normally
                return self._send(504, {"error": "inference request timed out"})
            except CancelledError:
                return self._send(503, {"error": "inference worker is closing"})
            except ValueError as e:          # unknown question type, prompt over the context limit
                return self._send(422, {"error": str(e)})
            except Exception as e:
                print(f"inference failed: {e!r}", flush=True)
                return self._send(500, {"error": "inference failed"})
            self._send(200, {"answers": answers, "usage": usage, "model": name,
                             "latency_ms": round(1000 * (time.perf_counter() - t), 2)})

        def log_message(self, *args):
            pass

    detail = (f"MLX, int8 projections: {engine.int8}" if backend == "mlx" else
              f"fast kernels: {engine.fast_kernels}, cuda graphs: {len(engine.graphs)}")
    print(f"{name} serving on http://{a.host}:{a.port}/v1/systemone ({detail}; batch <= {batch_size}, "
          f"wait {batch_wait_ms:g} ms)", flush=True)
    try:
        with ThreadingHTTPServer((a.host, a.port), Handler) as server:
            server.daemon_threads = True
            server.serve_forever()
    finally:
        dispatcher.close()


if __name__ == "__main__":
    main()
