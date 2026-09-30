"""TypeSafe-compatible decision server.

    python -m startlux_decision.server --model /path/to/StartLux-Decision-4B --port 8090

POST /v1/systemone  {"state": ..., "questions": {key: {"type", "instructions", "criteria"}}}
  -> {"answers": {key: answer}, "usage": {"input_tokens", "output_tokens"}, "model": ...}
GET  /health        {"status", "model", "backend", "fast_kernels", "cuda_graphs"}
GET  /v1/models
Requests are served one at a time on one GPU.  At start-up one request runs through both the CUDA-graph path and the
eager path; if their probabilities differ by more than 0.02 the graphs are dropped.

--backend auto (the default) runs the model with MLX on Apple Silicon when mlx-lm is installed (mlx_model.py) and with
PyTorch everywhere else; --int8 adds int8 matmuls on M5 and later Macs (mlx_int8.py).
"""
import argparse
import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


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
    a = ap.parse_args()
    backend = a.backend
    if backend == "auto":
        backend = "mlx" if a.device is None and mlx_available() else "torch"
    if backend == "mlx":
        from .mlx_model import MLXDecision
        engine = MLXDecision(a.model, int8=a.int8)
        engine.warm_up()
    else:
        from .model import StartLuxDecision
        engine = StartLuxDecision(a.model, device=a.device)
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
    lock = threading.Lock()
    name = a.name or os.path.basename(os.path.normpath(a.model))

    class Handler(BaseHTTPRequestHandler):
        def _send(self, code, obj):
            data = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            path = self.path.rstrip("/")
            if path in ("/health", "/v1/health"):
                return self._send(200, {"status": "ok", "model": name, "backend": backend, "fast_kernels": engine.fast_kernels,
                                        "cuda_graphs": len(engine.graphs)})
            if path == "/v1/models":
                return self._send(200, {"models": [{"name": name, "description": "StartLux-Decision typed decision model"}]})
            self._send(404, {"error": "not found"})

        def do_POST(self):
            if self.path.rstrip("/") != "/v1/systemone":
                return self._send(404, {"error": "not found"})
            try:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
                questions = body.get("questions")
                if not isinstance(questions, dict) or not questions:
                    raise ValueError("questions must be a non-empty object")
            except ValueError as e:
                return self._send(400, {"error": str(e)})
            t = time.perf_counter()
            try:
                with lock:
                    answers, usage = engine.decide(body.get("state"), questions)
            except ValueError as e:          # unknown question type, prompt over the context limit
                return self._send(422, {"error": str(e)})
            self._send(200, {"answers": answers, "usage": usage, "model": name,
                             "latency_ms": round(1000 * (time.perf_counter() - t), 2)})

        def log_message(self, *args):
            pass

    detail = (f"MLX, int8 projections: {engine.int8}" if backend == "mlx" else
              f"fast kernels: {engine.fast_kernels}, cuda graphs: {len(engine.graphs)}")
    print(f"{name} serving on http://{a.host}:{a.port}/v1/systemone ({detail})", flush=True)
    ThreadingHTTPServer((a.host, a.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
