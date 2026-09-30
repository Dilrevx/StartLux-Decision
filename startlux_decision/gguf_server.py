"""A /v1/systemone server for a StartLux-Decision GGUF file run by llama.cpp.

    llama-server -m StartLux-Decision-4B-Q8_0.gguf -ngl 99 -c 16384 --parallel 4 --port 8081
    python -m startlux_decision.gguf_server --model-dir StartLux-Decision-4B-GGUF --llama http://127.0.0.1:8081 --port 8090

It is StartLuxDecision with only the forward pass replaced by a llama-server call: prompt rendering, the option-letter
readout, the per-type temperatures and wide choices are the package's own code, so the answers match the original
model question by question up to the numerics of the GGUF file.  MODEL_DIR holds the tokenizer files, config.json and
decision_config.json (the GGUF repositories ship them next to the .gguf files).  llama-server returns the
log-probabilities of the whole vocabulary at the answer position; dividing them by the temperature and normalising over
the listed options is the same as doing it on the logits."""
import argparse
import json
import os
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import torch

from . import jevfmt as J
from .model import StartLuxDecision


class GGUFDecision(StartLuxDecision):
    def __init__(self, path, llama, workers):                     # no torch model: llama-server runs the weights
        from transformers import AutoTokenizer
        cfg = json.load(open(os.path.join(path, "decision_config.json")))
        self.tok = AutoTokenizer.from_pretrained(path)
        self.letters = J.check_tokenizer(self.tok)
        if self.letters != cfg["letter_token_ids"]:
            raise ValueError("tokenizer letter ids differ from decision_config.json")
        self.temperature = {k: float(v) for k, v in cfg["temperature_by_type"].items()}
        wide = cfg.get("wide_choice", {})
        self.group, self.keep, self.residual = wide.get("group", 25), wide.get("keep", 3), wide.get("residual", 1e-3)
        self.max_length, self.graphs = 65536, {}
        mc = json.load(open(os.path.join(path, "config.json")))
        self.vocab = int(mc.get("text_config", mc).get("vocab_size", 262144))
        self.url = llama.rstrip("/") + "/completion"
        self.pool = ThreadPoolExecutor(workers)

    def _letter_logprobs(self, ids, count):
        """Next-token log-probabilities of the option letters at the last prompt position (full-vocabulary softmax;
        dividing by the temperature and normalising over the options afterwards equals doing it on the logits)."""
        targets = self.letters[:count]
        for n in (256, self.vocab):
            body = json.dumps({"prompt": ids, "n_predict": 1, "temperature": -1, "n_probs": n, "cache_prompt": False}).encode()
            with urllib.request.urlopen(urllib.request.Request(self.url, body, {"Content-Type": "application/json"}), timeout=900) as r:
                out = json.loads(r.read())
            rows = out.get("completion_probabilities", out.get("probs"))
            got = {t["id"]: t["logprob"] for t in rows[0]["top_logprobs"]}
            if all(t in got for t in targets):
                return [got[t] for t in targets]
        raise RuntimeError("option letters missing from llama-server's log-probabilities")

    def _logits(self, rows):
        jobs = []
        for r in rows:
            order = [o["id"] for o in r["options"]]
            ids, _ = J.render_ids(r, self.tok, order, max_length=self.max_length)
            jobs.append((ids, len(order)))
        outs = list(self.pool.map(lambda j: self._letter_logprobs(*j), jobs))
        return [torch.tensor(o, dtype=torch.float32) for o in outs], sum(len(i) for i, _ in jobs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--llama", required=True)
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--name", help="model name reported in responses (default: the GGUF directory name)")
    a = ap.parse_args()
    engine = GGUFDecision(a.model_dir, a.llama, a.workers)
    a.name = a.name or os.path.basename(os.path.abspath(a.model_dir))

    class Handler(BaseHTTPRequestHandler):
        def _send(self, code, obj):
            data = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            self._send(200, {"status": "ok", "model": a.name})

        def do_POST(self):
            try:
                req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
                answers, usage = engine.decide(req["state"], req["questions"])
                self._send(200, {"answers": answers, "usage": usage, "model": a.name})
            except Exception as e:                                 # surfaced to the client; suites.py stops on it
                self._send(500, {"error": repr(e)})

        def log_message(self, *args):
            pass

    class Server(ThreadingHTTPServer):
        request_queue_size = 1024                           # the default backlog of 5 resets connections under load
        daemon_threads = True

    Server(("127.0.0.1", a.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
