"""StartLux-Decision on Apple Silicon: the forward pass in MLX, everything else unchanged.

    python -m startlux_decision.server --model StartLux-Decision-4B              # --backend auto picks MLX on a Mac
    python -m startlux_decision.server --model StartLux-Decision-4B --int8       # M5 and later: int8 matmuls

MLXDecision is StartLuxDecision with the forward pass run by mlx-lm's Qwen3.5 implementation, which has Metal kernels
for the linear-attention layers (PyTorch on MPS runs reference code there).  As in GGUFDecision, prompt rendering, the
option-letter readout, the per-type temperatures and wide choices are the package's own code.

All questions of a request run in one forward pass, one right-padded row per question.  Every prompt starts with the
same system text: it runs once at start-up and its cache (attention keys and values, linear-attention conv and recurrent
states) is copied to every row.  When the questions of a request share a long piece of evidence, the evidence runs once
too and is kept, so asking about the same evidence again runs only the questions.  int8=True moves the large
projections to int8 matmuls on the GPU's neural accelerators (mlx_int8.py).
"""
import json
import os

import mlx.core as mx
import numpy as np
import torch

from . import jevfmt as J
from .model import StartLuxDecision

CACHE_LIMIT = 1 << 30       # MLX keeps freed buffers for reuse; unbounded, the cache grows with every new request shape
SHARE_MIN = 192             # evidence tokens a shared prefix saves, above which running it separately pays off
MAX_BATCH_TOKENS = 16384    # padded tokens per forward pass when many rows arrive at once (decide_batch)


class MLXDecision(StartLuxDecision):
    backend = "mlx"

    def __init__(self, path, int8=False, max_length=65536):            # no torch model: MLX runs the weights
        from mlx_lm import load
        from transformers import AutoTokenizer

        config = json.load(open(os.path.join(path, "config.json")))
        if config.get("text_config", config).get("num_experts"):
            raise NotImplementedError("mixture-of-experts checkpoints such as StartLux-Decision-35B-A3B run on the torch "
                                      "backend: start the server with --backend torch")
        cfg = json.load(open(os.path.join(path, "decision_config.json")))
        self.tok = AutoTokenizer.from_pretrained(path)
        self.letters = J.check_tokenizer(self.tok)
        if self.letters != cfg["letter_token_ids"]:
            raise ValueError("tokenizer letter ids differ from decision_config.json")
        self.temperature = {k: float(v) for k, v in cfg["temperature_by_type"].items()}
        wide = cfg.get("wide_choice", {})
        self.group, self.keep, self.residual = wide.get("group", 25), wide.get("keep", 3), wide.get("residual", 1e-3)
        self.max_length, self.graphs, self.fast_kernels = int(max_length), {}, True
        self.pad = self.tok.pad_token_id if self.tok.pad_token_id is not None else 0
        mx.set_cache_limit(CACHE_LIMIT)
        self.model, _ = load(path, lazy=int8)
        lm = getattr(self.model, "language_model", self.model)
        self.body = lm.model
        self.int8 = 0
        if int8:
            from . import mlx_int8
            if not mlx_int8.supported():
                raise RuntimeError("int8 needs the neural accelerators of an M5 or later GPU; run without --int8")
            rows = [r for s, q in mlx_int8.CALIBRATION for _, r in self._split(s, q)[1]]
            self.int8 = mlx_int8.convert(self.body, [t for t, _ in self._encode(rows)])
        mx.eval(self.model.parameters())
        head = lm.lm_head if hasattr(lm, "lm_head") else self.body.embed_tokens
        idx = mx.array(self.letters)
        w = head.weight[idx]
        if hasattr(head, "scales"):                        # a quantized build: dequantize the letter rows only
            w = mx.dequantize(w, head.scales[idx], head.biases[idx], head.group_size, head.bits)
        self.letter_rows = w.astype(mx.float32)
        self.snapshots = {}                                            # token prefix -> prompt cache right after it
        probes = (({"x": "a"}, {"q": {"type": "noul", "instructions": "?"}}),
                  ({"y": 1}, {"q": {"type": "noul", "instructions": "!"}}))
        a, b = (self._encode([r for _, r in self._split(s, q)[1]])[0][0] for s, q in probes)
        n = 0
        while n < min(len(a), len(b)) and a[n] == b[n]:
            n += 1
        self.preamble = tuple(a[:n])                                   # the system text every prompt starts with
        self._prefill(self.preamble)

    def warm_up(self):
        """Run a few request shapes once, so Metal compiles their kernels before the first real request."""
        page = lambda n: {"page": " ".join(f"Row {i}: item {i}, qty {i % 7}." for i in range(n))}
        yes_no = {"q": {"type": "noul", "instructions": "Is the item in stock?"}}
        three = dict(yes_no,
                     c={"type": "choice", "instructions": "Which item?", "criteria": {"a": "first", "b": "second"}},
                     s={"type": "score", "instructions": "How full is the page?", "criteria": ["empty", "some", "full"]})
        for n in (4, 40, 160):
            self.decide(page(n), yes_no)
            self.decide(page(n), three)

    def _encode(self, rows):
        out = []
        for r in rows:
            order = [o["id"] for o in r["options"]]
            ids, _ = J.render_ids(r, self.tok, order, max_length=self.max_length)
            out.append((ids, len(order)))
        return out

    def _forward(self, ids, last, cache=None):
        h = self.body(ids, cache=cache)
        return h[mx.arange(ids.shape[0]), last].astype(mx.float32) @ self.letter_rows.T

    @staticmethod
    def _copy(cache, rows):
        """`cache` repeated to `rows` rows; the stored cache itself is never written to."""
        from mlx_lm.models.cache import ArraysCache, KVCache
        out = []
        for c in cache:
            if isinstance(c, KVCache):
                n = KVCache()
                if c.keys is not None:
                    n.keys = mx.repeat(c.keys[..., :c.offset, :], rows, axis=0)
                    n.values = mx.repeat(c.values[..., :c.offset, :], rows, axis=0)
                n.offset = c.offset
            else:
                n = ArraysCache(len(c.cache))
                n.cache = [None if a is None else mx.repeat(a, rows, axis=0) for a in c.cache]
            out.append(n)
        return out

    def _prefill(self, prefix):
        """The prompt cache right after `prefix`, resumed from the longest stored prefix it starts with, and stored."""
        from mlx_lm.models.cache import make_prompt_cache
        best = max((s for s in self.snapshots if prefix[:len(s)] == s), key=len, default=())
        cache = self._copy(self.snapshots[best], 1) if best else make_prompt_cache(self.model)
        if len(prefix) > len(best):
            self.body(mx.array([prefix[len(best):]]), cache=cache)
        mx.eval([c.state for c in cache])
        self.snapshots[prefix] = cache
        return cache

    def _logits(self, rows):
        enc = self._encode(rows)
        toks = [t for t, _ in enc]
        c = 0                                                          # common prefix, leaving every row one token
        while c < min(len(t) for t in toks) - 1 and all(t[c] == toks[0][c] for t in toks):
            c += 1
        prefix = tuple(toks[0][:c])
        pre = list(self.preamble)
        if prefix != self.preamble and prefix in self.snapshots:       # the same evidence as before: questions only
            groups = [(list(range(len(toks))), c, self.snapshots[prefix])]
        elif (len(toks) - 1) * (c - len(pre)) >= SHARE_MIN:            # long shared evidence: run it once, keep it
            for s in [s for s in self.snapshots if s != self.preamble]:
                del self.snapshots[s]
            groups = [(list(range(len(toks))), c, self._prefill(prefix))]
        else:                                                          # one pass from the stored system text
            start = len(pre) if all(t[:len(pre)] == pre for t in toks) else 0
            base = self.snapshots[self.preamble] if start else None
            groups, batch = [], []
            for j in sorted(range(len(toks)), key=lambda j: -len(toks[j])):
                if batch and (len(batch) + 1) * (len(toks[batch[0]]) - start) > MAX_BATCH_TOKENS:
                    groups.append((batch, start, base))
                    batch = []
                batch.append(j)
            groups.append((batch, start, base))
        out = [None] * len(toks)
        for idx, start, cache in groups:
            n = max(len(toks[j]) - start for j in idx)
            ids = np.full((len(idx), n), self.pad, dtype=np.int32)
            for r, j in enumerate(idx):
                ids[r, :len(toks[j]) - start] = toks[j][start:]
            z = np.array(self._forward(mx.array(ids), mx.array([len(toks[j]) - start - 1 for j in idx]),
                                       self._copy(cache, len(idx)) if cache is not None else None))
            for r, j in enumerate(idx):
                out[j] = torch.from_numpy(z[r, :enc[j][1]].copy())
        return out, sum(len(t) for t in toks)
