"""StartLux-Decision inference.

Each question is rendered as one prompt (startlux_decision/jevfmt.py), the model runs one forward pass, and the
answer is read from the next-token logits of the option letters at the last prompt position, divided by the
temperature of the question type and normalised over the listed options.  Nothing is generated.

All questions of one request run in one forward pass, one row per question.  On CUDA the pass is a recorded graph:
one per padded input length (128 ... 4096 tokens) for a single question, and one per (question count, length) for two
to four questions of up to 1024 tokens each.  Replaying a graph removes the per-layer launch overhead (the approach of
the JevK5 runtime, Apache-2.0).  Inputs are right-padded; every layer is causal and rows never mix, so padding after the
last prompt token never reaches the position that is read.  Other requests run eagerly as one padded batch (inputs
padded to one of PAD_LENGTHS); decide_batch() batches questions across many requests.
"""
import importlib
import json
import os
import warnings

import torch

from . import jevfmt as J

__all__ = ["StartLuxDecision", "load_model", "fast_kernels_active"]

GRAPH_LENGTHS = (128, 192, 256, 320, 384, 512, 640, 768, 1024, 1536, 2048, 3072, 4096)
GRAPH_ROWS = (1, 2, 3, 4)          # questions per request replayed as one graph
MULTI_ROW_MAX_LENGTH = 1024        # longest prompt for the multi-question graphs (rows x length <= 4096 tokens)
# Batched inputs are padded to one of these lengths.  The linear-attention kernels are compiled once per sequence length
# (about a second each), so padding to exact lengths would recompile for almost every batch.  Above 256 tokens a step
# adds at most 25% padding.
PAD_LENGTHS = (128, 192, 256, 320, 384, 448, 512, 640, 768, 896, 1024, 1280, 1536, 1792, 2048, 2560, 3072, 3584, 4096,
               5120, 6144, 7168, 8192, 10240, 12288, 14336, 16384, 20480, 24576, 28672, 32768, 40960, 49152, 57344, 65536)


def padded_length(n):
    """The length a batch whose longest input has n tokens is padded to."""
    for length in PAD_LENGTHS:
        if n <= length:
            return length
    return -(-n // 8192) * 8192


def fast_kernels_active(path):
    """True when transformers will use the fla / causal-conv1d kernels for the linear-attention layers of the model
    in `path`."""
    from transformers import AutoConfig

    kind = AutoConfig.from_pretrained(path).model_type
    try:
        modeling = importlib.import_module(f"transformers.models.{kind}.modeling_{kind}")
    except ImportError:
        return False
    return bool(getattr(modeling, "is_fast_path_available", False))


def load_model(path, device):
    """The checkpoint's own transformers class in bf16 on `device`, and its text decoder (the stack without the
    output head; checkpoints that also carry other towers keep the text decoder under .language_model)."""
    import transformers

    arch = transformers.AutoConfig.from_pretrained(path).architectures[0]
    model = getattr(transformers, arch).from_pretrained(path, dtype=torch.bfloat16, device_map={"": device})
    return model, getattr(model.model, "language_model", model.model)


class StartLuxDecision:
    """decide(state, questions) -> (answers, usage), answers in the TypeSafe /v1/systemone format."""

    def __init__(self, path, device=None, max_length=65536, max_batch_tokens=65536, graphs=True):
        from transformers import AutoTokenizer

        if not os.path.isdir(path):
            raise FileNotFoundError(f"{path}: expected a local StartLux-Decision directory (weights are shared separately)")
        cfg = json.load(open(os.path.join(path, "decision_config.json")))
        self.tok = AutoTokenizer.from_pretrained(path)
        self.letters = J.check_tokenizer(self.tok)
        if self.letters != cfg["letter_token_ids"]:
            raise ValueError("tokenizer letter ids differ from decision_config.json")
        self.temperature = {k: float(v) for k, v in cfg["temperature_by_type"].items()}
        wide = cfg.get("wide_choice", {})
        self.group, self.keep, self.residual = wide.get("group", 25), wide.get("keep", 3), wide.get("residual", 1e-3)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.fast_kernels = fast_kernels_active(path)
        if self.device.type == "cuda" and not self.fast_kernels:
            msg = ("flash-linear-attention and causal-conv1d are not active, so transformers would run the plain torch "
                   "path of the linear-attention layers (>10x slower). pip install flash-linear-attention causal-conv1d, "
                   "or set STARTLUX_ALLOW_SLOW=1 to run anyway.")
            if os.environ.get("STARTLUX_ALLOW_SLOW") != "1":
                raise RuntimeError(msg)
            warnings.warn(msg)
        model, body = load_model(path, self.device)
        self.body = body.eval()
        head = model.get_output_embeddings().weight
        self.letter_rows = head.index_select(0, torch.tensor(self.letters, device=head.device)).float()
        del model                       # only the decoder and the letter rows of the output head are used
        self.pad = self.tok.pad_token_id
        self.max_length, self.max_batch_tokens = int(max_length), int(max_batch_tokens)
        self.graphs = {}
        if graphs and self.device.type == "cuda" and os.environ.get("STARTLUX_GRAPHS", "1") != "0":
            self._capture()

    # ---- CUDA-graph path (the questions of one request as rows, right-padded, no attention mask)
    def _slot_logits(self, ids, last):
        with torch.autocast(self.device.type, dtype=torch.bfloat16):
            hidden = self.body(input_ids=ids, use_cache=False, return_dict=True).last_hidden_state
        h = hidden[torch.arange(ids.shape[0], device=ids.device), last].float()
        return h @ self.letter_rows.T

    @torch.inference_mode()
    def _capture(self):
        shapes = [(b, n) for b in GRAPH_ROWS for n in GRAPH_LENGTHS if b == 1 or n <= MULTI_ROW_MAX_LENGTH]
        pool = None                                 # one memory pool for every graph, sized by the largest
        for b, n in sorted(shapes, key=lambda s: (-s[0] * s[1], -s[1])):
            ids = torch.full((b, n), self.pad, dtype=torch.long, device=self.device)
            last = torch.full((b,), n - 1, dtype=torch.long, device=self.device)
            stream = torch.cuda.Stream(device=self.device)
            stream.wait_stream(torch.cuda.current_stream(self.device))
            with torch.cuda.stream(stream):
                for _ in range(3):                  # autotune and warm every kernel outside the capture
                    self._slot_logits(ids, last)
            torch.cuda.current_stream(self.device).wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, pool=pool):
                out = self._slot_logits(ids, last)
            pool = graph.pool()
            self.graphs[(b, n)] = (graph, ids, last, out)

    def _graph_length(self, enc):
        """The recorded length for these rows, or None when no graph fits."""
        longest = max(len(t) for _, t, _ in enc)
        fits = [n for b, n in self.graphs if b == len(enc) and n >= longest]
        return min(fits) if fits else None

    @torch.inference_mode()
    def _graph_logits(self, enc, n):
        graph, static_ids, last, out = self.graphs[(len(enc), n)]
        ids = torch.full((len(enc), n), self.pad, dtype=torch.long)
        for j, (_, t, _) in enumerate(enc):
            ids[j, :len(t)] = torch.tensor(t)
        static_ids.copy_(ids)
        last.copy_(torch.tensor([len(t) - 1 for _, t, _ in enc]))
        graph.replay()
        res = out.cpu()
        return [res[j, :c].clone() for j, (_, _, c) in enumerate(enc)]

    # ---- readout
    @torch.no_grad()
    def _logits(self, rows):
        """rows: rendered records (<= 26 options) -> (letter logits per row in option order, prompt tokens)."""
        enc = []
        for i, r in enumerate(rows):
            order = [o["id"] for o in r["options"]]
            ids, _ = J.render_ids(r, self.tok, order, max_length=self.max_length)
            enc.append((i, ids, len(order)))
        out, tokens = [None] * len(rows), 0
        n = self._graph_length(enc) if self.graphs and enc else None
        if n is not None:                           # every question of the request in one replay
            for (i, t, _), z in zip(enc, self._graph_logits(enc, n)):
                out[i] = z
                tokens += len(t)
            return out, tokens
        enc.sort(key=lambda x: -len(x[1]))
        start = 0
        while start < len(enc):
            longest = padded_length(len(enc[start][1]))
            n = max(1, min(len(enc) - start, self.max_batch_tokens // longest))
            batch = enc[start:start + n]
            start += n
            ids = torch.full((len(batch), longest), self.pad, dtype=torch.long)
            attn = torch.zeros_like(ids)
            for j, (_, t, _) in enumerate(batch):
                ids[j, :len(t)] = torch.tensor(t)
                attn[j, :len(t)] = 1
                tokens += len(t)
            ids, attn = ids.to(self.device), attn.to(self.device)
            with torch.autocast(self.device.type, dtype=torch.bfloat16):
                hidden = self.body(input_ids=ids, attention_mask=attn, use_cache=False, return_dict=True).last_hidden_state
            pos = attn.sum(1) - 1
            h = hidden[torch.arange(len(batch), device=self.device), pos].float()
            logits = h @ self.letter_rows.T
            for j, (i, _, c) in enumerate(batch):
                out[i] = logits[j, :c].cpu()
        return out, tokens

    def _probs(self, rows):
        logits, tokens = self._logits(rows)
        return [torch.softmax(z / self.temperature.get(r["type"], 1.0), -1).tolist() for r, z in zip(rows, logits)], tokens

    def self_test(self, state, questions):
        """Largest |p_graph - p_eager| over the questions of one request (0.0 when no graphs were recorded)."""
        if not self.graphs:
            return 0.0
        rows = [J.from_systemone(state, q) for q in questions.values()]
        graphed, _ = self._probs(rows)
        graphs, self.graphs = self.graphs, {}
        try:
            eager, _ = self._probs(rows)
        finally:
            self.graphs = graphs
        return max(abs(a - b) for pg, pe in zip(graphed, eager) for a, b in zip(pg, pe))

    def _wide(self, state, spec):
        """Choice lists over 26 options: near-equal groups, the top `keep` of each group go to a final round."""
        keys = list(spec["criteria"])
        n_groups = -(-len(keys) // self.group)
        size, extra = divmod(len(keys), n_groups)
        groups, start = [], 0
        for g in range(n_groups):
            end = start + size + (1 if g < extra else 0)
            groups.append(keys[start:end])
            start = end
        rows = [J.from_systemone(state, dict(spec, criteria={k: spec["criteria"][k] for k in g})) for g in groups]
        first, tokens = self._probs(rows)
        first_p = {k: p for g, ps in zip(groups, first) for k, p in zip(g, ps)}
        finalists = [k for g, ps in zip(groups, first) for k, _ in sorted(zip(g, ps), key=lambda x: -x[1])[:self.keep]]
        fin_spec = dict(spec, criteria={k: spec["criteria"][k] for k in finalists})
        if len(finalists) > J.MAX_OPTIONS:
            final_p, more = self._wide(state, fin_spec)
        else:
            ps, more = self._probs([J.from_systemone(state, fin_spec)])
            final_p = dict(zip(finalists, ps[0]))
        rest = [k for k in keys if k not in set(finalists)]
        mass = sum(first_p[k] for k in rest) or 1.0
        probs = {k: final_p[k] * (1 - self.residual) for k in finalists}
        probs.update({k: self.residual * first_p[k] / mass for k in rest})
        z = sum(probs.values())
        return {k: v / z for k, v in probs.items()}, tokens + more

    # ---- public API
    @staticmethod
    def _answer(row, p, question):
        ids = [o["id"] for o in row["options"]]
        if row["type"] == "noul":
            return {"type": "noul", "noul": p[ids.index("true")]}
        if row["type"] == "score":
            levels = question.get("criteria") or []
            levels = list(levels.values()) if isinstance(levels, dict) else list(levels)
            return {"type": "score", "score": sum(i * v for i, v in enumerate(p)), "confidence": max(p),
                    "legend": {str(i): (levels[i] if i < len(levels) else str(i)) for i in range(len(p))},
                    "probabilities": {str(i): v for i, v in enumerate(p)}}
        dist = dict(zip(ids, p))
        best = max(dist, key=dist.get)
        return {"type": "choice", "choice": best, "confidence": dist[best], "probabilities": dist}

    def _split(self, state, questions):
        """-> (answers decided without the model, [(key, rendered row)], tokens spent on wide lists)"""
        answers, rows, tokens = {}, [], 0
        for k, q in questions.items():
            t = q.get("type", "choice")
            crit = q.get("criteria") or {}
            if t == "choice" and isinstance(crit, dict) and len(crit) == 1:
                only = next(iter(crit))
                answers[k] = {"type": "choice", "choice": only, "confidence": 1.0, "probabilities": {only: 1.0}}
            elif t == "choice" and isinstance(crit, dict) and len(crit) > J.MAX_OPTIONS:
                p, n = self._wide(state, q)
                tokens += n
                best = max(p, key=p.get)
                answers[k] = {"type": "choice", "choice": best, "confidence": p[best], "probabilities": p}
            else:
                rows.append((k, J.from_systemone(state, q)))
        return answers, rows, tokens

    def decide(self, state, questions):
        """One request -> (answers, usage), answers in the TypeSafe /v1/systemone format."""
        answers, rows, tokens = self._split(state, questions)
        if rows:
            probs, n = self._probs([r for _, r in rows])
            tokens += n
            for (k, row), p in zip(rows, probs):
                answers[k] = self._answer(row, p, questions[k])
        return answers, {"input_tokens": tokens, "output_tokens": 0}

    def decide_batch(self, requests):
        """[(state, questions), ...] -> [answers, ...].  Every question of every request goes through one length-sorted
        set of padded forward passes (up to max_batch_tokens each), which is much faster than calling decide() in a loop
        when the requests are short.  Answers are the same as decide() up to bf16 rounding."""
        parts, flat = [], []
        for state, questions in requests:
            answers, rows, _ = self._split(state, questions)
            parts.append((answers, rows, questions))
            flat.extend(r for _, r in rows)
        graphs, self.graphs = self.graphs, {}        # batched eager path; graphs are sized for one request
        try:
            probs, _ = self._probs(flat) if flat else ([], 0)
        finally:
            self.graphs = graphs
        out, i = [], 0
        for answers, rows, questions in parts:
            for k, row in rows:
                answers[k] = self._answer(row, probs[i], questions[k])
                i += 1
            out.append(answers)
        return out
