# Inference

## Install

```bash
hf download startlux-models/Startlux-Decision-4B --local-dir Startlux-Decision-4B      # or any size, see below
pip install -r requirements.txt
```

The five models are in the [Startlux-Decision collection](https://huggingface.co/collections/startlux-models/startlux-decision-6abba92b301b573fa154d493) on Hugging Face: Startlux-Decision-0.8B, 2B, 4B, 9B and 27B, under
`startlux-models/`. The examples below use a local folder called `Startlux-Decision-4B`.

`requirements.txt` includes `flash-linear-attention` and `causal-conv1d`. They matter more than anything else on this
page. The models use linear-attention layers, and without these two packages transformers quietly falls back to a
plain PyTorch implementation that is more than ten times slower. Nothing errors; it is just slow. `causal-conv1d` builds
against your CUDA and PyTorch, and if pip ends up compiling it, add `--no-build-isolation`.

Check that the fast path is really on:

```bash
python -m startlux_decision.check Startlux-Decision-4B        # fast kernels: active
```

On a machine with a GPU it must say `active`. On CUDA, `StartluxDecision(...)` refuses to start when the kernels are not
active; set `STARTLUX_ALLOW_SLOW=1` if you really want to run without them. On a CPU-only machine the check is skipped and
everything runs, slowly.

## Serving

```bash
python -m startlux_decision.server --model Startlux-Decision-4B --port 8090
curl -s localhost:8090/health        # {"status": "ok", "model": "Startlux-Decision-4B", "fast_kernels": true}
```

The server speaks the TypeSafe `/v1/systemone` format and serves one request at a time per GPU. Run one server per GPU
and put a load balancer in front if you need more throughput. The official TypeSafe SDK works against it:

```bash
export TYPESAFE_BASE_URL=http://127.0.0.1:8090
export TYPESAFE_API_KEY=unused
```

## Request and response

A request has a `state` (a string or any JSON value) and `questions`, each with a `type`:

<div align="center">

| type | criteria | answer |
|:---:|:---:|:---:|
| `choice` | `{option: description or null}` | `choice`, `confidence`, `probabilities` over the options |
| `noul` (yes/no) | optional `{"true": ..., "false": ...}` | `noul` = probability of yes |
| `score` | a list of levels, lowest first | `score` (probability-weighted mean level), `confidence`, `legend`, `probabilities` keyed `"0".."n-1"` |

</div>

For the support-ticket request in the README, Startlux-Decision-4B answers (numbers rounded here):

```json
{"answers": {
   "team": {"type": "choice", "choice": "billing", "confidence": 0.880,
            "probabilities": {"billing": 0.880, "shipping": 0.005, "technical": 0.115}},
   "urgent": {"type": "noul", "noul": 0.634},
   "severity": {"type": "score", "score": 1.340, "confidence": 0.454,
                "legend": {"0": "cosmetic", "1": "annoying", "2": "blocks the customer"},
                "probabilities": {"0": 0.114, "1": 0.433, "2": 0.454}}},
 "usage": {"input_tokens": 290, "output_tokens": 0}, "model": "Startlux-Decision-4B", "latency_ms": 27.0}
```

Each question is rendered as its own prompt with the full state, options lettered A, B, C and so on in the order you
give them, and the answer is read from the logits of those letters. A choice question can have up to 26 options in one
pass. Longer lists are split into near-equal groups of up to 25, the top three of each group go to a final round, and
the options that miss the final keep a small share of probability in proportion to their group score, so every option
still gets a non-zero probability.

Temperatures are per question type and live in `decision_config.json`. Changing them never changes which option wins.

## Python

```python
from startlux_decision import StartluxDecision

m = StartluxDecision("Startlux-Decision-4B")                   # device defaults to cuda when available
answers, usage = m.decide(state, questions)
answers_list = m.decide_batch([(state1, questions1), (state2, questions2), ...])
```

`decide` is the latency path. `decide_batch` is the throughput path: every question of every request is sorted by
length and packed into padded forward passes of up to `max_batch_tokens` tokens (65,536 by default). On a random 2%
sample of the Decision Index suite (2,678 requests) Startlux-Decision-4B took 140 s with `decide_batch` against 337 s calling
`decide` once per request, and the chosen options agreed on 99.94% of the 6,897 questions. The differences are bf16
rounding between batch shapes; on 981 JevBench and Typed Decisions questions the largest probability difference was
0.008.

## Why it is fast

Three things, in order of how much they matter.

1. The fast kernels above. Without them everything else is moot.
2. No generation. One forward pass per request, each question one row of the batch, and only 26 rows of the output
   matrix are ever multiplied.
3. CUDA graphs. At start-up the model records graphs in a single shared memory pool: one per padded input length
   (128, 192, 256 ... 4096 tokens) for a single question, and one per question count and length for requests with two
   to four questions of up to 1024 tokens (40 graphs in all). A short request is right-padded to the next length and
   all its questions are replayed as one graph, which removes the per-layer kernel launch overhead that dominates small
   inputs. Padding goes after the last prompt token, every layer is causal and the rows never mix, so it never affects
   the position that is read. Recording adds to start-up time; set `STARTLUX_GRAPHS=0` to skip it. Requests with more
   or longer questions run on the eager path, still as one batch, and bulk work belongs in `decide_batch`.
   Eager and batched inputs are padded to a fixed ladder of lengths, because the linear-attention kernels are compiled
   once per sequence length. At start-up the server runs one request through both the graph and the eager path and
   prints the largest probability difference (0.0 for all five models); above 0.02 it drops the graphs.

![Latency by model size on one H200](../media/latency.png)

Measured end to end over HTTP on one H200, bf16, one request at a time (`eval/latency.py`, 20 warm-up and 200 timed
requests). "3 fields" is one choice, one yes/no and one score question on a support ticket; the three prompts total
891 tokens because each question carries the full state, and the three run together in one forward pass, as
Intern-Decision's fields do.

<div align="center">

| Model | 3 fields: mean | P50 | P95 | one yes/no question |
|:---:|:---:|:---:|:---:|:---:|
| Startlux-Decision-0.8B | 12.2 ms | 12.2 ms | 14.2 ms | 8.3 ms |
| Startlux-Decision-2B | 15.5 ms | 15.5 ms | 17.2 ms | 9.6 ms |
| Startlux-Decision-4B | 26.0 ms | 26.0 ms | 27.3 ms | 14.7 ms |
| Startlux-Decision-9B | 35.7 ms | 36.2 ms | 37.4 ms | 17.6 ms |
| Startlux-Decision-27B | 102.3 ms | 102.5 ms | 104.3 ms | 50.7 ms |
| Startlux-Decision-4B, graphs off | 90.3 ms | 89.5 ms | 92.5 ms | 87.5 ms |

</div>

For reference, Intern-Decision reports 34.0 ms (0.8B), 33.3 ms (2B) and 44.2 ms (4B) on an RTX 4090 for a request of the
same shape. Different hardware, so read it as a ballpark, not a head-to-head. For Jev 1.13 we sent the same two requests
to the TypeSafe API 100 times each on 2026-09-29. Its gateway reports 64.0 ms of server time for the three fields and
63.5 ms for the single yes/no question (`x-envoy-upstream-service-time`, means; medians 57.5 and 58.0 ms); Jev answers all
fields at once, so the count barely matters. End to end from our cluster the requests took about 330 ms, of which about
260 ms is the network. Intern-Decision's 109.7 ms for the Jev API is also an end-to-end number.

## FP8

The models also run with FP8 weights and activations (torchao, `Float8DynamicActivationFloat8WeightConfig` with per-row
scales on the text backbone; the letter readout stays in bf16). On 2,541 decisions (JevBench public, Typed Decisions
test, ToolACE test) the chosen option matched bf16 in this share of cases:

<div align="center">

| Model | agreement with bf16 | accuracy bf16 -> FP8 |
|:---:|:---:|:---:|
| Startlux-Decision-0.8B | 96.3% | 78.00 -> 77.76 |
| Startlux-Decision-2B | 96.5% | 80.44 -> 80.05 |
| Startlux-Decision-4B | 98.0% | 82.37 -> 81.98 |
| Startlux-Decision-9B | 98.2% | 83.67 -> 83.23 |
| Startlux-Decision-27B | 98.6% | 82.96 -> 82.76 |

</div>

FP8 is a memory option here, not a speed option. Weight memory roughly halves (Startlux-Decision-27B takes 27.5 GiB on the GPU
after quantisation), but in this test FP8 was about 2.7 times slower per decision than bf16, both run eagerly with one
question per forward pass: for inputs this short, quantising activations on the fly costs more than the smaller
matmuls save.

One caveat: with dynamic per-row activation scales, an all-zero padding row gets a zero scale and produces NaN, which
then leaks into real rows through attention. When we let FP8 run on padded batches, agreement with bf16 fell to about
52%. Run FP8 one question per forward pass, without padding.

## GGUF and llama.cpp

GGUF files of every size are in `startlux-models/Startlux-Decision-<size>-GGUF`: BF16, Q8_0 and Q4_K_M. They hold the
text decoder only. The prompt format, the option-letter readout and the per-type temperatures stay in this package,
and `startlux_decision.gguf_server` puts them in front of llama-server:

```bash
hf download startlux-models/Startlux-Decision-4B-GGUF --local-dir Startlux-Decision-4B-GGUF \
    --include "*Q8_0.gguf" "*.json" "*.txt" "*.jinja" "LICENSE" "startlux_decision/*"
cd Startlux-Decision-4B-GGUF
pip install -r requirements.txt                     # transformers and torch; a CPU build of torch is enough
llama-server -m Startlux-Decision-4B-Q8_0.gguf -ngl 99 -c 16384 --parallel 4 --port 8081
python -m startlux_decision.gguf_server --model-dir . --llama http://127.0.0.1:8081 --port 8090
```

The server speaks the same `/v1/systemone` format as `startlux_decision.server`. It asks llama-server for the
next-token log-probabilities at the answer position and applies the same readout, temperatures and wide-choice rounds,
so a GGUF file can be compared with the original weights question by question; the GGUF table in the README does that
on the public JevBench items. Plain chat with a GGUF file does not give these decisions. llama.cpp has to be recent
enough to support this model (build b10454 or newer).
