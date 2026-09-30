# Evaluation

Everything in the results tables can be rerun with the scripts in `eval/`. Each script works in two modes: in-process
with `--model /path/to/StartLux-Decision-4B`, or against any `/v1/systemone` server with `--endpoint http://host:port`, which
is how you would score another system the same way.

## Benchmark data

```bash
bash eval/fetch_benchmarks.sh          # into ./bench, or set BENCH=/somewhere
```

This clones the Intern-Decision benchmark bundle (InternLM/Intern-Decision @ 2f81580) and the JevBench scorer
(fstandhartinger/jevbench @ 7ce310c7), and downloads Typed Decisions (LocalLLaMA/typed-decisions @ f7a2487e; the test
split is scored, the train split only feeds the prior baseline). The revisions are pinned so the numbers stay comparable with ours. The Decision Index suite cannot be
redistributed; it is built with its own kit, see below.

## JevBench public tiers and the Intern-Decision suites

The Intern-Decision bundle has seven accuracy suites: the three public JevBench tiers (easy 48, original 72, hard
111), Typed Decisions test (2,000 decisions), ToolACE test (310), AG News test (7,600) and WildJailBreak test (2,210).

```bash
python eval/suites.py predict --model /path/to/StartLux-Decision-4B --out preds/StartLux-Decision-4B
python eval/suites.py score preds/StartLux-Decision-4B --json scores.json
```

`predict` accepts `--shard i --shards n` to split the work over several GPUs or servers. `score` prints accuracy per
suite, the unweighted average, and Brier and ECE on JevBench hard, then the rows published in the Intern-Decision
README for comparison. Scoring uses upstream JevBench code (`score_task`, `brier_score`, `ece_top_label` with 10
bins) and the bundle's own label sets, so our numbers and theirs come from the same scorer.

"JevBench public" in our tables is the number of correct answers over these 231 items, the sum of the three tiers.
The official JevBench score is a different measurement: it adds a sealed tier, speed and cost, which only the
maintainers can measure, and its public set is not exactly the same items. We do not quote one.

The calibration pilot (96 questions whose correct answer is a known probability distribution) is scored with the
bundle's own scorer:

```bash
python eval/suites.py pilot --model /path/to/StartLux-Decision-4B --out pilot.jsonl
cd bench/intern-decision && python -m src.eval.score_known_distribution \
    --dataset benchmarks/known-distribution-pilot-v1 --predictions ../../pilot.jsonl --output ../../pilot_score
```

We report its expected Brier score and expected ECE from `metrics.json`.

## Typed Decisions

```bash
python eval/typed_decisions.py predict --model /path/to/StartLux-Decision-4B --out typed.jsonl
python eval/typed_decisions.py score typed.jsonl
```

400 cases with five questions each. The script reports accuracy, KL, total variation and Brier against the soft gold
labels, plus a breakdown by question type and workflow, next to the numbers on the dataset card. Our KL, TV and Brier
reproduce the card's Uniform baseline exactly, so those three are comparable with the card. ECE is not: the card does
not say how it computes ECE.

## Decision Index

The Decision Index is built and scored with its reproduction kit
([apolinario/decision-index](https://github.com/apolinario/decision-index), version 0.2.1). Install it and build the
suite once, following its README (about 7 GB of downloads; you need to accept the HLE terms on Hugging Face):

```bash
git clone https://github.com/apolinario/decision-index && cd decision-index
pip install -e ".[transformers,rebuild]"
python -m decision_index suite rebuild --work work
python -m decision_index suite import \
    --rows work/artifacts/benchmark-suite/release-v2-rebuilt/selected-rows.jsonl.gz \
    --added-rows work/artifacts/benchmark-suite/release-v2-rebuilt/added-rows.jsonl.gz
```

The kit's `run` command needs Python 3.11 or newer (it hashes the suite files with `hashlib.file_digest`); `score` and
`eval/di/batched.py` also work on 3.10. Then, from the root of this repository, either use the kit's own runner, which
sends one request at a time:

```bash
python -m decision_index run --engine eval.di.startlux_decision_engine:StartLuxDecisionEngine \
    --option model=/path/to/StartLux-Decision-4B --option graphs=false --suite-dir /path/to/suite-0.2 --out runs/StartLux-Decision-4B
```

or the batched runner, one process per GPU:

```bash
for i in 0 1 2 3 4 5 6 7; do
  CUDA_VISIBLE_DEVICES=$i python eval/di/batched.py --model /path/to/StartLux-Decision-4B --suite /path/to/suite-0.2 \
      --out runs/StartLux-Decision-4B --shard $i --shards 8 &
done; wait
python eval/di/merge.py runs/StartLux-Decision-4B
```

Score the same results file under both editions:

```bash
python -m decision_index score --results runs/StartLux-Decision-4B/results.jsonl --suite-dir /path/to/suite-0.2 --edition 0.2
python -m decision_index score --results runs/StartLux-Decision-4B/results.jsonl --suite-dir /path/to/suite-0.2 --edition 0.2.1
```

0.2.1 reads the same files as 0.2, so one run gives both numbers. The index is `scores.balanced_skill` in the
`index.json` the kit writes.

What to expect. With this code, the batched runner on five H200s took about 25 minutes from start to the last shard
for StartLux-Decision-4B (about 1.9 GPU-hours, model loading and kernel compilation included) and scored 48.42 on 0.2 and 52.75
on 0.2.1, against 48.38 and 52.75 from our internal evaluation. To compare the three ways of running it, we timed
StartLux-Decision-4B on the same random 2% of the suite (2,678 requests) on one H200: 337 s through the kit runner with CUDA
graphs, 254 s through the kit runner with `--option graphs=false`, and 140 s with requests batched together. Graphs help
single short requests but not this mix, where many requests carry several longer questions that the eager path runs
together. The batched and one-at-a-time answers picked the same option for 99.94% of the questions.

Two details. `batched.py` runs only the benchmarks that count in the index, so the kit marks the run incomplete (the
board-only MMLU, ARC and SimpleBench rows are not run); the index itself is computed over complete benchmarks. And the
per-request times it writes are chunk averages, so measure latency with the kit runner or `eval/latency.py`.

## Latency

```bash
python -m startlux_decision.server --model /path/to/StartLux-Decision-4B --port 8090 &
python eval/latency.py http://127.0.0.1:8090/v1/systemone 200            # one choice, one yes/no, one score field
python eval/latency.py http://127.0.0.1:8090/v1/systemone 200 single     # one yes/no field
```

The request mirrors the one Intern-Decision uses for its latency numbers: a support ticket with one field of each type.
The script sends 20 warm-up requests and then N timed ones, one at a time, and reports mean, P50, P95 and the token
count the server saw.

## Games and other tasks

The game results in [results.md](results.md) (Dino Run, chess, NPC addressee detection) were
run with the benchmark authors' own harnesses against a StartLux-Decision `/v1/systemone` server: through the TypeSafe SDK
(`TYPESAFE_BASE_URL`) where the harness uses it, and through a small backend class calling the same endpoint in the
arena harness. We did not change their prompts, state renderers or scoring, so this repository does not repeat their
instructions; follow each harness's README.
