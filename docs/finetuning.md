# Fine-tuning on your own decisions

Startlux-Decision handles most routing, triage and classification questions as it is. Fine-tuning pays off when your labels
follow house rules the model cannot guess: what counts as urgent for your team, which queue owns an edge case, where
the line between two severity levels sits. A LoRA run on a few thousand labelled examples is usually enough, and it
fits on one GPU.

## Data

One JSON object per line, in the same shape as a request plus the answers you want:

```json
{"state": {"ticket": "Refund still missing after two weeks, order 5521", "customer_tier": "pro"},
 "questions": {
   "team": {"type": "choice", "instructions": "Which team should handle this ticket?",
            "criteria": {"billing": "Payments, refunds and invoices", "shipping": "Delivery and tracking",
                         "technical": "App bugs and outages"}},
   "urgent": {"type": "noul", "instructions": "Does the customer ask for this to be solved today?"},
   "severity": {"type": "score", "instructions": "How badly is the customer affected?",
                "criteria": ["cosmetic", "annoying", "blocks the customer"]}},
 "targets": {"team": {"billing": 1.0}, "urgent": {"false": 1.0}, "severity": {"1": 0.7, "2": 0.3}}}
```

Target keys follow the question type: the criteria keys for `choice`, `"true"` and `"false"` for yes/no, and `"0"` to
`"n-1"` for a score, lowest level first. A single label is a one-hot target. When several people labelled the same item,
pass their vote shares as they are (`"severity"` above); the loss uses the whole distribution. Questions without a
target are skipped, and so are choice lists with more than 26 options (split those into groups of up to 26 first).

Keep 10 to 20% of your data aside as a dev set, and do not tune on anything you will later report as a test score.

## Train

```bash
python finetune/finetune_lora.py --model /path/to/Startlux-Decision-4B --train train.jsonl --dev dev.jsonl --out runs/mine
```

Each question is rendered exactly as the server renders it, and the loss is the cross-entropy between your target and
the softmax over the option letters at the answer position. That is the readout the model is served with, so nothing
changes between training and serving. Only LoRA adapters on the linear layers of the language model are trained. At the
end they are merged into a full copy of the model in `runs/mine/merged`, which you serve like any other Startlux-Decision
directory; `--no-merge` saves only the adapter.

Defaults: rank 16, alpha 32, dropout 0.05, learning rate 1e-4 with 5% warm-up and cosine decay, 2 epochs, micro-batches
of up to 16,384 padded tokens and 4 micro-batches per optimizer step. The script prints dev loss and accuracy before
training, after each epoch and at the end. Inputs are padded to a small set of lengths so the fast kernels are compiled
once per length rather than once per batch.

For a sense of speed: on one H200, Startlux-Decision-4B went through 6,000 short synthetic ticket questions (about one epoch,
13 optimizer steps) in about three minutes, kernel compilation included. The script uses a single GPU; for Startlux-Decision-27B,
the bf16 weights alone take about 54 GB, so plan for an 80 GB card or larger.

Before you ship the result, run your dev set and a sample of the other decisions you rely on through `eval/` with both
the original and the fine-tuned model. If the other decisions got worse, use fewer epochs or a lower learning rate.

## Calibrate

```bash
python finetune/calibrate.py --model runs/mine/merged --dev dev.jsonl
```

This fits one temperature per question type (choice, yes/no, score) by minimising the negative log-likelihood of your
dev targets, prints ECE and NLL before and after, and writes the temperatures into the model's `decision_config.json`
(`--dry-run` only prints). A temperature never changes which option wins; it only makes the probabilities sharper or
flatter. Types with fewer than 200 dev questions get the pooled temperature.

Two things to watch. If the model is right on nearly every dev question, the fit pushes the temperature to the bottom
of its range (0.2) and the probabilities become very sharp, which is rarely what you want on new data; use a larger or
harder dev set. And a temperature fitted on one source of data can over- or under-correct on another, so check it on a
second held-out set if you have one.

## Serve

```bash
python -m startlux_decision.server --model runs/mine/merged --port 8090
```
