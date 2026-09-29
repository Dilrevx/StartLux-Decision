"""LoRA fine-tuning of a Startlux-Decision model on your own typed decisions, written out as a new Startlux-Decision directory.

    python finetune/finetune_lora.py --model /path/to/Startlux-Decision-4B --train train.jsonl --dev dev.jsonl --out runs/mine
    python finetune/calibrate.py --model runs/mine/merged --dev dev.jsonl
    python -m startlux_decision.server --model runs/mine/merged --port 8090

Data, one JSON object per line (see docs/finetuning.md):

    {"state": <string or object>, "questions": {key: <TypeSafe question>}, "targets": {key: {option: probability}}}

Target keys follow the question type: choice -> its criteria keys, noul -> "true" / "false", score -> "0".."n-1"
(lowest level first).  A hard label is a one-hot distribution; a spread of annotator votes can be given as is.

Each question is rendered exactly as Startlux-Decision renders it when serving, and the loss is the cross-entropy between the
target and the softmax over the option letters at the answer position, which is the readout Startlux-Decision uses, so the
result is served with the same code.  Only LoRA weights are trained; they are merged into the saved model.
"""
import argparse
import json
import math
import os
import random
import shutil
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from startlux_decision import jevfmt as J  # noqa: E402
from startlux_decision.model import load_model, padded_length  # noqa: E402


def load_items(path, tok, max_length):
    """-> [(input_ids, number of options, target vector in option order, question type)]"""
    items, skipped = [], 0
    with open(path, encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            if not line.strip():
                continue
            rec = json.loads(line)
            for key, q in rec["questions"].items():
                target = (rec.get("targets") or {}).get(key)
                if target is None:
                    continue
                row = J.from_systemone(rec["state"], q, qid=key)
                order = [o["id"] for o in row["options"]]
                if row["type"] == "choice" and len(order) > J.MAX_OPTIONS:
                    skipped += 1                      # split very wide lists into groups of up to 26 before training
                    continue
                vec = [float(target.get(o, 0.0)) for o in order]
                total = math.fsum(vec)
                if total <= 0:
                    raise ValueError(f"{path}:{n} question {key!r}: target has no mass on the listed options")
                try:
                    ids, _ = J.render_ids(row, tok, order, max_length=max_length)
                except ValueError:
                    skipped += 1
                    continue
                items.append((ids, len(order), [v / total for v in vec], row["type"]))
    return items, skipped


def batches(items, max_tokens, shuffle, seed):
    """Length-sorted buckets of up to max_tokens padded tokens, shuffled bucket order."""
    order = sorted(range(len(items)), key=lambda i: -len(items[i][0]))
    out, cur, longest = [], [], 0
    for i in order:
        n = padded_length(len(items[i][0]))
        if cur and max(longest, n) * (len(cur) + 1) > max_tokens:
            out.append(cur)
            cur, longest = [], 0
        cur.append(i)
        longest = max(longest, n)
    if cur:
        out.append(cur)
    if shuffle:
        random.Random(seed).shuffle(out)
    return out


def step_loss(body, letter_rows, items, idx, pad, device):
    longest = padded_length(max(len(items[i][0]) for i in idx))     # a few fixed lengths: kernels compile once each
    k = max(items[i][1] for i in idx)
    ids = torch.full((len(idx), longest), pad, dtype=torch.long)
    attn = torch.zeros_like(ids)
    target = torch.zeros((len(idx), k))
    mask = torch.zeros((len(idx), k), dtype=torch.bool)
    for j, i in enumerate(idx):
        t, c, vec, _ = items[i]
        ids[j, :len(t)] = torch.tensor(t)
        attn[j, :len(t)] = 1
        target[j, :c] = torch.tensor(vec)
        mask[j, :c] = True
    ids, attn, target, mask = ids.to(device), attn.to(device), target.to(device), mask.to(device)
    with torch.autocast(device.type, dtype=torch.bfloat16):
        hidden = body(input_ids=ids, attention_mask=attn, use_cache=False, return_dict=True).last_hidden_state
    h = hidden[torch.arange(len(idx), device=device), attn.sum(1) - 1].float()
    logits = (h @ letter_rows[:k].T).masked_fill(~mask, -torch.inf)
    logp = F.log_softmax(logits, -1).masked_fill(~mask, 0.0)
    loss = -(target * logp).sum(-1).mean()
    correct = (logits.argmax(-1) == target.argmax(-1)).float().sum().item()
    return loss, correct


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="Startlux-Decision directory to start from")
    ap.add_argument("--train", required=True)
    ap.add_argument("--dev")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=float, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--max-tokens", type=int, default=16384, help="padded tokens per micro-batch")
    ap.add_argument("--accum", type=int, default=4, help="micro-batches per optimizer step")
    ap.add_argument("--max-length", type=int, default=8192)
    ap.add_argument("--warmup", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-merge", action="store_true", help="only save the LoRA adapter")
    a = ap.parse_args()

    from peft import LoraConfig, get_peft_model
    from transformers import AutoTokenizer

    torch.manual_seed(a.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = json.load(open(os.path.join(a.model, "decision_config.json")))
    tok = AutoTokenizer.from_pretrained(a.model)
    if J.check_tokenizer(tok) != cfg["letter_token_ids"]:
        raise ValueError("tokenizer letter ids differ from decision_config.json")
    train, skipped = load_items(a.train, tok, a.max_length)
    dev, _ = load_items(a.dev, tok, a.max_length) if a.dev else ([], 0)
    print(f"{len(train)} training questions ({skipped} skipped: too long or over 26 options), {len(dev)} dev questions", flush=True)

    model, body = load_model(a.model, device)
    head = model.get_output_embeddings().weight
    letter_rows = head.index_select(0, torch.tensor(cfg["letter_token_ids"], device=head.device)).detach().float()
    body.gradient_checkpointing_enable()
    body.enable_input_require_grads()
    body = get_peft_model(body, LoraConfig(r=a.rank, lora_alpha=a.alpha, lora_dropout=a.dropout, target_modules="all-linear"))
    body.print_trainable_parameters()

    plan = batches(train, a.max_tokens, True, a.seed)
    total_steps = max(1, math.ceil(len(plan) * a.epochs / a.accum))
    params = [p for p in body.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
    warm = max(1, int(a.warmup * total_steps))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / total_steps))))
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    def evaluate():
        if not dev:
            return None
        body.eval()
        loss_sum, correct, n = 0.0, 0.0, 0
        with torch.no_grad():
            for idx in batches(dev, a.max_tokens, False, 0):
                loss, c = step_loss(body, letter_rows, dev, idx, pad, device)
                loss_sum += loss.item() * len(idx)
                correct += c
                n += len(idx)
        body.train()
        return {"dev_loss": round(loss_sum / n, 4), "dev_accuracy": round(correct / n, 4)}

    print(json.dumps({"optimizer_steps": total_steps, "micro_batches_per_epoch": len(plan), "before": evaluate()}), flush=True)
    body.train()
    step, micro, t0 = 0, 0, time.time()
    epoch = 0
    while step < total_steps:
        for idx in batches(train, a.max_tokens, True, a.seed + epoch):
            loss, _ = step_loss(body, letter_rows, train, idx, pad, device)
            (loss / a.accum).backward()
            micro += 1
            if micro % a.accum:
                continue
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            if step % 10 == 0 or step == total_steps:
                print(json.dumps({"step": step, "loss": round(loss.item(), 4), "lr": sched.get_last_lr()[0],
                                  "elapsed_s": round(time.time() - t0)}), flush=True)
            if step >= total_steps:
                break
        epoch += 1
        if step < total_steps:
            print(json.dumps({"epoch": epoch, **(evaluate() or {})}), flush=True)
    print(json.dumps({"done": {"optimizer_steps": step, "elapsed_s": round(time.time() - t0), **(evaluate() or {})}}), flush=True)

    os.makedirs(a.out, exist_ok=True)
    body.save_pretrained(os.path.join(a.out, "adapter"))
    if a.no_merge:
        return
    body.merge_and_unload()                                 # merges the LoRA weights into the decoder in place
    merged = os.path.join(a.out, "merged")
    model.save_pretrained(merged, safe_serialization=True)
    tok.save_pretrained(merged)
    for f in os.listdir(a.model):                           # processor configs and the Startlux-Decision config travel along
        if f.endswith(".json") and not os.path.exists(os.path.join(merged, f)):
            shutil.copy(os.path.join(a.model, f), merged)
    print("merged model written to", merged, "- fit its temperatures with finetune/calibrate.py", flush=True)


if __name__ == "__main__":
    main()
