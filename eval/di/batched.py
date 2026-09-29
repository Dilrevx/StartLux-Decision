"""Decision Index runner with requests batched together: the kit's rows, shard filter and result records (the same
file layout as `python -m decision_index run`), but many requests share forward passes through StartLuxDecision.decide_batch().
On an H200 this took a 4B from 3.96 to 2.20 GPU-hours for the whole suite (about 1.8x) with the same index.

    python eval/di/batched.py --model /path/to/StartLux-Decision-4B --suite /path/to/suite-0.2 --out runs/StartLux-Decision-4B \
        --shard 0 --shards 8          # one process per GPU, shards 0..7
    python -m decision_index score --results runs/StartLux-Decision-4B/results.jsonl --edition 0.2.1

Shards write runs/.../shardNNN/results.jsonl; concatenate them (eval/di/merge.py) before scoring.  A request with a
choice list over 26 options, or one that fails to render, falls back to the per-request path with the kit runner's
statuses.  total_wall_ms / model_request_wall_ms are the chunk time spread over its requests: reported latency only,
never read by the scorer.
"""
import argparse
import hashlib
import json
import os
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

ENGINE = "eval.di.startlux_decision_engine:StartLuxDecisionEngine"


def index_ids(edition):
    """Catalog ids that enter the index of an edition (MMLU and other board-only rows are skipped)."""
    try:
        from decision_index.scoring import index02
        spec = index02.spec()
    except Exception:
        return None
    ids = set()
    for area in spec.get("areas", []):
        for b in area.get("benchmarks", area.get("panel", [])):
            ids.add(int(b["catalog_id"]) if isinstance(b, dict) else int(b))
    ids |= {int(n) for n in spec.get("added", [])}
    return ids


def stamp():
    return datetime.now(timezone.utc).isoformat()


class Single(Exception):
    """This request goes through the per-request engine call."""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="local StartLux-Decision directory")
    ap.add_argument("--suite", required=True, help="the kit's built suite directory (suite-0.2)")
    ap.add_argument("--edition", default="0.2")
    ap.add_argument("--out", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--chunk", type=int, default=2048, help="requests rendered and batched together")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    from decision_index import constants as C
    from decision_index.engines import NativeAbstention, Unsupported, load_engine, validate
    from decision_index.runner import iter_rows
    from decision_index.suite.io import Suite, atomic_json, dumps, read_jsonl
    suite = Suite(a.suite, a.edition)
    wanted = index_ids(a.edition)
    keep_ids = suite.in_edition

    def keep(e):                     # di_eval.py run, without --only / --frac
        if wanted and e["catalog_id"] not in wanted:
            return False
        if not keep_ids(e):
            return False
        return int(hashlib.md5(e["run_id"].encode()).hexdigest(), 16) % a.shards == a.shard

    out = Path(a.out) / f"shard{a.shard:03d}"
    out.mkdir(parents=True, exist_ok=True)

    def event(**kw):
        atomic_json(out / "status.json", {"time": stamp(), **kw}, indent=None)

    torch.manual_seed(C.RUN_SEED)
    t = time.perf_counter()
    event(event="loading", engine=ENGINE)
    options = {"model": a.model}
    engine = load_engine(ENGINE, **options)
    engine.synchronize()
    atomic_json(out / "environment.json", {"engine": ENGINE, "engine_options": options, "model_source": engine.provenance, **engine.runtime(),
                                           "loaded_seconds": time.perf_counter() - t, "rows_path": [str(p) for p in suite.row_paths],
                                           "runner": "eval/di/batched.py (requests batched through StartLuxDecision.decide_batch)",
                                           "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "latency": engine.latency})
    engine.warmup()
    engine.synchronize()
    event(event="ready", engine=ENGINE)
    results_path = out / "results.jsonl"
    previous = {}
    if results_path.exists():
        for r in read_jsonl(results_path, complete_lines_only=True):
            previous[r["run_id"]] = r["status"]
        text = results_path.read_text(encoding="utf-8")
        if previous and not text.endswith("\n"):
            results_path.write_text(text[: text.rfind("\n") + 1], encoding="utf-8")
    counts = {}
    start = time.perf_counter()

    def single(row):
        """The kit runner's per-request block."""
        e = row["_evaluation"]
        payload = {"state": row["state"], "questions": row["questions"]}
        t0 = time.perf_counter()
        engine.synchronize()
        result = {**e, "started_utc": stamp(), "engine": ENGINE}
        try:
            response, _ = engine(**payload)
            engine.synchronize()
            validate(payload["questions"], response)
            result.update(status="ok", response=response)
        except NativeAbstention as exc:
            result.update(status="abstained", error=str(exc))
        except Unsupported as exc:
            result.update(status="unsupported", error=str(exc))
        except Exception as exc:
            result.update(status="error", error=str(exc), exception=type(exc).__name__, traceback=traceback.format_exc())
        ms = (time.perf_counter() - t0) * 1000
        result.update(completed_utc=stamp(), total_wall_ms=ms, model_request_wall_ms=ms)
        return result

    def process(chunk, logf):
        t0 = time.perf_counter()
        started = stamp()
        batch_rows, single_rows = [], []
        for row in chunk:                                    # wide lists and unrenderable requests take the kit path
            wide = any(q.get("type") == "choice" and len(q.get("criteria") or {}) > 26 for q in row["questions"].values())
            (single_rows if wide else batch_rows).append(row)
        answers = []
        if batch_rows:
            try:
                answers = engine.m.decide_batch([(r["state"], r["questions"]) for r in batch_rows])
            except (torch.OutOfMemoryError, ValueError):
                torch.cuda.empty_cache()
                single_rows, batch_rows, answers = single_rows + batch_rows, [], []
        engine.synchronize()
        per = (time.perf_counter() - t0) * 1000 / max(1, len(batch_rows))
        done = stamp()
        records = []
        for row, ans in zip(batch_rows, answers):
            response = {"answers": ans}
            result = {**row["_evaluation"], "started_utc": started, "engine": ENGINE}
            try:
                validate(row["questions"], response)
                result.update(status="ok", response=response)
            except Exception as exc:
                result.update(status="error", error=str(exc), exception=type(exc).__name__, traceback=traceback.format_exc())
            result.update(completed_utc=done, total_wall_ms=per, model_request_wall_ms=per)
            records.append(result)
        records.extend(single(row) for row in single_rows)
        for result in records:
            logf.write(dumps(result) + "\n")
            counts[result["status"]] = counts.get(result["status"], 0) + 1
        logf.flush()
        event(event="progress", engine=ENGINE, completed=len(previous) + sum(counts.values()), counts=counts,
              elapsed_seconds=round(time.perf_counter() - start, 1))
        if any(r.get("exception") in ("OutOfMemoryError", "AcceleratorError") or "device-side assert" in r.get("error", "") for r in records):
            raise RuntimeError("Device error; stopping before the accelerator context is reused")

    with results_path.open("a", encoding="utf-8") as logf:
        chunk, seen = [], 0
        for row in iter_rows(suite.row_paths, keep):
            rid = row["_evaluation"]["run_id"]
            if rid in previous and previous[rid] != "error":
                continue
            chunk.append(row)
            seen += 1
            if len(chunk) >= a.chunk:
                process(chunk, logf)
                chunk = []
            if a.limit and seen >= a.limit:
                break
        if chunk:
            process(chunk, logf)
    engine.close()
    event(event="complete", engine=ENGINE, completed=len(previous) + sum(counts.values()), counts=counts,
          elapsed_seconds=round(time.perf_counter() - start, 1))
    print(json.dumps({"shard": a.shard, "counts": counts, "seconds": round(time.perf_counter() - start, 1)}))


if __name__ == "__main__":
    main()
