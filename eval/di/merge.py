"""Concatenate the shard results of eval/di/batched.py into one results.jsonl for the kit's scorer.

    python eval/di/merge.py runs/Startlux-Decision-4B          -> runs/Startlux-Decision-4B/results.jsonl
    python -m decision_index score --results runs/Startlux-Decision-4B/results.jsonl --edition 0.2.1   (and --edition 0.2)
"""
import json
import sys
from pathlib import Path

out = Path(sys.argv[1])
best = {}
for p in sorted(out.glob("shard*/results.jsonl")):
    for line in open(p, encoding="utf-8"):
        if line.strip():
            r = json.loads(line)
            if r["run_id"] not in best or best[r["run_id"]]["status"] != "ok":
                best[r["run_id"]] = r
with open(out / "results.jsonl", "w", encoding="utf-8") as f:
    for r in best.values():
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(json.dumps({"requests": len(best), "ok": sum(r["status"] == "ok" for r in best.values())}))
