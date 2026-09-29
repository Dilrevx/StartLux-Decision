"""Seven accuracy suites plus the calibration pilot from the Intern-Decision benchmark bundle.

    python eval/suites.py predict --model /path/to/Startlux-Decision-4B --out preds/Startlux-Decision-4B          # in-process
    python eval/suites.py predict --endpoint http://127.0.0.1:8090 --out preds/server          # any /v1/systemone server
    python eval/suites.py score preds/Startlux-Decision-4B [preds/other ...] [--json scores.json]
    python eval/suites.py pilot --model /path/to/Startlux-Decision-4B --out pilot_predictions.jsonl

The suites are JevBench easy / original / hard (the public tiers), Typed Decisions test, ToolACE test, AG News test and
WildJailBreak test: 10,751 rows and 12,351 decisions.  Scoring follows the bundle exactly: upstream jevbench (commit
7ce310c7) score_task, brier_score and ece_top_label, their label sets, and their expected labels.  Accuracy counts every
decision; Average is the unweighted mean of the seven suites; Brier and ECE (max-P confidence, 10 bins) are on
JevBench hard.  Score the pilot with the bundle's own scorer:
    cd $BENCH/intern-decision && python -m src.eval.score_known_distribution \
        --dataset benchmarks/known-distribution-pilot-v1 --predictions PRED.jsonl --output OUT

Run eval/fetch_benchmarks.sh first; BENCH (default ./bench) holds the bundle, the pilot and the pinned jevbench.
"""
import argparse
import glob
import json
import os
import sys
from types import SimpleNamespace

HERE = os.path.dirname(os.path.abspath(__file__))
BENCH = os.environ.get("BENCH", os.path.join(os.path.dirname(HERE), "bench"))
BUNDLE = f"{BENCH}/intern-decision/benchmarks/accuracy-v1"
PILOT = f"{BENCH}/intern-decision/benchmarks/known-distribution-pilot-v1"
JEVBENCH = f"{BENCH}/jevbench"
SUITES = [("jevbench-easy", "jevbench/easy.jsonl"), ("jevbench-original", "jevbench/original.jsonl"),
          ("jevbench-hard", "jevbench/hard.jsonl"), ("typed_decisions-test", "typed_decisions/test.jsonl"),
          ("toolace-test", "toolace/test.jsonl"), ("agnews-test", "agnews/test.jsonl"),
          ("wildjailbreak-test", "wildjailbreak/test.jsonl")]
SHORT = {"jevbench-easy": "Easy", "jevbench-original": "Original", "jevbench-hard": "Hard", "typed_decisions-test": "Typed",
         "toolace-test": "ToolACE", "agnews-test": "AG News", "wildjailbreak-test": "WildJB"}
# their README table (accuracy %, then Jevbench-Hard Brier / ECE)
THEIRS = [
    ("Jev", [100.00, 98.61, 72.07, 73.35, 91.29, 89.57, 96.29], 0.358420, 0.094685),
    ("JevK5", [100.00, 97.22, 73.87, 64.50, 80.97, 89.13, 90.45], 0.366162, 0.046672),
    ("SemIf", [100.00, 98.61, 61.26, 62.80, 85.16, 89.22, 92.53], 0.498012, 0.112213),
    ("Kev", [100.00, 93.06, 45.05, 65.60, 87.42, 89.82, 75.97], 0.738253, 0.262200),
    ("Intern-Decision-2B", [100.00, 84.72, 63.96, 79.35, 96.45, 89.96, 78.33], 0.437256, 0.100208),
    ("Intern-Decision-4B", [100.00, 98.61, 73.87, 80.55, 96.45, 90.82, 89.86], 0.346762, 0.065267),
]


def read(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def canonical(raw):
    if "questions" in raw:
        return raw
    return {"id": raw["id"], "state": raw["state"], "questions": {"decision": raw["question"]}}


def options(q):
    """Their label set per question (src/inputs/schema.py `_options`)."""
    kind, crit = q.get("type"), q.get("criteria")
    if kind == "choice":
        return [str(k) for k in crit]
    if kind == "score":
        return [str(i) for i in range(len(crit))] if isinstance(crit, list) else [str(k) for k in crit]
    if kind == "noul":
        return ["no", "yes"]
    raise ValueError(f"unsupported question type {kind!r}")


def answer_value(q, target):
    """Their expected label (src/inputs/schema.py `_answer_value`)."""
    answer = q.get("answer") if isinstance(q.get("answer"), dict) else target
    if not isinstance(answer, dict):
        return None
    kind = q.get("type")
    if "label" in answer:
        v = answer["label"]
        if kind == "noul":
            return "yes" if str(v).lower() in {"yes", "true", "1"} else "no"
        return str(v)
    if kind == "choice":
        return None if answer.get("choice") is None else str(answer["choice"])
    if kind == "score":
        v = answer.get("score")
        return None if v is None else str(int(v) if isinstance(v, float) and v.is_integer() else v)
    if kind == "noul":
        return None if answer.get("noul") is None else ("yes" if float(answer["noul"]) >= 0.5 else "no")
    return None


def to_labels(q, ans, score_keys):
    """One engine answer -> {label: p} over options(q)."""
    labels = options(q)
    if q["type"] == "noul":
        p = float(ans["noul"]) if "noul" in ans else float(ans["probabilities"].get("true", ans["probabilities"].get("yes")))
        return {"no": 1.0 - p, "yes": p}
    probs = {str(k): float(v) for k, v in ans["probabilities"].items()}
    crit = q.get("criteria")
    if q["type"] == "score":
        if isinstance(crit, dict):                                  # level i is the i-th of the renderer's score_keys
            probs = {str(k): probs.get(str(i), 0.0) for i, k in enumerate(score_keys(crit))}
        elif not all(label in probs for label in labels):            # some servers key score levels by description
            probs = {str(i): probs.get(str(c), 0.0) for i, c in enumerate(crit)}
    return {label: probs.get(label, 0.0) for label in labels}


# ---------------------------------------------------------------- predict

def engine(a):
    """-> (ask(state, questions) -> answers, score_keys)"""
    if a.endpoint:
        import urllib.request
        url = a.endpoint.rstrip("/") + "/v1/systemone"
        key = os.environ.get(a.api_key_env, "")

        def ask(state, questions):
            body = json.dumps({"model": a.model_name, "state": state, "questions": questions}).encode()
            headers = {"Content-Type": "application/json", **({"Authorization": f"Bearer {key}"} if key else {})}
            with urllib.request.urlopen(urllib.request.Request(url, body, headers), timeout=300) as r:
                return json.loads(r.read())["answers"]

        def score_keys(crit):
            try:
                return sorted(crit, key=lambda k: float(k))
            except (TypeError, ValueError):
                return list(crit)
        return ask, score_keys
    sys.path.insert(0, os.path.dirname(HERE))
    from startlux_decision import StartluxDecision
    from startlux_decision import jevfmt as J
    model = StartluxDecision(a.model)
    if a.temperature is not None:
        model.temperature = {k: float(a.temperature) for k in model.temperature}

    def ask(state, questions):
        return model.decide(state, questions)[0]
    return ask, J.score_keys


def predict(a):
    ask, score_keys = engine(a)
    os.makedirs(a.out, exist_ok=True)
    for name, rel in SUITES:
        rows = read(f"{BUNDLE}/{rel}")
        with open(f"{a.out}/{name}.{a.shard:03d}.jsonl", "w") as out:
            for line, raw in enumerate(rows, 1):
                if (line - 1) % a.shards != a.shard:
                    continue
                row = canonical(raw)
                ans = ask(row["state"], row["questions"])
                out.write(json.dumps({"id": raw["id"], "line": line,
                                      "answers": {f: to_labels(q, ans[f], score_keys) for f, q in row["questions"].items()}}) + "\n")
        print(json.dumps({"suite": name, "shard": a.shard, "done": True}), flush=True)


def pilot(a):
    ask, score_keys = engine(a)
    with open(a.out, "w") as out:
        for row in read(f"{PILOT}/inputs.jsonl"):
            ans = ask(row["state"], row["questions"])
            answers = {}
            for f, q in row["questions"].items():
                p = to_labels(q, ans[f], score_keys)
                answers[f] = {"type": q["type"], "probabilities": p}
            out.write(json.dumps({"id": row["id"], "answers": answers}) + "\n")
    print("wrote", a.out)


# ---------------------------------------------------------------- score

def score(pred_dir):
    sys.path.insert(0, JEVBENCH)
    from jevbench.metrics import brier_score, ece_top_label
    from jevbench.scoring import score_task
    from jevbench.tasks import Task
    res = {}
    for name, rel in SUITES:
        pred = {}
        for f in glob.glob(f"{pred_dir}/{name}.*.jsonl"):
            for p in read(f):
                pred[(p["line"], p["id"])] = p
        pairs, briers, invalid, missing = [], [], 0, 0
        for line, raw in enumerate(read(f"{BUNDLE}/{rel}"), 1):
            p = pred.get((line, raw["id"]))
            if p is None:
                missing += 1
                continue
            row = canonical(raw)
            for field, q in row["questions"].items():
                if "question" in raw:
                    task = Task.from_dict(raw)
                else:
                    task = SimpleNamespace(question=q, labels=options(q), expected=answer_value(q, (raw.get("targets") or {}).get(field)))
                s = score_task(p["answers"][field], task)
                if not s["valid"]:
                    invalid += 1
                    pairs.append((0.0, False))
                    continue
                if s["correct"] is not None:
                    pairs.append((max(s["probs"].values()), s["correct"]))
                    briers.append(brier_score(s["probs"], str(task.expected), task.labels))
        res[name] = {"total": len(pairs), "correct": sum(int(c) for _, c in pairs), "accuracy": sum(int(c) for _, c in pairs) / max(1, len(pairs)),
                     "brier": sum(briers) / max(1, len(briers)), "ece": ece_top_label(pairs, n_bins=10)["ece"],
                     "invalid": invalid, "missing_rows": missing}
    return res


def show(a):
    head = f"{'model':24s} " + " ".join(f"{SHORT[n]:>8s}" for n, _ in SUITES) + f" {'Average':>8s} {'Brier':>7s} {'ECE':>7s}"
    print(head)
    out = {}
    for d in a.pred:
        r = score(d)
        out[d] = r
        acc = [100 * r[n]["accuracy"] for n, _ in SUITES]
        h = r["jevbench-hard"]
        warn = "".join(f"  [{n}: {r[n]['missing_rows']} rows missing, {r[n]['invalid']} invalid]" for n, _ in SUITES
                       if r[n]["missing_rows"] or r[n]["invalid"])
        print(f"{os.path.basename(d.rstrip('/')):24s} " + " ".join(f"{x:8.2f}" for x in acc)
              + f" {sum(acc) / len(acc):8.2f} {h['brier']:7.4f} {h['ece']:7.4f}" + warn)
    print("-- published (Intern-Decision README)")
    for name, acc, brier, ece in THEIRS:
        print(f"{name:24s} " + " ".join(f"{x:8.2f}" for x in acc) + f" {sum(acc) / len(acc):8.2f} {brier:7.4f} {ece:7.4f}")
    if a.json:
        with open(a.json, "w") as f:
            json.dump(out, f, indent=1)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("predict", "pilot"):
        p = sub.add_parser(name)
        p.add_argument("--model", help="local Startlux-Decision directory (in-process)")
        p.add_argument("--endpoint", help="base URL of a /v1/systemone server instead of --model")
        p.add_argument("--model-name", default="Startlux-Decision", help="value of the request's model field (endpoint mode)")
        p.add_argument("--api-key-env", default="SYSTEMONE_API_KEY", help="env var holding a bearer token (endpoint mode)")
        p.add_argument("--temperature", help="override every per-type temperature (in-process only)")
        p.add_argument("--out", required=True)
        if name == "predict":
            p.add_argument("--shard", type=int, default=0)
            p.add_argument("--shards", type=int, default=1)
    s = sub.add_parser("score")
    s.add_argument("pred", nargs="+")
    s.add_argument("--json")
    a = ap.parse_args()
    if a.cmd != "score" and not (a.model or a.endpoint):
        ap.error("--model or --endpoint is required")
    {"predict": predict, "pilot": pilot, "score": show}[a.cmd](a)


if __name__ == "__main__":
    main()
