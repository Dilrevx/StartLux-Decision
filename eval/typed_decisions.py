"""Typed Decisions benchmark (LocalLLaMA/typed-decisions, test split: 400 cases x 5 questions = 2,000 decisions).

    python eval/typed_decisions.py predict --model /path/to/StartLux-Decision-4B --out preds.jsonl
    python eval/typed_decisions.py predict --endpoint http://127.0.0.1:8090 --out preds.jsonl
    python eval/typed_decisions.py baseline --kind uniform|prior --out preds.jsonl
    python eval/typed_decisions.py score preds.jsonl [more.jsonl ...]

A prediction line is {"id", "answers": {question: {option: p}}, "ms"} with options keyed like the gold: yes/no as
"false"/"true", score levels "0".."n-1", choice criteria keys.  "ms" is the wall time of one case (5 questions, one
request at a time).

Metrics over all 2,000 decisions.  KL = KL(gold || pred), TV = half L1, Brier = sum over options of (pred - gold)^2
against the soft gold; these three reproduce the card's Uniform row exactly (0.444 / 0.381 / 0.238).  Accuracy = argmax
vs the gold label; soft acc = gold mass on the predicted option; macro F1 per question schema, averaged over the 20;
ECE = top-label, 15 bins; score MAE = |expected level - gold expected score|; within-1 = |argmax level - gold level| <= 1.
The card's Prior row differs slightly from ours (its prior is smoothed), and its ECE definition is not published, so
do not compare ECE with the card.

"""
import argparse
import collections
import glob
import json
import math
import os
import statistics
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
BENCH = os.environ.get("BENCH", os.path.join(os.path.dirname(HERE), "bench"))
TEST = f"{BENCH}/typed_decisions/all/test-00000-of-00001.parquet"
TRAIN = f"{BENCH}/typed_decisions/all/train-*.parquet"           # only for the prior baseline
EPS = 1e-9

# card leaderboard (dataset revision f7a2487e), test split
CARD = [
    ("meraGPT Decider 1 (general)", dict(acc=0.768, soft=0.608, mf1=0.641, kl=0.096, tv=0.149, brier=0.052, mae=0.219, w1=0.984, ms=526)),
    ("TypeSafe Jev 1.13.0 (general)", dict(acc=0.727, soft=0.580, mf1=0.613, kl=1.442, tv=0.251, brier=0.148, mae=0.391, w1=0.952, ms=710)),
    ("Featherless Simple Jev (general)", dict(acc=0.716, kl=0.488, brier=0.176)),
    ("ModernBERT-base (specialist)", dict(acc=0.646, soft=0.542, mf1=0.469, kl=0.223, tv=0.249, brier=0.119, mae=0.444, w1=0.931, ms=349)),
    ("Prior (reference)", dict(acc=0.470, soft=0.430, mf1=0.207, kl=0.347, tv=0.317, brier=0.189)),
    ("teacher self-agreement (ceiling)", dict(acc=0.735)),
]


def rows(path):
    import pyarrow.parquet as pq
    return pq.read_table(path).to_pylist()


def option_keys(q):
    c = q.get("criteria")
    if isinstance(c, list):
        return [str(i) for i in range(len(c))]
    if c:
        return [str(k) for k in c]
    return ["false", "true"]                                    # noul questions without criteria


def to_dist(q, ans):
    """An engine answer -> {option key: p} over option_keys(q)."""
    keys = option_keys(q)
    if "noul" in ans:
        p = float(ans["noul"])
        return {"false": 1 - p, "true": p}
    probs = {str(k): float(v) for k, v in ans["probabilities"].items()}
    crit = q.get("criteria")
    if isinstance(crit, list) and not all(k in probs for k in keys):   # score answered with level descriptions as keys
        probs = {str(i): probs.get(str(c), 0.0) for i, c in enumerate(crit)}
    return {k: probs.get(k, 0.0) for k in keys}


# ---------------------------------------------------------------- predict

def predict(a):
    test = rows(TEST)
    if a.endpoint:
        import urllib.request
        url = a.endpoint.rstrip("/") + "/v1/systemone"
        key = os.environ.get(a.api_key_env, "")

        def ask(state, questions):
            body = json.dumps({"model": a.model_name, "state": state, "questions": questions}).encode()
            headers = {"Content-Type": "application/json", **({"Authorization": f"Bearer {key}"} if key else {})}
            with urllib.request.urlopen(urllib.request.Request(url, body, headers), timeout=300) as r:
                return json.loads(r.read())["answers"]
    else:
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from startlux_decision import StartLuxDecision
        model = StartLuxDecision(a.model)
        if a.temperature is not None:
            model.temperature = {k: float(a.temperature) for k in model.temperature}

        def ask(state, questions):
            return model.decide(state, questions)[0]
    for r in test[:3]:                                          # warm-up, not timed
        ask(json.loads(r["state"]), json.loads(r["questions"]))
    with open(a.out, "w") as out:
        for r in test:
            state, qs = json.loads(r["state"]), json.loads(r["questions"])
            t0 = time.perf_counter()
            ans = ask(state, qs)
            ms = (time.perf_counter() - t0) * 1000
            out.write(json.dumps({"id": r["id"], "answers": {k: to_dist(q, ans[k]) for k, q in qs.items()}, "ms": round(ms, 2)}) + "\n")
    print("wrote", a.out)


def baseline(a):
    test = rows(TEST)
    freq = collections.defaultdict(collections.Counter)
    if a.kind == "prior":
        for r in rows(glob.glob(TRAIN)[0]):
            for k, g in json.loads(r["gold"]).items():
                freq[(r["workflow"], k)][str(g["label"])] += 1
    with open(a.out, "w") as out:
        for r in test:
            ans = {}
            for k, q in json.loads(r["questions"]).items():
                keys = option_keys(q)
                c = freq[(r["workflow"], k)]
                n = sum(c.values())
                ans[k] = {x: (c[x] / n if a.kind == "prior" else 1 / len(keys)) for x in keys}
            out.write(json.dumps({"id": r["id"], "answers": ans, "ms": 0}) + "\n")
    print("wrote", a.out)


# ---------------------------------------------------------------- score

def score(path, test):
    pred = {}
    with open(path) as f:
        for line in f:
            p = json.loads(line)
            pred[p["id"]] = p
    D, ms, missing = [], [], 0
    for r in test:
        p = pred.get(r["id"])
        if p is None:
            missing += 1
            continue
        ms.append(p.get("ms", 0))
        qs, gold = json.loads(r["questions"]), json.loads(r["gold"])
        for k, q in qs.items():
            keys = option_keys(q)
            pv = [max(0.0, float(p["answers"][k].get(x, 0.0))) for x in keys]
            z = sum(pv) or 1.0
            pv = [x / z for x in pv]
            gv = [float(gold[k]["probabilities"].get(x, 0.0)) for x in keys]
            D.append(dict(q=(r["workflow"], k), wf=r["workflow"], t=q["type"], p=pv, g=gv, lab=keys.index(str(gold[k]["label"])),
                          gs=gold[k].get("score")))
    m = collections.defaultdict(list)
    for d in D:
        p, g = d["p"], d["g"]
        d["am"] = max(range(len(p)), key=lambda i: (p[i], -i))
        d["conf"] = p[d["am"]]
        vals = dict(acc=float(d["am"] == d["lab"]), soft=g[d["am"]],
                    kl=sum(b * math.log(b / max(a, EPS)) for a, b in zip(p, g) if b > 0),
                    tv=0.5 * sum(abs(a - b) for a, b in zip(p, g)), brier=sum((a - b) ** 2 for a, b in zip(p, g)))
        if d["t"] == "score":
            vals["mae"] = abs(sum(i * a for i, a in enumerate(p)) - d["gs"])
            vals["w1"] = float(abs(d["am"] - d["lab"]) <= 1)
        for key, v in vals.items():
            for group in ("all", d["t"], d["wf"]):
                m[(group, key)].append(v)
    res = collections.defaultdict(dict)
    for (group, key), v in m.items():
        res[group][key] = sum(v) / len(v)
    byq = collections.defaultdict(list)
    for d in D:
        byq[d["q"]].append(d)
    f1 = []
    for ds in byq.values():
        labels = sorted({d["am"] for d in ds} | {d["lab"] for d in ds})
        per = []
        for L in labels:
            tp = sum(d["am"] == L and d["lab"] == L for d in ds)
            fp = sum(d["am"] == L and d["lab"] != L for d in ds)
            fn = sum(d["am"] != L and d["lab"] == L for d in ds)
            per.append(2 * tp / (2 * tp + fp + fn) if tp else 0.0)
        f1.append(sum(per) / len(per))
    res["all"]["mf1"] = sum(f1) / len(f1)
    bins = collections.defaultdict(list)
    for d in D:
        bins[min(14, int(d["conf"] * 15))].append(d)
    res["all"]["ece"] = sum(len(b) / len(D) * abs(statistics.mean(x["conf"] for x in b) - statistics.mean(x["am"] == x["lab"] for x in b))
                            for b in bins.values())
    res["all"]["ms"] = statistics.median(ms) if ms else 0
    res["all"]["n"] = len(D)
    res["all"]["missing_cases"] = missing
    return res


def show(a):
    test = rows(TEST)
    cols = ["acc", "soft", "mf1", "kl", "tv", "brier", "ece", "mae", "w1", "ms"]
    fmt = lambda v, c: "-" if v is None else (f"{v:.0f}" if c == "ms" else f"{v:.3f}")
    print(f"{'model':40s} " + " ".join(f"{c:>6s}" for c in cols))
    results = {}
    for path in a.pred:
        res = score(path, test)
        results[path] = res
        name = os.path.basename(path).replace(".jsonl", "")
        print(f"{name:40s} " + " ".join(f"{fmt(res['all'].get(c), c):>6s}" for c in cols)
              + (f"   ({res['all']['n']} decisions, {res['all']['missing_cases']} cases missing)" if res["all"]["missing_cases"] else ""))
    print("-- card")
    for name, v in CARD:
        print(f"{name:40s} " + " ".join(f"{fmt(v.get(c), c):>6s}" for c in cols))
    for path, res in results.items():
        print(f"\n{os.path.basename(path)} by type / workflow:")
        for group in ("noul", "choice", "score", "agent_trace_observability", "customer_service", "invoice_processing", "security_incidents"):
            g = res.get(group, {})
            print(f"  {group:28s} acc {g.get('acc', 0):.3f}  kl {g.get('kl', 0):.3f}  brier {g.get('brier', 0):.3f}")
    if a.json:
        with open(a.json, "w") as f:
            json.dump(results, f, indent=1)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("predict")
    p.add_argument("--model", help="local StartLux-Decision directory (in-process)")
    p.add_argument("--endpoint", help="base URL of a /v1/systemone server instead of --model")
    p.add_argument("--model-name", default="StartLux-Decision", help="value of the request's model field (endpoint mode)")
    p.add_argument("--api-key-env", default="SYSTEMONE_API_KEY", help="env var holding a bearer token (endpoint mode)")
    p.add_argument("--temperature", help="override every per-type temperature (in-process only)")
    p.add_argument("--out", required=True)
    b = sub.add_parser("baseline")
    b.add_argument("--kind", choices=["uniform", "prior"], required=True)
    b.add_argument("--out", required=True)
    s = sub.add_parser("score")
    s.add_argument("pred", nargs="+")
    s.add_argument("--json")
    a = ap.parse_args()
    if a.cmd == "predict" and not (a.model or a.endpoint):
        ap.error("--model or --endpoint is required")
    {"predict": predict, "baseline": baseline, "score": show}[a.cmd](a)


if __name__ == "__main__":
    main()
