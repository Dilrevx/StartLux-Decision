"""Score of one match directory (every *_oursW.json / *_jevW.json game): W-D-L, points, Elo difference with a 95%
bootstrap interval over starts (both games of a start stay together), likelihood of superiority, colour split, how
games ended and per-move quality from Stockfish (annotate.py).        python tally.py DIR [DIR ...] > OUT.json"""
import collections
import glob
import json
import math
import os
import random
import sys

CAP = 1000


def elo(s):
    s = min(max(s, 1e-9), 1 - 1e-9)
    return -400 * math.log10(1 / s - 1)


def quality(moves):
    losses = [min(m["cp_loss"], CAP) for m in moves if m.get("cp_loss") is not None]
    n = len(losses)
    return {"moves": n, "mean_cp_loss": round(sum(losses) / n, 1), "best_move_pct": round(100 * sum(l == 0 for l in losses) / n, 1),
            "blunders_200cp_per_100_moves": round(100 * sum(l >= 200 for l in losses) / n, 2)} if n else {}


def tally(d):
    games = [json.load(open(p)) | {"_tag": os.path.basename(p)[:-5]} for p in sorted(glob.glob(f"{d}/*_*W.json"))]
    rows, by_start = [], collections.defaultdict(list)
    for g in games:
        final = g["adjudicated_result"] or g["result"]
        ours_white = g["white"]["id"] == "ours"
        pts = 0.5 if final == "1/2-1/2" else (1.0 if (final == "1-0") == ours_white else 0.0)
        rows.append({"pts": pts, "ours_white": ours_white, "termination": g["termination"], "adjudicated": g["adjudicated_result"] is not None,
                     "fails": sum(1 for m in g["moves"] if m.get("fallback_random") or m.get("error"))})
        by_start[g["_tag"].rsplit("_", 1)[0]].append(pts)
    n = len(rows)
    W, D, L = (sum(r["pts"] == v for r in rows) for v in (1.0, 0.5, 0.0))
    score = (W + 0.5 * D) / n
    rng, keys, boots = random.Random(0), list(by_start), []
    for _ in range(20000):
        pts = [p for k in (rng.choice(keys) for _ in keys) for p in by_start[k]]
        boots.append(sum(pts) / len(pts))
    boots.sort()
    lo, hi = boots[int(0.025 * len(boots))], boots[int(0.975 * len(boots))]
    return {"dir": d, "level": games[0]["level"], "start": "chess960" if games[0].get("chess960") is not None else "book openings",
            "games": n, "starts": len(by_start), "wins": W, "draws": D, "losses": L, "points": W + 0.5 * D, "score": round(score, 4),
            "elo_diff": round(elo(score)), "elo_diff_95": [round(elo(lo)), round(elo(hi))],
            "los": round(0.5 * (1 + math.erf((W - L) / math.sqrt(2 * (W + L)))), 4) if W + L else 0.5,
            "as_white": sum(r["pts"] for r in rows if r["ours_white"]), "as_black": sum(r["pts"] for r in rows if not r["ours_white"]),
            "checkmates_for": sum(r["termination"] == "CHECKMATE" and r["pts"] == 1 for r in rows),
            "checkmates_against": sum(r["termination"] == "CHECKMATE" and r["pts"] == 0 for r in rows),
            "adjudicated": sum(r["adjudicated"] for r in rows),
            "draw_kinds": dict(collections.Counter(r["termination"] for r in rows if r["pts"] == 0.5 and not r["adjudicated"])),
            "api_failures": sum(r["fails"] for r in rows),
            "quality": {s: quality([m for g in games for m in g["moves"] if m["side"] == s]) for s in ("ours", "jev")},
            "versions": dict(collections.Counter(json.dumps(g.get("model_versions"), sort_keys=True) for g in games))}


print(json.dumps([tally(d) for d in sys.argv[1:]], indent=1))
