"""A match between StartLux-Decision and Jev 1.13 under one condition: every start in starts.json (a book opening or a
Chess960 position) is played twice, each side White once, at the given state level, at most P games at a time, spread
over one or more StartLux-Decision servers.  A finished game is not replayed unless it had an API failure (the harness
then plays a random legal move), in which case it is replayed up to three times.

    python match.py --level rich --starts openings --out games/rich_openings --ours-urls http://127.0.0.1:8090 -P 32
    python annotate.py games/rich_openings/*.json
    python tally.py games/rich_openings
Environment as for duel.py (JEVBENCH, JEV_API_KEY).
"""
import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument("--level", required=True, choices=("rich", "tactical"))
ap.add_argument("--starts", required=True, choices=("openings", "chess960"))
ap.add_argument("--out", required=True)
ap.add_argument("--ours-urls", required=True, help="comma-separated StartLux-Decision servers")
ap.add_argument("-P", type=int, default=16)
a = ap.parse_args()
os.makedirs(f"{a.out}/logs", exist_ok=True)
starts = json.load(open(f"{HERE}/starts.json"))
urls = a.ours_urls.split(",")
if a.starts == "openings":
    jobs = [(f"o{i:03d}", ["--opening", " ".join(o["san"])]) for i, o in enumerate(starts["openings"])]
else:
    jobs = [(f"c{n:03d}", ["--chess960", str(n)]) for n in starts["chess960"]]
jobs = [(f"{tag}_{w}W", w, extra, urls[(2 * i + (w == "jev")) % len(urls)]) for i, (tag, extra) in enumerate(jobs) for w in ("ours", "jev")]


def failures(path):
    return sum(1 for m in json.load(open(path))["moves"] if m.get("fallback_random") or m.get("error"))


def play(job):
    tag, white, extra, url = job
    path = f"{a.out}/{tag}.json"
    for _ in range(3):
        if os.path.exists(path) and failures(path) == 0:
            return tag, "ok"
        with open(f"{a.out}/logs/{tag}.log", "a") as log:
            subprocess.run([sys.executable, f"{HERE}/duel.py", "--white", white, "--level", a.level, *extra, "--no-annotate",
                            "--max-plies", "160", "--ours-url", url, "--out", path],
                           stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
    return tag, "ok" if os.path.exists(path) and failures(path) == 0 else "FAILED"


with ThreadPoolExecutor(a.P) as ex:
    for tag, status in ex.map(play, jobs):
        print(tag, status, flush=True)
