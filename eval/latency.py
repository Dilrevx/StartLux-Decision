"""End-to-end latency of a /v1/systemone server, measured the way Intern-Decision reports it: one request holds one
choice, one yes/no and one score field (about 289 tokens of request content), requests are sent one at a time,
20 warm-up requests, then N timed ones.

    python -m startlux_decision.server --model /path/to/Startlux-Decision-4B --port 8090 &
    python eval/latency.py http://127.0.0.1:8090/v1/systemone 200            # three fields
    python eval/latency.py http://127.0.0.1:8090/v1/systemone 200 single     # one yes/no field
"""
import json, statistics, sys, time, urllib.request
url, n = sys.argv[1], int(sys.argv[2])
single = len(sys.argv) > 3
state = {"ticket": {"id": "T-48213", "channel": "email", "customer_tier": "business",
                    "subject": "Charged twice for renewal, dashboard still shows the account as suspended",
                    "body": ("Hello, our annual renewal for the Business plan went through on the 3rd, and the card statement shows "
                             "two identical charges of 1,140 USD from you on the same day. Since then the admin dashboard still says "
                             "the workspace is suspended for non-payment, so eleven people on my team cannot log in to the reporting "
                             "module and we have a board meeting tomorrow morning. I already tried logging out and clearing the cache, "
                             "and the status page says all systems are operational. Please refund the duplicate charge and restore "
                             "access as soon as possible. Invoice numbers INV-2291 and INV-2292 are attached.")},
         "account": {"plan": "Business annual", "seats": 11, "open_incidents": 0, "previous_tickets_90d": 1}}
questions = {"team": {"type": "choice", "instructions": "Which team should own this ticket?",
                      "criteria": {"billing": "Payments, refunds and invoices", "access": "Login, suspension and permissions",
                                   "technical": "Bugs and outages", "sales": "Plans and upgrades"}},
             "urgent": {"type": "noul", "instructions": "Does this ticket need a reply within one hour?"},
             "severity": {"type": "score", "instructions": "How severe is the customer impact?",
                          "criteria": ["cosmetic", "minor inconvenience", "blocks part of the team", "blocks the whole business"]}}
if single:
    questions = {"urgent": questions["urgent"]}
body = json.dumps({"model": "Startlux-Decision", "state": state, "questions": questions}).encode()
def call():
    t = time.perf_counter()
    with urllib.request.urlopen(urllib.request.Request(url, body, {"Content-Type": "application/json"}), timeout=60) as r:
        out = json.loads(r.read())
    return (time.perf_counter() - t) * 1000, out
for _ in range(20):
    call()
lat, last = [], None
for _ in range(n):
    ms, last = call()
    lat.append(ms)
lat.sort()
p = lambda q: lat[min(len(lat) - 1, int(q * len(lat)))]
print(json.dumps({"url": url, "questions": len(questions), "n": n, "mean_ms": round(statistics.mean(lat), 2), "p50_ms": round(p(0.5), 2),
                  "p95_ms": round(p(0.95), 2), "max_ms": round(lat[-1], 2), "decisions_per_s": round(1000 / statistics.mean(lat), 1),
                  "server_input_tokens": last["usage"]["input_tokens"], "server_ms": last.get("latency_ms"),
                  "answers": {k: v.get("choice", v.get("noul", v.get("score"))) for k, v in last["answers"].items()}}))
