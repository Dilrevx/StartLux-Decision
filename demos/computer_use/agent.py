"""Computer-use run: a real Chrome window on a local mock site, driven by a /v1/systemone model.

Every step sends one request with two typed questions: which visible element to use next (choice over the page's
visible controls, plus scrolling) and whether the task is already done (yes/no).  The harness executes the chosen
element and records a screenshot, the element boxes and the full answer.  Nothing is scripted and no text is typed:
text boxes only get focus, which opens their suggestion lists.

    python demos/computer_use/agent.py store http://127.0.0.1:8090/v1/systemone runs/store
    python demos/computer_use/agent.py workspace http://127.0.0.1:8090/v1/systemone runs/workspace

Needs `pip install playwright` and a local Google Chrome (or `playwright install chromium` and drop channel="chrome").
"""
import argparse
import http.server
import json
import os
import socketserver
import threading
import time
import urllib.request
from functools import partial

from playwright.sync_api import sync_playwright

HERE = os.path.dirname(os.path.abspath(__file__))
CASES = {
    "store": {"root": "sites/store", "start": "index.html", "host": "kestrel-market.test",
             "task": "Order the cheapest 8-pack of AA batteries that has free delivery, and ship it to my home address."},
    "workspace": {"root": "sites/workspace", "start": "index.html", "host": "app.tandem.test",
               "task": "Invite dana@harborline.example to the Design team as an Editor."},
}
VIEW = {"width": 1120, "height": 700}

COLLECT = r"""
() => {
  const sel = 'a[href], button, input, select, textarea, [role=button], [role=link], [role=tab]';
  const clean = s => (s || '').replace(/\s+/g, ' ').trim();
  const out = [];
  document.querySelectorAll('[data-agent]').forEach(e => e.removeAttribute('data-agent'));
  let k = 0;
  for (const el of document.querySelectorAll(sel)) {
    const r = el.getBoundingClientRect();
    if (r.width < 2 || r.height < 2) continue;
    if (r.bottom <= 0 || r.top >= innerHeight || r.right <= 0 || r.left >= innerWidth) continue;
    const st = getComputedStyle(el);
    if (st.visibility === 'hidden' || st.display === 'none') continue;
    const cx = Math.min(Math.max(r.left + r.width / 2, 1), innerWidth - 2);
    const cy = Math.min(Math.max(r.top + r.height / 2, 1), innerHeight - 2);
    const hit = document.elementFromPoint(cx, cy);
    const lab = el.closest('label');
    if (!hit || !(hit === el || el.contains(hit) || (lab && lab.contains(hit)))) continue;
    const tag = el.tagName.toLowerCase(), type = (el.getAttribute('type') || '').toLowerCase();
    const ctx = el.closest('.item, .card, .trow, tr');
    let context = '';
    if (ctx) {
      context = clean(ctx.innerText).replace(clean(el.innerText), '').trim();
      if (context.length > 90) context = context.slice(0, 90);
    }
    const base = {box: [r.left, r.top, r.width, r.height]};
    if (lab && (type === 'checkbox' || type === 'radio')) {
      const q = lab.getBoundingClientRect();
      base.hbox = [q.left, q.top, q.width, q.height];
    }
    if (tag === 'select') {
      const name = clean(el.getAttribute('aria-label') || (lab && lab.innerText) || el.name);
      for (const o of el.options) {
        const id = 'e' + (++k);
        el.setAttribute('data-agent', (el.getAttribute('data-agent') || '') + ' ' + id);
        out.push({...base, id, kind: 'option', value: o.value,
                  text: `option "${clean(o.text)}" in "${name}"` + (o.selected ? ' (selected)' : '')});
      }
      continue;
    }
    const id = 'e' + (++k);
    el.setAttribute('data-agent', id);
    let text;
    if (tag === 'input' && (type === 'checkbox' || type === 'radio')) {
      text = `${type} "${clean(lab ? lab.innerText : el.value)}"` + (el.checked ? ' (selected)' : ' (not selected)');
    } else if (tag === 'input' || tag === 'textarea') {
      text = `text box "${clean(el.getAttribute('aria-label') || el.placeholder)}"` + (el.value ? ` containing "${clean(el.value)}"` : '')
        + (el === document.activeElement ? ' (focused' + (el.getAttribute('aria-expanded') === 'true' ? ', suggestions open' : '') + ')' : '');
    } else {
      const role = tag === 'a' ? 'link' : 'button';
      const name = clean(el.getAttribute('aria-label') || el.innerText);
      text = `${role} "${name}"` + (el.getAttribute('aria-pressed') === 'true' ? ' (selected)' : '') + (context ? ` (${context})` : '');
    }
    const typing = (tag === 'input' && type !== 'checkbox' && type !== 'radio') || tag === 'textarea';
    out.push({...base, id, kind: typing ? 'focus' : 'click', text});
  }
  const more = document.scrollingElement.scrollHeight - innerHeight - scrollY > 8;
  const main = document.querySelector('.modal.open .dlg') || document.querySelector('main, .content') || document.body;
  let text = clean(main.innerText);
  const toast = document.querySelector('.toast.show');
  if (toast) text = clean(toast.innerText) + ' | ' + text;
  return {elements: out, more, up: scrollY > 8, title: document.title, text: text.slice(0, 700)};
}
"""


def serve(root):
    class Quiet(http.server.SimpleHTTPRequestHandler):
        def log_message(self, *args):
            pass
    handler = partial(Quiet, directory=os.path.join(HERE, root))
    srv = socketserver.TCPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, srv.server_address[1]


def ask(endpoint, state, criteria):
    body = {"state": state, "questions": {
        "action": {"type": "choice", "instructions": "Which page element should be interacted with next to make progress on the task?",
                   "criteria": criteria},
        "done": {"type": "noul", "instructions": "Has the task already been completed on this page?"}}}
    t = time.time()
    req = urllib.request.Request(endpoint, json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        resp = json.loads(r.read())
    resp["wall_ms"] = round((time.time() - t) * 1000, 1)
    return resp


def check(case, page):
    if case == "store":
        o = page.evaluate("() => JSON.parse(localStorage.getItem('order') || 'null')")
        if not o:
            return None
        ok = o["items"] == [{"id": "p5", "qty": 1}] and o["address"] == "home"
        return {"order": o, "matches_task": ok}
    inv = page.evaluate("() => JSON.parse(localStorage.getItem('invites') || '[]')")
    if not inv:
        return None
    ok = any(x["team"] == "design" and x["email"] == "dana@harborline.example" and x["role"] == "Editor" for x in inv)
    return {"invites": inv, "matches_task": ok}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("case", choices=sorted(CASES))
    ap.add_argument("endpoint")
    ap.add_argument("out")
    ap.add_argument("--steps", type=int, default=14)
    a = ap.parse_args()
    case = CASES[a.case]
    os.makedirs(a.out, exist_ok=True)
    srv, port = serve(case["root"])
    log = {"case": a.case, "task": case["task"], "endpoint": a.endpoint, "viewport": VIEW, "steps": []}
    history = []
    with sync_playwright() as p:
        browser = p.chromium.launch(channel="chrome", headless=True)
        page = browser.new_page(viewport=VIEW, device_scale_factor=2)
        page.goto(f"http://127.0.0.1:{port}/{case['start']}")
        page.evaluate("() => localStorage.clear()")
        page.reload()
        for step in range(a.steps):
            page.wait_for_timeout(250)
            info = page.evaluate(COLLECT)
            els = info["elements"]
            if info["more"]:
                els.append({"id": "scroll_down", "kind": "scroll", "text": "scroll down the page", "box": None})
            if info["up"]:
                els.append({"id": "scroll_up", "kind": "scroll", "text": "scroll up the page", "box": None})
            url = page.url.split(f"127.0.0.1:{port}", 1)[1]
            state = json.dumps({"task": case["task"], "website": case["host"], "url": case["host"] + url,
                                "page_title": info["title"], "page_text": info["text"], "previous_actions": history[-5:],
                                "elements": [{"id": e["id"], "text": e["text"]} for e in els]}, ensure_ascii=False)
            shot = os.path.join(a.out, f"step{step:02d}.png")
            page.screenshot(path=shot)
            resp = ask(a.endpoint, state, {e["id"]: e["text"] for e in els})
            ans = resp["answers"]
            pick = ans["action"]["choice"]
            el = next(e for e in els if e["id"] == pick)
            rec = {"step": step, "url": case["host"] + url, "screenshot": os.path.basename(shot), "elements": els,
                   "probabilities": ans["action"]["probabilities"], "choice": pick, "done": ans["done"]["noul"],
                   "latency_ms": resp.get("latency_ms"), "wall_ms": resp["wall_ms"],
                   "input_tokens": resp.get("usage", {}).get("input_tokens")}
            log["steps"].append(rec)
            print(f"step {step}: {len(els)} options, done {ans['done']['noul']:.2f}, pick {pick} "
                  f"({ans['action']['probabilities'][pick]:.2f}) {el['text'][:90]}", flush=True)
            if ans["done"]["noul"] >= 0.5:
                rec["stopped"] = True
                break
            if el["kind"] == "scroll":
                page.evaluate(f"() => window.scrollBy(0, {'' if pick == 'scroll_down' else '-'}innerHeight * 0.7)")
                history.append("scrolled " + ("down" if pick == "scroll_down" else "up"))
            elif el["kind"] == "option":
                sel = page.locator(f"select[data-agent~='{pick}']")
                with page.expect_navigation(wait_until="load", timeout=5000):
                    sel.select_option(el["value"])
                history.append(el["text"].split(" (selected)")[0].replace("option ", "selected ", 1))
            else:
                loc = page.locator(f"[data-agent='{pick}']")
                try:
                    with page.expect_navigation(wait_until="load", timeout=1500):
                        loc.click()
                except Exception:  # noqa: BLE001  (clicks that do not navigate)
                    pass
                history.append(("focused " if el["kind"] == "focus" else "clicked ") + el["text"].split(" (")[0])
        final = os.path.join(a.out, "final.png")
        page.wait_for_timeout(300)
        page.screenshot(path=final)
        log["final_screenshot"] = "final.png"
        log["final_url"] = page.url.split(f"127.0.0.1:{port}", 1)[1]
        log["result"] = check(a.case, page)
        browser.close()
    srv.shutdown()
    json.dump(log, open(os.path.join(a.out, "run.json"), "w"), indent=1)
    print(json.dumps(log["result"]))


if __name__ == "__main__":
    main()
