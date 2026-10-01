"""StartLux-Decision and Jev 1.13 play one game of chess against each other in the wondertwins/jev-benchmark harness.

Both sides get exactly the request the harness sends in its own games at the chosen level ("rich": the board, the move
history, piece lists and facts for the side to move, and every legal move; "tactical" adds one-ply facts computed by
code for every move). Each side answers one choice question and its most likely legal move is played: no search, no
code overrides, no blunder filter. Stockfish only annotates (centipawn loss and the evaluation after each move), as in
the harness's ladder games. Games stop at checkmate, a draw the rules end by themselves, or 160 plies, where they are
adjudicated at +-300 centipawns like every ladder game.

    python duel.py --white ours|jev --out GAME.json [--level rich|tactical] [--opening "e4 e5 Nf3 Nc6" | --chess960 N]
        [--ours-url http://127.0.0.1:8090] [--no-annotate]

JEVBENCH: a checkout of wondertwins/jev-benchmark (its packages jevchess and typesafe_sdk, and Stockfish on PATH).
OURS_BASE_URL (or --ours-url): a StartLux-Decision server (python -m startlux_decision.server). JEV_API_KEY: a TypeSafe
API key for Jev. OURS_NAME and JEV_MODEL name the two players.
"""
import argparse
import asyncio
import json
import os
import random
import statistics
import sys
import time

import chess

sys.path.insert(0, os.environ.get("JEVBENCH", "jev-benchmark"))
from typesafe_sdk import AsyncTypeSafeClient, Choice, RetryPolicy  # noqa: E402
from jevchess.engine import cp_loss_table, eval_cp, open_engine  # noqa: E402
from jevchess.experiments import _rank_of, choice_instructions, ground_truth  # noqa: E402
from jevchess.state import build_state, legal_move_options, san_to_move  # noqa: E402

LABEL = {"ours": os.environ.get("OURS_NAME", "StartLux-Decision-27B"), "jev": "Jev 1.13"}


def summary(plies, side):
    own = [p for p in plies if p["side"] == side]
    losses = [p["cp_loss"] for p in own]
    return {"moves": len(own), "mean_cp_loss": round(statistics.mean(losses), 1) if losses else None,
            "best_move_pct": round(100 * sum(l == 0 for l in losses) / len(losses), 1) if losses else None,
            "blunders_200cp": sum(l >= 200 for l in losses), "fallback_random": sum(bool(p.get("fallback_random")) for p in own),
            "median_latency_s": round(statistics.median(p["latency_s"] for p in own), 3) if own else None}


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--white", choices=("ours", "jev"), required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-plies", type=int, default=160)
    ap.add_argument("--level", default="tactical", choices=("tactical", "rich"))
    ap.add_argument("--chess960", type=int, default=None, help="Chess960 start position number (0-959); default: the standard start")
    ap.add_argument("--ours-url", default=None, help="our /v1/systemone server root (default: OURS_BASE_URL)")
    ap.add_argument("--no-annotate", action="store_true", help="play only; annotate.py adds the Stockfish columns afterwards")
    ap.add_argument("--opening", default="", help="book moves in SAN, played before either side chooses (standard start)")
    a = ap.parse_args()
    clients = {
        "ours": AsyncTypeSafeClient(api_key="local", model=os.environ.get("OURS_MODEL", "startlux-decision"),
                                    base_url=a.ours_url or os.environ["OURS_BASE_URL"], timeout=120.0,
                                    retry=RetryPolicy(max_retries=3, backoff_initial=1.0, backoff_max=10.0, timeout=120.0)),
        "jev": AsyncTypeSafeClient(api_key=os.environ["JEV_API_KEY"], model=os.environ.get("JEV_MODEL", "jev-latest"), timeout=60.0,
                                   retry=RetryPolicy(max_retries=6, backoff_initial=1.0, backoff_max=20.0, timeout=90.0)),
    }
    white, black = a.white, ("jev" if a.white == "ours" else "ours")
    LEVEL = a.level
    board = chess.Board.from_chess960_pos(a.chess960) if a.chess960 is not None else chess.Board()
    start_fen = board.fen()
    sans, plies, versions = [], [], {}
    for san in a.opening.split():                      # book moves: part of the move history both sides see
        mv = board.parse_san(san)
        sans.append(board.san(mv))
        board.push(mv)
    opening = list(sans)
    rng = random.Random(3)
    async with clients["ours"], clients["jev"]:
        with open_engine() as eng:
            while not board.is_game_over() and len(sans) < a.max_plies:
                side = white if board.turn == chess.WHITE else black
                if not a.no_annotate:
                    gt = ground_truth(eng, board, depth=10)
                    losses = cp_loss_table(gt)
                opts = legal_move_options(board, LEVEL)
                rec = {"ply": len(sans), "side": side, "color": "white" if board.turn == chess.WHITE else "black",
                       "fen_before": board.fen(), "n_legal": board.legal_moves.count()}
                t0 = time.perf_counter()
                try:
                    resp = await clients[side].system_one(build_state(board, sans, LEVEL),
                                                          {"move": Choice(instructions=choice_instructions(board), criteria=opts)})
                    raw = resp.raw_http_response.json()
                    ans = raw["answers"]["move"]
                    versions.setdefault(side, raw.get("model") or raw.get("model_version"))
                except Exception as e:  # noqa: BLE001 - the harness plays a random legal move then; keep the record
                    ans = None
                    rec["error"] = repr(e)[:300]
                rec["latency_s"] = round(time.perf_counter() - t0, 3)
                mv = san_to_move(board, ans["choice"]) if ans else None
                if mv is None:
                    mv = rng.choice(list(board.legal_moves))
                    rec["fallback_random"] = True
                else:
                    rec["probabilities"] = ans.get("probabilities")
                    rec["confidence"] = ans.get("confidence")
                rec.update(san=board.san(mv), uci=mv.uci())
                if not a.no_annotate:
                    best = min(losses, key=losses.get)
                    rec.update(cp_loss=losses[mv.uci()], rank=_rank_of(gt, mv.uci()), best_san=board.san(chess.Move.from_uci(best)))
                sans.append(board.san(mv))
                board.push(mv)
                if not a.no_annotate:
                    rec["eval_cp_white_after"] = eval_cp(eng, board, 12) if not board.is_game_over() else None
                plies.append(rec)
                print(json.dumps({"ply": rec["ply"], "side": side, "san": rec["san"], "p": rec.get("confidence"),
                                  "cp_loss": rec.get("cp_loss"), "eval": rec.get("eval_cp_white_after"), "s": rec["latency_s"]}), flush=True)
            result = board.result(claim_draw=True) if board.is_game_over(claim_draw=True) else "unfinished"
            outcome = board.outcome(claim_draw=True)
            adjudicated = None
            if result == "unfinished":
                cp = eval_cp(eng, board, 12)
                adjudicated = "1-0" if cp > 300 else "0-1" if cp < -300 else "1/2-1/2"
    out = {"experiment": "duel", "level": LEVEL, "max_plies": a.max_plies, "chess960": a.chess960, "start_fen": start_fen,
           "opening": opening,
           "white": {"id": white, "label": LABEL[white]}, "black": {"id": black, "label": LABEL[black]},
           "model_versions": versions, "result": result, "adjudicated_result": adjudicated,
           "termination": outcome.termination.name if outcome else "ply_cap", "plies": len(sans),
           "pgn_moves": " ".join(f"{i // 2 + 1}. {s}" if i % 2 == 0 else s for i, s in enumerate(sans)),
           "final_fen": board.fen(), "summary": None if a.no_annotate else {s: summary(plies, s) for s in ("ours", "jev")}, "moves": plies}
    json.dump(out, open(a.out, "w"), indent=1)
    print(json.dumps({"result": result, "adjudicated": adjudicated, "termination": out["termination"], "plies": len(sans),
                      "summary": out["summary"]}), flush=True)


if __name__ == "__main__":
    asyncio.run(main())
