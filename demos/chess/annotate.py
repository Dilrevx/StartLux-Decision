"""Add the Stockfish columns to duel games played with --no-annotate: for every move the centipawn loss against the best
move (depth 10 over all legal moves, as the harness does), its rank, Stockfish's choice and the evaluation after it
(depth 12); then the per-side summary.        python annotate.py GAME.json ...   (one process per game)"""
import json
import os
import statistics
import sys
from concurrent.futures import ProcessPoolExecutor

import chess

sys.path.insert(0, os.environ.get("JEVBENCH", "jev-benchmark"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from jevchess.engine import cp_loss_table, eval_cp, open_engine  # noqa: E402
from jevchess.experiments import _rank_of, ground_truth  # noqa: E402
from duel import summary  # noqa: E402


def one(path):
    g = json.load(open(path))
    if g.get("summary"):
        return path, "already annotated"
    board = chess.Board.from_chess960_pos(g["chess960"]) if g.get("chess960") is not None else chess.Board()
    for san in g.get("opening") or []:                 # book moves are not annotated
        board.push_san(san)
    with open_engine(threads=1, hash_mb=64) as eng:
        for m in g["moves"]:
            gt = ground_truth(eng, board, depth=10)
            losses = cp_loss_table(gt)
            best = min(losses, key=losses.get)
            m.update(cp_loss=losses[m["uci"]], rank=_rank_of(gt, m["uci"]), best_san=board.san(chess.Move.from_uci(best)),
                     n_legal=len(losses))
            board.push(chess.Move.from_uci(m["uci"]))
            m["eval_cp_white_after"] = eval_cp(eng, board, 12) if not board.is_game_over() else None
    g["summary"] = {s: summary(g["moves"], s) for s in ("ours", "jev")}
    json.dump(g, open(path, "w"), indent=1)
    return path, "ok"


if __name__ == "__main__":
    with ProcessPoolExecutor(min(len(sys.argv) - 1, 40)) as pool:
        for path, status in pool.map(one, sys.argv[1:]):
            print(status, path, flush=True)
