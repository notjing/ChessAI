import sys
import chess
import time
import traceback

import search   # your existing search code


# -------------------------
# Utilities
# -------------------------

def log(msg):
    """Debug logging → stderr (never stdout)"""
    print(msg, file=sys.stderr, flush=True)


# -------------------------
# Engine state
# -------------------------

board = chess.Board()
search_depth = 3


# -------------------------
# UCI main loop
# -------------------------

def main():
    global board, search_depth

    while True:
        try:
            line = sys.stdin.readline()
            if not line:
                continue

            line = line.strip()

            # --- UCI handshake ---
            if line == "uci":
                print("id name MarsIsRed", flush=True)
                print("id author RedIsMars", flush=True)
                print("uciok", flush=True)


            elif line == "isready":
                print("readyok", flush=True)

            # --- Set position ---
            elif line.startswith("position"):
                tokens = line.split()

                if "startpos" in tokens:
                    board = chess.Board()
                    moves_start = tokens.index("moves") + 1 if "moves" in tokens else None
                else:
                    fen_start = tokens.index("fen") + 1
                    fen = " ".join(tokens[fen_start:fen_start + 6])
                    board = chess.Board(fen)
                    moves_start = fen_start + 6 if "moves" in tokens else None

                if moves_start:
                    for move in tokens[moves_start:]:
                        board.push_uci(move)

            # --- Go / search ---
            elif line.startswith("go"):
                start = time.time()

                # Your existing search function
                eval_score, move = search.find_move(board, search_depth)

                if move is None or move not in board.legal_moves:
                    print("bestmove 0000", flush=True)
                else:
                    print(f"bestmove {move.uci()}", flush=True)

                log(f"Move computed in {time.time() - start:.3f}s")

            # --- Quit ---
            elif line == "quit":
                break

        except Exception:
            log(traceback.format_exc())


if __name__ == "__main__":
    main()
