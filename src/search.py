import sys
import chess
import chess.engine
import random
import time
import evaluate as model
from ChessAI.src.move_ordering import score_move

search_deadline = None


# -----------------------------
# Zobrist Hash Setup
# -----------------------------
ZOBRIST_PIECE = [[random.getrandbits(64) for _ in range(64)] for _ in range(12)]
ZOBRIST_BLACK_TO_MOVE = random.getrandbits(64)

# Optional: castling and en passant
ZOBRIST_CASTLING = [random.getrandbits(64) for _ in range(16)]
ZOBRIST_EN_PASSANT = [random.getrandbits(64) for _ in range(8)]

def piece_index(piece):
    base = {
        chess.PAWN: 0,
        chess.KNIGHT: 1,
        chess.BISHOP: 2,
        chess.ROOK: 3,
        chess.QUEEN: 4,
        chess.KING: 5
    }[piece.piece_type]
    return base + (0 if piece.color == chess.WHITE else 6)

def compute_zobrist(board):
    h = 0
    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece:
            idx = piece_index(piece)
            h ^= ZOBRIST_PIECE[idx][sq]
    if board.turn == chess.BLACK:
        h ^= ZOBRIST_BLACK_TO_MOVE

    # Castling rights
    castling = 0
    if board.has_kingside_castling_rights(chess.WHITE):
        castling |= 1
    if board.has_queenside_castling_rights(chess.WHITE):
        castling |= 2
    if board.has_kingside_castling_rights(chess.BLACK):
        castling |= 4
    if board.has_queenside_castling_rights(chess.BLACK):
        castling |= 8
    h ^= ZOBRIST_CASTLING[castling]

    # En passant
    if board.ep_square is not None:
        file = chess.square_file(board.ep_square)
        h ^= ZOBRIST_EN_PASSANT[file]

    return h

# -----------------------------
# Transposition Table
# -----------------------------
TT = {}

class TTEntry:
    def __init__(self, depth, eval, flag, move):
        self.depth = depth
        self.eval = eval
        self.flag = flag  # "EXACT", "LOWERBOUND", "UPPERBOUND"
        self.move = move

nodes = 0
leaf_nodes = 0

real_eval=True

def evaluate(board):
    global leaf_nodes
    leaf_nodes += 1
    engine_start = time.time()
    if real_eval:

        #info = engine.analyse(board, chess.engine.Limit(time=STOCKFISH_TIME))
        val = model.evaluate_board(board)
        #val = info["score"].white().score(mate_score=1000000)

        return val
    else:
        return 0
    engine_end = time.time()
    print(str(engine_end - engine_start))

    #time.sleep(0.001)  # 5 ms
    #return 0

# -----------------------------
# Global to store PV from previous depth
# -----------------------------
pv_move = None


def compute_child_hash(board, move, parent_hash):
    # simple, correct: push the move and recompute full zobrist from board
    board.push(move)
    new_hash = compute_zobrist(board)
    board.pop()
    return new_hash


def piece_value(piece):
    if not piece:
        return 0
    values = {
        chess.PAWN: 100,
        chess.KNIGHT: 320,
        chess.BISHOP: 330,
        chess.ROOK: 500,
        chess.QUEEN: 900,
        chess.KING: 20000
    }
    return values[piece.piece_type]

def search(board, depth, alpha, beta, zobrist_hash):
    global nodes, pv_move
    nodes += 1

    # Check for cancellation
    if search_deadline is not None and time.time() >= search_deadline:
        print("TIME IS UP")
        return None, None

    # Transposition table lookup
    entry = TT.get(zobrist_hash)
    if entry and entry.depth >= depth:
        if entry.flag == "EXACT":
            return entry.eval, entry.move
        elif entry.flag == "LOWERBOUND":
            alpha = max(alpha, entry.eval)
        elif entry.flag == "UPPERBOUND":
            beta = min(beta, entry.eval)
        if alpha >= beta:
            return entry.eval, entry.move

    if depth == 0 or board.is_game_over():
        val = evaluate(board)
        return val, None

    best_move = None
    maximizing = board.turn
    best_eval = -1e9 if maximizing else 1e9

    moves = list(board.legal_moves)
    moves.sort(key=lambda m: score_move(board, m, pv_move, zobrist_hash), reverse=True)

    orig_alpha = alpha
    orig_beta = beta

    for move_index, move in enumerate(moves):
        new_hash = compute_child_hash(board, move, zobrist_hash)

        # Late Move Reductions
        reduction = 0
        if (
            depth >= 3
            and move_index >= 3
            and not board.is_capture(move)
            and not board.gives_check(move)
            and move.promotion is None
            and not board.is_check()
        ):
            reduction = 1

        board.push(move)
        if reduction > 0:
            # first search on a small alpha-beta window (can this move do better/worse than alpha/beta)
            if maximizing:
                val, mov = search(board, depth - 1 - reduction, alpha, alpha + 1, new_hash)
            else:
                val, mov = search(board, depth - 1 - reduction, beta - 1, beta, new_hash)

            # not clear if move is bad or good --> research
            if val is not None and alpha < val < beta:
                val, mov = search(board, depth - 1, alpha, beta, new_hash)
        else:
            val, mov = search(board, depth - 1, alpha, beta, new_hash)

        board.pop()

        # Propagate cancellation
        if val is None:
            return None, None

        if maximizing:
            if val > best_eval:
                best_eval = val
                best_move = move
            alpha = max(alpha, val)
        else:
            if val < best_eval:
                best_eval = val
                best_move = move
            beta = min(beta, val)

        if alpha >= beta:
            break

    # Store in TT
    flag = "EXACT"
    if best_eval <= orig_alpha:
        flag = "UPPERBOUND"
    elif best_eval >= orig_beta:
        flag = "LOWERBOUND"
    TT[zobrist_hash] = TTEntry(depth, best_eval, flag, best_move)

    # Update PV
    if depth == 1:
        pv_move = best_move

    return best_eval, best_move


def find_move(board, max_depth):
    global pv_move, search_deadline
    search_deadline = time.time() + 10.0
    #if ctx.timeLeft<10000:
    #    search_deadline = time.time() + 0.5
    #elif ctx.timeLeft<5000:
    #    search_deadline = time.time() + 0.01
    zob_hash = compute_zobrist(board)
    best_eval, best_move = None, None

    pv_move = None

    time_start = time.time()
    #if ctx.timeLeft<30000:
    #    max_depth -=1
    #if ctx.timeLeft<20000:
    #    max_depth -=1

    for depth in range(1, max_depth + 1):
        print(search_deadline - time.time())
        print(f"\n=== Starting depth {depth} ===")
        eval, move = search(board, depth, float('-inf'), float('inf'), zob_hash)

        if eval is None:
            print("Search cancelled due to timeout")
            break

        best_eval, best_move = eval, move
        pv_move = best_move

        print(f"Depth {depth} finished. Best move found: {best_move}\n")

    time_end = time.time()
    elapsed = time_end - time_start
    nps = nodes / elapsed

    print("===================================")
    print(f"Depth: {depth}")
    print(f"Best Move: {best_move}")
    print(f"Eval: {best_eval}")
    print(f"Nodes: {nodes:,}")
    print(f"Leaf Nodes: {leaf_nodes:,}")
    print(f"Time: {elapsed:.4f}s")
    print(f"NPS: {nps:,.0f}")
    print("===================================\n")

    return best_eval, best_move

board = chess.Board("rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 0 1")
print(evaluate(board))
