import chess
import chess.engine
import random
import time
from src import evaluate as model

# -----------------------------
# Zobrist Hash Setup
# -----------------------------
ZOBRIST_PIECE = [[random.getrandbits(64) for _ in range(64)] for _ in range(12)]
ZOBRIST_BLACK_TO_MOVE = random.getrandbits(64)
# Optional: castling and en passant
ZOBRIST_CASTLING = [random.getrandbits(64) for _ in range(16)]
ZOBRIST_EN_PASSANT = [random.getrandbits(64) for _ in range(8)]

def piece_index(piece):
    """Map python-chess piece to 0..11"""
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

engine = chess.engine.SimpleEngine.popen_uci(
    r"C:\Users\itish\Documents\ChessHacks\stockfish\stockfish-windows-x86-64-avx2.exe"
)
try:
    engine.configure({"Threads": 1})
except Exception(BaseException):
    pass

STOCKFISH_TIME = 0.001
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

def score_move(board, move, parent_hash=None):
    score = 0

    # PV move first
    if move == pv_move:
        return 1000000

    # Check bonus
    board.push(move)
    if board.is_check():
        score += 1000

    #prioritize castling
    if board.is_castling(move):
        score+=500
    board.pop()

    # Captures (MVV-LVA)
    if board.is_capture(move):
        captured = board.piece_at(move.to_square)
        attacker_value = piece_value(board.piece_at(move.from_square))
        victim_value = piece_value(captured)
        score += 1000 + min(0,victim_value - attacker_value)

    # Promotions
    if move.promotion:
        score += 900


    # ----- TT ordering -----
    if parent_hash is not None:
        child_hash = compute_child_hash(board, move, parent_hash)
        entry = TT.get(child_hash)

        if entry:
            # EXACT nodes are strongest predictions
            if entry.flag == "EXACT":
                score += 3000 + entry.eval
            elif entry.flag == "LOWERBOUND":  # likely good
                score += 2000
            elif entry.flag == "UPPERBOUND":  # weaker info
                score += 1000

    return score


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

    # TT lookup
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

    # --- Move Ordering ---
    moves = list(board.legal_moves)
    moves.sort(key=lambda m: score_move(board, m, zobrist_hash), reverse=True)
    #print(moves)

    orig_alpha = alpha
    orig_beta = beta

    for move in moves:
        print(" " * depth, f"Exploring move: {move}, depth {depth}")
        new_hash = compute_child_hash(board, move, zobrist_hash)
        board.push(move)
        val, _ = search(board, depth - 1, alpha, beta, new_hash)
        board.pop()

        if maximizing:
            if val > best_eval:
                print(" " *depth,f"New best move at depth {depth}: {move} (score {val})")
                best_eval = val
                best_move = move
            alpha = max(alpha, val)
        else:
            if val < best_eval:
                print(" " * depth,f"New best move at depth {depth}: {move} (score {val})")
                best_eval = val
                best_move = move
            beta = min(beta, val)

        if alpha >= beta:
            #print("PRUNE!")
            break
    #print("Done!")


    # Store in TT
    flag = "EXACT"
    if best_eval <= orig_alpha:
        flag = "UPPERBOUND"
    elif best_eval >= orig_beta:
        flag = "LOWERBOUND"
    TT[zobrist_hash] = TTEntry(depth, best_eval, flag, best_move)

    # Update PV move for next iterative deepening
    if depth == 1:  # could also store PV from root
        pv_move = best_move

    return best_eval, best_move

#@chess_manager.entrypoint
def find_move(board, max_depth):
    global pv_move
    zob_hash = compute_zobrist(board)
    best_eval, best_move = None, None

    pv_move = None  # reset before iterative deepening

    time_start=time.time()
    for depth in range(1, max_depth + 1):
        print(f"\n=== Starting depth {depth} ===")
        best_eval, best_move = search(board, depth, float('-inf'), float('inf'), zob_hash)
        print(f"Depth {depth} finished. Best move found: {best_move}\n")
        pv_move = best_move  # save PV for next depth
    time_end=time.time()
    elapsed=time_end-time_start
    nps=nodes/elapsed

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


