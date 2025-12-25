import chess
import random

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

TT = {}

class TTEntry:
    def __init__(self, depth, eval, flag, move):
        self.depth = depth
        self.eval = eval
        self.flag = flag  # "EXACT", "LOWERBOUND", "UPPERBOUND"
        self.move = move


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

def compute_child_hash(board, move, parent_hash):
    # simple, correct: push the move and recompute full zobrist from board
    board.push(move)
    new_hash = compute_zobrist(board)
    board.pop()
    return new_hash
