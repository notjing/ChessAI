import chess

def piece_value(piece):
    if not piece:
        return 0
    values = {
        chess.PAWN: 1,
        chess.KNIGHT: 3.2,
        chess.BISHOP: 3.3,
        chess.ROOK: 5,
        chess.QUEEN: 9,
        chess.KING: 200
    }
    return values[piece.piece_type]

# Central pawns are encouraged (e4/d4/c4/f4)
PAWN_PST = [
    0,  0,  0,  0,  0,  0,  0,  0,
    5, 10, 10, -20, -20, 10, 10, 5,   # Rank 2
    5, -5, -10,  0,  0, -10, -5, 5,   # Rank 3
    0,  0,  0, 20, 20,  0,  0,  0,    # Rank 4
    5,  5, 10, 25, 25, 10,  5,  5,    # Rank 5
    10, 10, 20, 30, 30, 20, 10, 10,   # Rank 6
    50, 50, 50, 50, 50, 50, 50, 50,   # Rank 7
    0,  0,  0,  0,  0,  0,  0,  0     # Rank 8
]

# Knights want to be in the center (d4, e4, c3, f3)
KNIGHT_PST = [
    -50,-40,-30,-30,-30,-30,-40,-50,
    -40,-20,  0,  5,  5,  0,-20,-40,
    -30,  5, 10, 15, 15, 10,  5,-30,
    -30,  0, 15, 20, 20, 15,  0,-30,
    -30,  5, 15, 20, 20, 15,  5,-30,
    -30,  0, 10, 15, 15, 10,  0,-30,
    -40,-20,  0,  0,  0,  0,-20,-40,
    -50,-40,-30,-30,-30,-30,-40,-50
]

# Bishops like diagonals
BISHOP_PST = [
    -20,-10,-10,-10,-10,-10,-10,-20,
    -10,  5,  0,  0,  0,  0,  5,-10,
    -10, 10, 10, 10, 10, 10, 10,-10,
    -10,  0, 10, 10, 10, 10,  0,-10,
    -10,  5,  5, 10, 10,  5,  5,-10,
    -10,  0,  5, 10, 10,  5,  0,-10,
    -10,  0,  0,  0,  0,  0,  0,-10,
    -20,-10,-10,-10,-10,-10,-10,-20
]

# Rooks like open files (simplified here to just 7th rank bonus)
ROOK_PST = [
    0,  0,  0,  5,  5,  0,  0,  0,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    5, 10, 10, 10, 10, 10, 10,  5,
    0,  0,  0,  0,  0,  0,  0,  0
]

# Queen is generally safe everywhere, slightly better in center
QUEEN_PST = [
    -20,-10,-10, -5, -5,-10,-10,-20,
    -10,  0,  0,  0,  0,  0,  0,-10,
    -10,  5,  5,  5,  5,  5,  0,-10,
     0,  0,  5,  5,  5,  5,  0, -5,
    -5,  0,  5,  5,  5,  5,  0, -5,
    -10,  0,  5,  5,  5,  5,  0,-10,
    -10,  0,  0,  0,  0,  0,  0,-10,
    -20,-10,-10, -5, -5,-10,-10,-20
]

# King safety (Stay in g1/c1/g8/c8). BIG penalty for e2/d2.
KING_PST = [
     20, 30, 10,  0,  0, 10, 30, 20,
     20, 20,  0,  0,  0,  0, 20, 20,
    -10,-20,-20,-20,-20,-20,-20,-10,
    -20,-30,-30,-40,-40,-30,-30,-20,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30
]

PST_MAP = {
    chess.PAWN: PAWN_PST,
    chess.KNIGHT: KNIGHT_PST,
    chess.BISHOP: BISHOP_PST,
    chess.ROOK: ROOK_PST,
    chess.QUEEN: QUEEN_PST,
    chess.KING: KING_PST,
}

def get_heuristic_policy(board):
    """
    Generates 'fake' probabilities since your model doesn't have a Policy Head.
    Captures/Checks = High Probability. Quiet moves = Low.
    """
    moves = list(board.legal_moves)
    scores = []

    is_white = (board.turn == chess.WHITE)

    for move in moves:
        score = 1.0
        piece = board.piece_type_at(move.from_square)
        PST = PST_MAP[piece]

        if board.is_capture(move):
            captured = board.piece_at(move.to_square)
            attacker_value = piece_value(board.piece_at(move.from_square))
            victim_value = piece_value(captured)

            score += 15 + victim_value * 5 - attacker_value

        if board.gives_check(move): score += 15.0
        if move.promotion: score += 100.0
        if board.is_castling(move): score += 25.0

        from_sq = move.from_square
        to_sq = move.to_square

        if not is_white:
            from_sq = from_sq ^ 56
            to_sq = to_sq ^ 56

        score += (PST[to_sq] - PST[from_sq]) / 5

        scores.append(max(1.0, score + 5))

    # Normalize scores to sum to 1.0 (like a probability distribution)
    total = sum(scores) if scores else 1
    probs = [s / total for s in scores]

    return zip(moves, probs)
