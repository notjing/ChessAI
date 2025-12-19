import chess
import numpy as np

(4,2)
def get_mapped_coords(square, flip):
    """
    given a square, return the square with the correct perspective
    """
    rank = chess.square_rank(square)
    file = chess.square_file(square)

    # FEN always has white at the bottom and black at the top
    # so from black pov, you need to flip the files, the row will be correct when counting from the top
    # from white pov, the file will be correct, but your row counts from the top, not the bottom
    if flip:
        # to flip the board
        return rank, 7 - file
    else:
        return 7 - rank, file


def square_control(board):
    """
    returns 8x8 bitboards, highlighting how many times a square is being attacked
    """
    flip = (board.turn == chess.BLACK)
    own_control = np.zeros((8, 8), dtype=np.float32)
    opp_control = np.zeros((8, 8), dtype=np.float32)

    own_color = board.turn
    opp_color = not board.turn

    for square in chess.SQUARES:
        r, c = get_mapped_coords(square, flip)
        own_control[r][c] = len(board.attackers(own_color, square))
        opp_control[r][c] = len(board.attackers(opp_color, square))

    return own_control, opp_control


def makeboards(board):
    """
    Generates [my pieces] [their pieces] heatmaps indicating the location of each piece
    """
    flip = (board.turn == chess.BLACK)
    piece_types = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]

    layers = []
    for color in [board.turn, not board.turn]:
        for pt in piece_types:
            grid = np.zeros((8, 8), dtype=np.float32)
            for square in board.pieces(pt, color):
                r, c = get_mapped_coords(square, flip)
                grid[r][c] = 1
            layers.append(grid)
    return layers

def board_parameters(board):
    """
    Generates all the parameters needed for the dense network
    """
    # Castling
    rights = [
        board.has_kingside_castling_rights(chess.WHITE),
        board.has_queenside_castling_rights(chess.WHITE),
        board.has_kingside_castling_rights(chess.BLACK),
        board.has_queenside_castling_rights(chess.BLACK)
    ]
    castling = [1.0 if r else 0.0 for r in rights]

    # Number of pieces
    counts = []
    for color in [chess.WHITE, chess.BLACK]:
        for pt in [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]:
            counts.append(float(len(board.pieces(pt, color))))

    # Material difference
    vals = {1: 1, 2: 3, 3: 3.4, 4: 5, 5: 9, 6: 0}
    score = 0
    for sq, pc in board.piece_map().items():
        v = vals[pc.piece_type]
        score += v if pc.color == board.turn else -v
    material_diff = [score / 40.0]

    # Check / Checkmate
    in_check = [1.0 if board.is_check() else 0.0]
    is_mate = [1.0 if board.is_checkmate() else 0.0]

    return castling + counts + material_diff + in_check + is_mate
