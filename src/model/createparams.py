import chess
def square_control(board):
    """Creates 2 8x8 matrices for square control: own and opponent."""
    white_control = [[0 for _ in range(8)] for _ in range(8)]
    black_control = [[0 for _ in range(8)] for _ in range(8)]

    for square in chess.SQUARES:
        # Attacks from white
        attackers_our = board.attackers(chess.WHITE, square)
        white_control[chess.square_rank(square)][chess.square_file(square)] = len(attackers_our)

        # Attacks from black
        attackers_opponent = board.attackers(chess.BLACK, square)
        black_control[chess.square_rank(square)][chess.square_file(square)] = len(attackers_opponent)

    return white_control, black_control


def find_piece(board, piece_type, color):
    """Finds all squares with a given piece type and color."""
    coords = []
    for square in board.pieces(piece_type, color):
        row = 7 - (square // 8)  # Flip row so top of board is 0
        col = square % 8
        coords.append((row, col))
    return coords


def makeboards(board):
    """
    Converts a python-chess Board object into 12 8x8 layers:
    White: pawn, knight, bishop, rook, queen, king
    Black: pawn, knight, bishop, rook, queen, king
    """
    piece_types = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]
    colors = [chess.WHITE, chess.BLACK]

    layers = []
    for color in colors:
        for piece in piece_types:
            setup = [[0 for _ in range(8)] for _ in range(8)]
            coords = find_piece(board, piece, color)
            for x, y in coords:
                setup[x][y] = 1
            layers.append(setup)

    return layers


def board_parameters(board):
    """Extracts board features like turn, castling, material, and en passant."""
    turn = 1 if board.turn else 0

    checkmate = board.is_checkmate()

    # Castling rights
    white_rights = [1 if board.has_kingside_castling_rights(chess.WHITE) else 0,
                    1 if board.has_queenside_castling_rights(chess.WHITE) else 0]
    black_rights = [1 if board.has_kingside_castling_rights(chess.BLACK) else 0,
                    1 if board.has_queenside_castling_rights(chess.BLACK) else 0]
    castling_rights = white_rights + black_rights

    # Material count
    piece_types = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]
    material_count = [len(board.pieces(pt, color)) for color in [chess.WHITE, chess.BLACK] for pt in piece_types]

    # Material value normalized
    piece_values = {
        chess.PAWN: 1,
        chess.KNIGHT: 3,
        chess.BISHOP: 3.4,
        chess.ROOK: 5,
        chess.QUEEN: 9,
        chess.KING: 0  # Not counted
    }
    white_material, black_material = 0, 0
    for square, piece in board.piece_map().items():
        value = piece_values[piece.piece_type]
        if piece.color == chess.WHITE:
            white_material += value
        else:
            black_material += value
    material = [white_material / 39.8 - black_material / 39.8]

    # En passant
    ep_square = board.ep_square
    ep_layer = [[0 for _ in range(8)] for _ in range(8)]
    if ep_square is not None:
        rank = chess.square_rank(ep_square)
        file = chess.square_file(ep_square)
        ep_layer[7 - rank][file] = 1

    return turn, castling_rights, material, ep_layer, material_count, checkmate

def hanging_grids(board):
    white_grid = [[0 for _ in range(8)] for _ in range(8)]
    black_grid = [[0 for _ in range(8)] for _ in range(8)]

    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if not piece:
            continue

        row = 7 - (square // 8)
        col = square % 8

        attackers = board.attackers(not piece.color, square)
        defenders = board.attackers(piece.color, square)

        is_hanging = len(attackers) > 0 and len(defenders) == 0

        if is_hanging:
            if piece.color == chess.WHITE:
                white_grid[row][col] = 1
            else:
                black_grid[row][col] = 1

    return [white_grid, black_grid]
