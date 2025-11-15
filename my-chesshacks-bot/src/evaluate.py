import numpy as np
import chess
import tensorflow as tf
from keras.models import load_model
from huggingface_hub import hf_hub_download

# -------------------------------
# Board encoding (same as training)
# -------------------------------
def find_piece(board, piece_type, color):
    coords = []
    for square in board.pieces(piece_type, color):
        row = 7 - (square // 8)
        col = square % 8
        coords.append((row, col))
    return coords

def makeboards(board):
    piece_types = [
        chess.PAWN, chess.KNIGHT, chess.BISHOP,
        chess.ROOK, chess.QUEEN, chess.KING
    ]
    colors = [chess.WHITE, chess.BLACK]

    layers = []
    for color in colors:
        for piece in piece_types:
            setup = [[0 for _ in range(8)] for _ in range(8)]
            coords = find_piece(board, piece, color)
            for x, y in coords:
                setup[x][y] = 1
            layers.append(setup)

    return np.array(layers, dtype="float32")


# -------------------------------
# Load model
# -------------------------------
model_path = hf_hub_download(
    repo_id="notjing/chessai",
    filename="chessai_model.keras"
)

mod = tf.keras.models.load_model(model_path)

# IMPORTANT:
# Replace with the y_mean and y_std printed during training!
y_mean = 51.231956481933594
y_std = 383.2938537597656


def square_control(board):
    # Creates 2 8x8 matrices; one for our own control and one for opponent control
    white_control = [[0 for _ in range(8)] for _ in range(8)]
    black_control = [[0 for _ in range(8)] for _ in range(8)]

    for square in chess.SQUARES:
        # Who controls this square?

        # Attacks from our side (white)
        attackers_our = board.attackers(chess.WHITE, square)
        white_control[chess.square_rank(square)][chess.square_file(square)] = len(attackers_our)

        # Attacks from opponent (black)
        attackers_opponent = board.attackers(chess.BLACK, square)
        black_control[chess.square_rank(square)][chess.square_file(square)] = len(attackers_opponent)
    return white_control, black_control

def board_parameters(board):

    #turn.board() returns true if it's white's turn, else, black's turn
    turn = board.turn
    if turn:
        turn = 1
    else:
        turn = 0

    #Kingside, queenside
    white_rights = [0,0]
    black_rights= [0,0]

    white_ks = board.has_kingside_castling_rights(chess.WHITE)
    if white_ks:
        white_rights[0] = 1
    white_qs = board.has_queenside_castling_rights(chess.WHITE)
    if white_qs:
        white_rights[1] = 1
    black_ks = board.has_kingside_castling_rights(chess.BLACK )
    if black_ks:
        black_rights[0] = 1
    black_qs = board.has_queenside_castling_rights(chess.BLACK)
    if black_qs:
        black_rights[1] = 1


    castling_rights = white_rights + black_rights

    material = []
    white_material, black_material = 0 , 0

    piece_values = {
        chess.PAWN: 1,
        chess.KNIGHT: 3,
        chess.BISHOP: 3.4,
        chess.ROOK: 5,
        chess.QUEEN: 9,
        chess.KING: 0  # Kings are not typically counted in material value
    }

    for square, piece in board.piece_map().items():
        value = piece_values[piece.piece_type]
        if piece.color == chess.WHITE:
            white_material += value
        else:
            black_material += value

    material = [white_material, black_material]

    ep_rights = board.ep_square
    setup = [[0 for _ in range(8)] for _ in range(8)]

    if ep_rights != None:
        rank = chess.square_rank(ep_rights)    # 0–7  (0 = rank 1)
        file = chess.square_file(ep_rights)    # 0–7  (0 = file a)
        coords = (7 - rank , file)
        setup[coords[0]][coords[1]] = 1
        ep_rights = True

    return turn, castling_rights, material, setup


# -------------------------------
# Evaluation function
# -------------------------------
def evaluate_board(board):
    turn, castling_rights, material, setup = board_parameters(board)
    white_control, black_control = square_control(board)

    # Base 12 planes
    planes = makeboards(board)  # (12, 8, 8)

    # Extra planes
    setup_plane = np.array(setup, dtype="float32")              # (8,8)
    white_control = np.array(white_control, dtype="float32")    # (8,8)
    black_control = np.array(black_control, dtype="float32")    # (8,8)

    # Step 1: Build (15,8,8)
    planes = np.concatenate(
        [
            planes,
            setup_plane[None, ...],
            white_control[None, ...],
            black_control[None, ...]
        ],
        axis=0
    )

    # Step 2: (8,8,15)
    planes = np.transpose(planes, (1, 2, 0))

    # Step 3: (1,8,8,15)
    planes = np.expand_dims(planes, 0)

    # Extra vector: (1,7)
    vec = np.array([castling_rights + [turn] + material], dtype="float32")

    preds = predict_fn([planes, vec])

    pred_norm = preds.numpy()[0][0]
    return pred_norm * y_std + y_mean



@tf.function
def predict_fn(inputs):
    return mod(inputs, training=False)