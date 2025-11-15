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

model = tf.keras.models.load_model(model_path)

# IMPORTANT:
# Replace with the y_mean and y_std printed during training!
y_mean = 41.808250427246094
y_std = 370.33038330078125


# -------------------------------
# Evaluation function
# -------------------------------
def evaluate_board(board):
    print("testing")

    planes = makeboards(board)      # (12, 8, 8)
    planes = np.transpose(planes, (1, 2, 0))  # (8, 8, 12)
    planes = np.expand_dims(planes, 0)        # (1, 8, 8, 12)

    pred_norm = model.predict(planes)[0][0]
    pred = pred_norm * y_std + y_mean

    return pred

