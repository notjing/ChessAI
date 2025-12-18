import sys
import numpy as np
import chess
import tensorflow as tf
from keras.models import load_model
from huggingface_hub import hf_hub_download

from model import createparams

# -------------------------------
# Board encoding (same as training)
# -------------------------------
def find_piece(board, piece_type, color):
    return createparams.find_piece(board, piece_type, color)

def makeboards(board):
    return createparams.makeboards(board)


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
    return createparams.square_control(board)

def hanging_grids(board):
    return createparams.hanging_grids(board)

def board_parameters(board):
    return createparams.board_parameters(board)


# -------------------------------
# Evaluation function
# -------------------------------
def evaluate_board(board):
    turn, castling_rights, material, setup, material_count, checkmate = board_parameters(board)
    white_control, black_control = square_control(board)
    hanging = hanging_grids(board)
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
    vec = np.array([castling_rights + [turn] + material + material_count + [checkmate]], dtype="float32")

    preds = predict_fn([planes, vec])

    pred_norm = preds.numpy()[0][0]
    return pred_norm * y_std + y_mean



@tf.function
def predict_fn(inputs):
    return mod(inputs, training=False)