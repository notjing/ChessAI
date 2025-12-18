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
model_dir = "model_cache"

model_path = hf_hub_download(
    repo_id="notjing/chessai",
    filename="chessai_model.keras",
    local_dir=model_dir,
    local_dir_use_symlinks=False
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
    # 1. Extract all data
    turn, castling_rights, material, setup, material_count, checkmate = board_parameters(board)
    white_control, black_control = square_control(board)
    hanging = hanging_grids(board)

    pieces = np.array(makeboards(board), dtype="float32")
    setup_plane = np.expand_dims(np.array(setup, dtype="float32"), 0)
    white_ctrl_plane = np.expand_dims(np.array(white_control, dtype="float32"), 0)
    black_ctrl_plane = np.expand_dims(np.array(black_control, dtype="float32"), 0)

    white_hanging = np.array(hanging[0], dtype="float32")
    black_hanging = np.array(hanging[1], dtype="float32")

    # 4. Concatenate all 17 layers
    planes = np.concatenate(
        [
            pieces,  # 12 layers
            setup_plane,  # 1 layer
            white_ctrl_plane,  # 1 layer
            black_ctrl_plane,  # 1 layer

        ],
        axis=0
    )

    #(17, 8, 8) -> (8, 8, 17) -> (1, 8, 8, 17)
    planes = np.transpose(planes, (1, 2, 0))
    planes = np.expand_dims(planes, 0)

    vec_data = castling_rights + [turn] + material + material_count + [int(checkmate)]
    vec = np.array([vec_data], dtype="float32")

    preds = predict_fn([planes, vec])

    pred_norm = preds.numpy()[0][0]
    return pred_norm * y_std + y_mean



@tf.function
def predict_fn(inputs):
    return mod(inputs, training=False)
