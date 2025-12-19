import sys
import numpy as np
import chess
import tensorflow as tf
from huggingface_hub import hf_hub_download
from model.createparams import square_control, makeboards, board_parameters, get_mapped_coords


def pawn_error(y_true, y_pred):
    return tf.reduce_mean(tf.abs(y_true - y_pred)) * 1500


model_dir = "model_cache"
model_path = hf_hub_download(
    repo_id="notjing/chessai",
    filename="chessai_model.keras",
    local_dir=model_dir
)

mod = tf.keras.models.load_model(
    model_path,
    custom_objects={'pawn_error': pawn_error}
)


@tf.function
def predict_fn(inputs):
    return mod(inputs, training=False)

def evaluate_board(board):
    flip = (board.turn == chess.BLACK)

    extra_data = board_parameters(board)
    vec = np.array([extra_data], dtype="float32")

    layers_23 = makeboards(board)
    s_control = square_control(board)

    # Creates enpassant grid
    ep_grid = np.zeros((8, 8), dtype=np.float32)
    if board.ep_square is not None:
        r, c = get_mapped_coords(board.ep_square, flip)
        ep_grid[r][c] = 1.0

    planes = np.array(layers_23 + s_control +[ep_grid], dtype="float32")

    planes = np.transpose(planes, (1, 2, 0))
    planes = np.expand_dims(planes, 0)

    preds = predict_fn([planes, vec])
    pred_norm = preds.numpy()[0][0]

    centipawns = pred_norm * 1500

    return centipawns


