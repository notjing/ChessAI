import math
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

def evaluate_board(boards):

    batch_planes = []
    batch_vecs = []

    for board in boards:
        flip = (board.turn == chess.BLACK)

        dense_input = board_parameters(board)
        batch_vecs.append(dense_input)

        layers = makeboards(board)
        s_control = square_control(board)

        # Creates enpassant grid
        ep_grid = np.zeros((8, 8), dtype=np.float32)
        if board.ep_square is not None:
            r, c = get_mapped_coords(board.ep_square, flip)
            ep_grid[r][c] = 1.0

        planes = np.array(layers + s_control + [ep_grid], dtype="float32")

        planes = np.transpose(planes, (1, 2, 0))
        # planes = np.expand_dims(planes, 0)

        batch_planes.append(planes)

    input_planes = np.array(batch_planes, dtype="float32")
    input_vecs = np.array(batch_vecs, dtype="float32")

    preds = predict_fn([input_planes, input_vecs])
    raw_scores = preds.numpy().flatten()

    win_probabilities = 2 / (1 + np.exp(-0.00368208 * raw_scores*1500)) - 1

    return win_probabilities

print(evaluate_board([chess.Board("r7/p5k1/1p1Q2p1/2pPp2p/2P1R3/5r2/PPB2qPP/4R2K w - - 2 33")]))


