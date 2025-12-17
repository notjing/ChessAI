import os
import shutil
import numpy as np
import tensorflow as tf
import chess
import pandas as pd
from huggingface_hub import HfApi, Repository


def parse(file, nrows=None):
    chess_data = pd.read_csv(file, nrows=nrows)
    print("Parsing...")
    first_column = chess_data.iloc[:, 0].tolist()
    second_column = chess_data.iloc[:, 1].tolist()
    print(f"Parsed: {len(first_column)} rows")
    return first_column, second_column


def convert_eval_to_numeric(eval_str):
    try:
        value = float(eval_str)
        return np.clip(value, -1500, 1500)
    except (ValueError, TypeError):
        if eval_str[1] == '+':
            return 1500
        else:
            return -1500


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

def flip_fen_board(fen: str) -> str:
    """Flips a FEN string vertically and switches colors."""
    parts = fen.split()
    board_part, turn, castling, ep, halfmove, fullmove = parts

    ranks = board_part.split('/')
    flipped_ranks = []
    for rank in reversed(ranks):
        new_rank = ''
        for c in rank:
            if c.isalpha():
                new_rank += c.lower() if c.isupper() else c.upper()
            else:
                new_rank += c
        flipped_ranks.append(new_rank)

    new_board = '/'.join(flipped_ranks)
    new_turn = 'b' if turn == 'w' else 'w'
    return f"{new_board} {new_turn} {castling} {ep} {halfmove} {fullmove}"


def main():
    # Load FENs and evaluations
    fens, evals = parse('chess_evaluations/random_evals.csv', nrows=450000)
    evals = [convert_eval_to_numeric(e) for e in evals]

    all_boards, all_x_extra, all_evals = [], [], []

    print(f"Processing {len(fens)} positions")

    for idx, fen in enumerate(fens):
        board = chess.Board(fen)

        # Original board
        turn, castling_rights, material, ep_layer, material_count, checkmate = board_parameters(board)
        white_control, black_control = square_control(board)
        hanging = hanging_grids(board)
        board_layers = np.array(makeboards(board) + [ep_layer, white_control, black_control] + hanging, dtype='float32')
        all_boards.append(board_layers)
        all_x_extra.append(castling_rights + [turn] + material + material_count + [checkmate])
        all_evals.append(evals[idx])

        # Flipped board
        flipped_fen = flip_fen_board(fen)
        flipped_board = chess.Board(flipped_fen)
        turn_f, castling_rights_f, material_f, ep_layer_f, material_count_f, checkmate_f = board_parameters(flipped_board)
        white_control_f, black_control_f = square_control(flipped_board)
        hanging_f = hanging_grids(flipped_board)
        flipped_layers = np.array(makeboards(flipped_board) + [ep_layer_f, white_control_f, black_control_f] + hanging_f, dtype='float32')
        all_boards.append(flipped_layers)
        all_x_extra.append(castling_rights_f + [turn_f] + material_f + material_count_f + [checkmate_f])
        all_evals.append(-evals[idx])  # Negated evaluation

    # Convert to numpy arrays
    boards = np.array(all_boards, dtype='float32')
    x_train_extra = np.array(all_x_extra, dtype='float32')
    evals = np.array(all_evals, dtype='float32')

    # Shuffle data
    indices = np.arange(len(boards))
    np.random.shuffle(indices)
    boards = boards[indices]
    evals = evals[indices]

    # Train/test split
    split_index = int(len(boards) * 0.8)
    x_train, y_train = boards[:split_index], evals[:split_index]
    x_test, y_test = boards[split_index:], evals[split_index:]
    x_train_extra_split = x_train_extra[:split_index]
    x_test_extra_split = x_train_extra[split_index:]

    # Normalize evaluations
    y_mean = np.mean(y_train)
    y_std = np.std(y_train) + 1e-6
    print(f"mean: {y_mean}, std: {y_std}")
    y_train = (y_train - y_mean) / y_std
    y_test = (y_test - y_mean) / y_std

    # Transpose to (batch, height, width, channels)
    x_train = np.transpose(x_train, (0, 2, 3, 1))
    x_test = np.transpose(x_test, (0, 2, 3, 1))
    print(f"Training shape: {x_train.shape}")

    # Define CNN + extra input model
    cnn_input = tf.keras.Input(shape=(8, 8, 17), name="board_input")
    x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding="same")(cnn_input)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding="same", kernel_regularizer=regularizers.l2(0.001))(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dense(1, activation='linear')(x)

    extra_input = tf.keras.Input(shape=(19,), name="extra_input")
    y = tf.keras.layers.Dense(32, activation='relu')(extra_input)
    y = tf.keras.layers.Dense(16, activation='relu')(y)

    combined = tf.keras.layers.Concatenate()([x, y])
    z = tf.keras.layers.Dense(64, activation='relu')(combined)
    z = tf.keras.layers.Dense(32, activation='relu')(z)
    output = tf.keras.layers.Dense(1, activation='linear')(z)

    aimodel = Model(inputs=[cnn_input, extra_input], outputs=output)
    aimodel.compile(optimizer=tf.keras.optimizers.Adam(0.0002),
                     loss=tf.keras.losses.MeanSquaredError(),
                     metrics=['mse'])

    # Train the model
    print("Starting training...")
    aimodel.fit([x_train, x_train_extra_split], y_train, validation_split=0.1, epochs=25, batch_size=64, shuffle=True)

    # Evaluate the model
    print("Evaluating...")
    aimodel.evaluate([x_test, x_test_extra_split], y_test, verbose=2)

    # Save model
    model_file = "chessai_model.keras"
    aimodel.save(model_file)

    # Upload to Hugging Face
    hf_folder = "chessai_model_repo"
    if os.path.exists(hf_folder):
        print(f"{hf_folder} already exists, using it.")

    repo = Repository(local_dir=hf_folder,
                      clone_from="notjing/chessai",  # <-- replace with your HF repo
                      use_auth_token=True)
    shutil.copy(model_file, hf_folder)
    repo.push_to_hub(commit_message="Upload trained chess AI aimodel")
    print("Model uploaded successfully!")


if __name__ == "__main__":
    main()
