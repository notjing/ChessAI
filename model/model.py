import os
import shutil
import numpy as np
import tensorflow as tf
from tensorflow.keras import regularizers
import chess
import pandas as pd
from huggingface_hub import HfApi, Repository

# Make sure you've logged in via CLI: huggingface-cli login

# Create the repo if it doesn't exist
api = HfApi()
api.create_repo("chessai", exist_ok=True)

def parse(file, nrows=None):
    chess_data = pd.read_csv(file, nrows=nrows)
    print("parsing")
    first_column = chess_data.iloc[:, 0].tolist()
    second_column = chess_data.iloc[:, 1].tolist()
    print(f"parsed: {len(first_column)} rows")
    return first_column, second_column

def convert_eval_to_numeric(eval_str):
    try:
        value = float(eval_str)
        return np.clip(value, -1500, 1500)
    except (ValueError, TypeError):
        return 1500.0

def find_piece(board, piece_type, color):
    coords = []
    for square in board.pieces(piece_type, color):
        row = 7 - (square // 8)
        col = square % 8
        coords.append((row, col))
    return coords

def makeboards(board):
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

def main():
    # Load FENs and evaluations
    fens, evals = parse('chess_evaluations/chessData.csv', nrows=10000)
    evals = [convert_eval_to_numeric(e) for e in evals]

    print(f"Processing {len(fens)} positions")
    boards = []
    for idx, fen in enumerate(fens):
        board = chess.Board(fen)
        boards.append(makeboards(board))
        if idx % 10000 == 0:
            print(f"Processed {idx}/{len(fens)}")

    boards = np.array(boards, dtype='float32')
    evals = np.array(evals, dtype='float32')

    # Shuffle data
    indices = np.arange(len(boards))
    np.random.shuffle(indices)
    boards = boards[indices]
    evals = evals[indices]

    # Train/test split
    split_index = int(len(boards) * 0.8)
    x_train, y_train = boards[:split_index], evals[:split_index]
    x_test, y_test = boards[split_index:], evals[split_index:]

    # Normalize evaluations
    y_mean = np.mean(y_train)
    y_std = np.std(y_train) + 1e-6
    y_train = (y_train - y_mean) / y_std
    y_test = (y_test - y_mean) / y_std

    # Transpose to (batch, height, width, channels)e
    x_train = np.transpose(x_train, (0, 2, 3, 1))
    x_test = np.transpose(x_test, (0, 2, 3, 1))
    print(f"Training shape: {x_train.shape}")

    # Define model
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(8, 8, 12), padding="same"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(16, (3, 3), activation='relu', padding="same"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding="same"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding="same"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding="same"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding="same",
                               kernel_regularizer=regularizers.l2(0.001)),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(1, activation='linear')
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(0.0001),
                  loss=tf.keras.losses.MeanAbsoluteError(),
                  metrics=['mae'])

    # Train
    print("Starting training...")
    model.fit(x_train, y_train, epochs=1, validation_split=0.1, batch_size=128)
    print("Evaluating...")
    model.evaluate(x_test, y_test, verbose=2)

    # Save model as .keras file
    model_file = "chessai_model.keras"
    model.save(model_file)

    # Folder for Hugging Face repo (let Repository clone it)
    hf_folder = "chessai_model_repo"
    if os.path.exists(hf_folder):
        shutil.rmtree(hf_folder)

    # Clone the repo into an empty folder
    repo = Repository(
        local_dir=hf_folder,
        clone_from="notjing/chessai",  # <-- REPLACE with your HF username
        use_auth_token=True
    )

    # Copy the model file into the repo folder
    shutil.copy(model_file, hf_folder)

    # Push to Hugging Face
    repo.push_to_hub(commit_message="Upload trained chess AI model")
    print("Model uploaded successfully!")

if __name__ == "__main__":
    main()
