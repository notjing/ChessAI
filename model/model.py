import numpy as np
import tensorflow as tf
import chess
import pandas as pd

def parse(file):
    chess_data = pd.read_csv('model/chess_evaluations/chessData.csv')

    first_column = chess_data.iloc[:, 0].tolist()
    second_column = chess_data.iloc[:, 1].tolist()

    return first_column, second_column


def find_piece(board, piece_type, color):
    """
    Finds all squares with a given piece type and color.
    Returns a list of (row, col) coordinates.
    """
    coords = []
    
    for square in board.pieces(piece_type, color):
        row = 7 - (square // 8)  # Flip row so top of board is 0
        col = square % 8
        coords.append((row, col))
    
    return coords

#board is a pythn chess board object
#make 12 8x8 arrays that are 1s and 0s for if the piece is there
def makeboards(board):
    """
    Converts a python-chess Board object into 12 8x8 lists of lists.
    Order of layers:
    White: pawn, knight, bishop, rook, queen, king
    Black: pawn, knight, bishop, rook, queen, king
    """
    
    #List of all chess pieces corresponding to python-chess
    piece_types = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]
    #The 2 colours
    colors = [chess.WHITE, chess.BLACK]
    
    #12 layers
    layers = []

    for color in (colors):
        for piece in (piece_types):
            setup = [[0 for _ in range(8)] for _ in range(8)]
            coords = find_piece(board, piece, color)
            for x in range (len(coords)):
                setup[coords[x][0]][coords[x][1]] = 1
            layers.append(setup)

    return layers

def reshape(data):
    red = data[:, 0:1024].reshape(-1, 32, 32)
    green = data[:, 1024:2048].reshape(-1, 32, 32)
    blue = data[:, 2048:3072].reshape(-1, 32, 32)

    images = np.stack([red, green, blue], axis=-1)

    return images

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

    castling_rights = []
    castling_rights.append(white_rights , black_rights)

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

    material.append(white_material , black_material)

    ep_rights = board.ep_square
    if ep_rights != None:
        setup = [[0 for _ in range(8)] for _ in range(8)]
        rank = chess.square_rank(ep_rights)    # 0–7  (0 = rank 1)
        file = chess.square_file(ep_rights)    # 0–7  (0 = file a)
        coords = (8 - rank , file + 1)
        setup[coords[0]][coords[1]] = 1
        ep_rights = True 
        return turn, castling_rights , material , ep_rights , setup

    return turn, castling_rights , material , ep_rights


def main():

    x_train = []
    y_train = []

    for i in range(1, 6):
        data = unpickle(f'cifar-10-batches-py/data_batch_{i}')
        x_train.extend(data[b'data'])
        y_train.extend(data[b'labels'])

    test_data = unpickle('cifar-10-batches-py/test_batch')
    x_test = np.array(test_data[b'data'])
    y_test = np.array(test_data[b'labels'])

    x_train = np.array(x_train)
    x_test = np.array(x_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    x_train /= 255.0
    x_test /= 255.0

    x_train = reshape(x_train)
    x_test = reshape(x_test)

    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(32, 32, 3), padding="same"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(32, 32, 3), padding="same"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Dropout(0.2),

        tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu', padding="same"),
        tf.keras.layers.BatchNormalization(),

        #tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu', padding="same"),
        #tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

        tf.keras.layers.Dropout(0.3),

        tf.keras.layers.Conv2D(128, kernel_size=(3, 3), activation='relu', padding="same"),
        tf.keras.layers.BatchNormalization(),

        #tf.keras.layers.Conv2D(128, kernel_size=(3, 3), activation='relu', padding="same"),
        #tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

        tf.keras.layers.Dropout(0.4),


        tf.keras.layers.Flatten(),
        #tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)

    initial_learning_rate = 0.001

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate,
        decay_steps=1000,
        decay_rate=0.9,
        staircase=True
    )

    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy', patience=10, restore_best_weights=True)

    model.compile(optimizer=optimizer,
                  loss=loss_fn,
                  metrics=['accuracy'],
                  )

    model.fit(x_train, y_train, epochs=5, validation_split=0.1, callbacks=[early_stopping])

    model.evaluate(x_test, y_test, verbose=2)