def parse(file, nrows=None):
    """parses csv file into two lists (fen, eval)"""
    chess_data = pd.read_csv(file, nrows=nrows)

    print("Parsing...")
    first_column = chess_data.iloc[:, 0].tolist()
    second_column = []

    if "mate_in_one" in file:
        second_column = ["#+1" if fen.split()[1] == 'w' else "#-1" for fen in first_column]
    else:
        second_column = chess_data.iloc[:, 1].tolist()

    print(f"Parsed: {len(first_column)} rows")
    return first_column, second_column

def convert_eval_to_numeric(eval_str):
    """if the eval is mate, changes to 1500cp"""
    try:
        value = float(eval_str)
        return np.clip(value, -1500, 1500)
    except (ValueError, TypeError):
        if eval_str[1] == '+':
            return 1500
        else:
            return -1500

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

def generate_data(fens, evals):
    """Creates the data needed for training from the boards"""
    all_boards, all_x_dense, all_evals = [], [], []

    for idx, fen in enumerate(fens):
        if idx % 10000 == 0:
            print(idx)

        board = chess.Board(fen)

        # Original board
        turn, castling_rights, material, ep_layer, material_count, checkmate = board_parameters(board)
        white_control, black_control = square_control(board)
        hanging = hanging_grids(board)
        board_layers = np.array(makeboards(board) + [ep_layer, white_control, black_control], dtype='float32')
        all_boards.append(board_layers)
        all_x_dense.append(castling_rights + [turn] + material + material_count + [checkmate])
        all_evals.append(evals[idx])

        # Flipped board
        flipped_fen = flip_fen_board(fen)
        flipped_board = chess.Board(flipped_fen)
        turn_f, castling_rights_f, material_f, ep_layer_f, material_count_f, checkmate_f = board_parameters(flipped_board)
        white_control_f, black_control_f = square_control(flipped_board)
        hanging_f = hanging_grids(flipped_board)
        flipped_layers = np.array(makeboards(flipped_board) + [ep_layer_f, white_control_f, black_control_f], dtype='float32')
        all_boards.append(flipped_layers)
        all_x_dense.append(castling_rights_f + [turn_f] + material_f + material_count_f + [checkmate_f])
        all_evals.append(-evals[idx])

    return all_boards, all_x_dense, all_evals

def prep_data(all_boards, all_x_dense, all_evals):
    """Creates the training/test data sets and normalises them"""
    # Convert to numpy arrays
    boards = np.array(all_boards, dtype='float32')
    x_dense = np.array(all_x_dense, dtype='float32')
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
    x_train_dense = x_dense[:split_index]
    x_test_dense = x_dense[split_index:]

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

    return x_train, y_train, x_test, y_test, x_train_dense, x_test_dense
