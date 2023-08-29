import os
import chess
import random
import argparse
import chess.pgn
import tables as tb
from tqdm import tqdm
from importlib import import_module


random.seed(1234)


def parse_pgn_data(
    data_folder,
    h5_file,
    max_move_sequence_length,
    move_sequence_notation,
    expected_rows,
):
    """
    Parse all PGN data in a given location into a H5 store.

    Args:

        data_folder (str): Folder containing the PGN files.

        h5_file (str): Name of the H5 file.

        max_move_sequence_length (str): Maximum number of next-moves to
        parse at each position.

        move_sequence_notation (str): The move notation to be used.

        expected_rows (int): The expected number of rows, approximately,
        in the H5 table.
    """

    # Create table description for HDF5 file
    class ChessTable(tb.IsDescription):
        board_position = tb.StringCol(64)
        turn = tb.StringCol(1)
        white_kingside_castling_rights = tb.BoolCol()
        white_queenside_castling_rights = tb.BoolCol()
        black_kingside_castling_rights = tb.BoolCol()
        black_queenside_castling_rights = tb.BoolCol()
        can_claim_draw = tb.BoolCol()
        move_sequence = tb.StringCol(
            shape=(max_move_sequence_length), itemsize=8, dflt="<pad>"
        )  # "dflt" doesn't work for some reason

    # Delete HDF5 file if it already exists; start anew
    if os.path.exists(os.path.join(data_folder, h5_file)):
        os.remove(os.path.join(data_folder, h5_file))

    # Create new HDF5 file
    h5_file = tb.open_file(
        os.path.join(data_folder, h5_file), mode="w", title="data file"
    )

    # Create table in HDF5 file
    table = h5_file.create_table("/", "data", ChessTable, expectedrows=expected_rows)

    # Create pointer to next row in this table
    row = table.row

    # Get names of PGN files
    pgn_files = [f for f in os.listdir(data_folder) if f.endswith(".pgn")]
    print("\nThere are %d PGN files to parse." % len(pgn_files))

    # Parse PGN files
    for pgn_file in sorted(pgn_files):
        pgn_file_path = os.path.join(data_folder, pgn_file)

        # Read, extract, parse games
        print("\nReading {}...".format(pgn_file))
        games = read_pgn_file(pgn_file_path)
        print("...done.\n")
        board_status_and_move_sequences = list()
        for game in tqdm(games, desc="Parsing {}".format(pgn_file)):
            board_status_and_move_sequences.extend(
                get_board_status_and_move_sequences(
                    game,
                    max_move_sequence_length=max_move_sequence_length,
                    move_sequence_notation=move_sequence_notation,
                )
            )
        del games

        # Write parsed datapoints to the HDF5 file
        for datapoint in board_status_and_move_sequences:
            row["board_position"] = datapoint["board_status_sequence"]["board_position"]
            row["turn"] = datapoint["board_status_sequence"]["turn"]
            row["white_kingside_castling_rights"] = datapoint["board_status_sequence"][
                "white_kingside_castling_rights"
            ]
            row["white_queenside_castling_rights"] = datapoint["board_status_sequence"][
                "white_queenside_castling_rights"
            ]
            row["black_kingside_castling_rights"] = datapoint["board_status_sequence"][
                "black_kingside_castling_rights"
            ]
            row["black_queenside_castling_rights"] = datapoint["board_status_sequence"][
                "black_queenside_castling_rights"
            ]
            row["can_claim_draw"] = datapoint["board_status_sequence"]["can_claim_draw"]
            row["move_sequence"] = datapoint["move_sequence"] + ["<pad>"] * (
                max_move_sequence_length - len(datapoint["move_sequence"])
            )
            row.append()
        table.flush()
        print("\nA total of %d datapoints have been saved to disk." % table.nrows)
        del board_status_and_move_sequences

    print("\n")
    h5_file.close()


def read_pgn_file(pgn_file_path):
    """
    Read the games in a PGN file.

    Args:

        pgn_file_path (str): Path to the PGN file.

    Returns:

        list: A list of games.
    """
    pgn_file_contents = open(pgn_file_path)
    games = list()
    there_are_more_games = True
    while there_are_more_games:
        game = chess.pgn.read_game(pgn_file_contents)
        there_are_more_games = game is not None
        if there_are_more_games:
            games.append(game)

    return games


def get_board_status(board):
    """
    Describe a chessboard.

    Args:

        board (chess.Board): The chessboard.

    Returns:

        dict: A description of the chessboard.
    """
    # Get current board position
    ranks = str(board).splitlines()
    ranks.reverse()
    board_position = " ".join(ranks).replace(" ", "")

    # If there is a square that can be attacked with a legal en-passant
    # move, indicate in board position
    if board.ep_square is not None and board.has_legal_en_passant():
        board_position = list(board_position)
        board_position[board.ep_square] = ","
        board_position = "".join(board_position)

    # Get player who should move in the next turn
    turn = "w" if board.turn else "b"

    # Get castling rights
    white_can_castle_kingside = board.has_kingside_castling_rights(chess.WHITE)
    white_can_castle_queenside = board.has_queenside_castling_rights(chess.WHITE)
    black_can_castle_kingside = board.has_kingside_castling_rights(chess.BLACK)
    black_can_castle_queenside = board.has_queenside_castling_rights(chess.BLACK)

    # Check if a draw can be claimed in the next turn
    can_claim_draw = (
        board.can_claim_fifty_moves() or board.can_claim_threefold_repetition()
    )

    # Compile board status
    board_status = {
        "board_position": board_position,
        "turn": turn,
        "white_kingside_castling_rights": white_can_castle_kingside,
        "white_queenside_castling_rights": white_can_castle_queenside,
        "black_kingside_castling_rights": black_can_castle_kingside,
        "black_queenside_castling_rights": black_can_castle_queenside,
        "can_claim_draw": can_claim_draw,
    }

    return board_status


def get_board_status_and_move_sequences(
    game, move_sequence_notation, max_move_sequence_length
):
    """
    Get board descriptions and (next) move sequences at all positions in
    a chess game.

    Args:

        game (chess.pgn.Game): The chess game.

        move_sequence_notation (str): Move notation.

        max_move_sequence_length (int): Maximum number of next-moves for
        each position.

    Raises:

        NotImplementedError: If the specified move notation is not one
        of "uci", "lan", or "san".

    Returns:

        dict: Board descriptions and (next) move sequences at all
        positions in the chess game.
    """

    # Get all moves from this game, and board status before each move
    board = game.board()
    all_moves = list()
    board_status_before_moves = list()
    for move_number, move in enumerate(game.mainline_moves()):

        # Get current board status (i.e., before the next move is made)
        board_status_before_moves.append(get_board_status(board))

        # Get this move in the desired move sequence notation
        if move_sequence_notation == "uci":
            all_moves.append(
                board.uci(move).replace("+", "").replace("#", "").replace("x", "")
            )
        elif move_sequence_notation == "lan":
            all_moves.append(
                board.lan(move).replace("+", "").replace("#", "").replace("x", "")
            )
        elif move_sequence_notation == "san":
            all_moves.append(
                board.san(move).replace("+", "").replace("#", "").replace("x", "")
            )
        else:
            raise NotImplementedError

        # Make the move on the board
        board.push(move)

    # If game ended in checkmate, denote the loss as a "move" by the
    # losing player This is to condition the network to recognize the
    # checkmate and to end the sequence with this move (like an "end
    # generation" token)
    if board.is_checkmate():
        all_moves.append("<loss>")
        board_status_before_moves.append(
            get_board_status(board)
        )  # get board state at this "move"

    # Find winner because we want board_status and move sequences for
    # the winner's moves only
    result = game.headers["Result"]
    if result == "1-0":
        winner = "w"

    elif result == "0-1":
        winner = "b"

    elif result == "1/2-1/2":
        winner = random.choice(["w", "b"])  # choose winner at random
        all_moves.append(
            "<draw>"
        )  # assumes player whose turn is next offered a draw, as a "move"
        board_status_before_moves.append(
            get_board_status(board)
        )  # get board state at this "move"

    else:
        return []  # return no sequences in games that had an unnatural ending

    # Create board_status sequences and move sequences
    board_status_and_move_sequences = list()
    for move_number in range(len(all_moves)):
        # Consider only the winner's moves
        if board_status_before_moves[move_number]["turn"] == winner:
            board_status_and_move_sequences.append(
                {
                    "board_status_sequence": board_status_before_moves[move_number],
                    "move_sequence": all_moves[
                        move_number : move_number + max_move_sequence_length
                    ],
                }
            )

    return board_status_and_move_sequences


if __name__ == "__main__":
    # Get configuration
    parser = argparse.ArgumentParser()
    parser.add_argument("config_name", type=str, help="Name of configuration file.")
    args = parser.parse_args()
    CONFIG = import_module("configs.{}".format(args.config_name))

    # Parse PGN data
    parse_pgn_data(
        data_folder=CONFIG.DATA_FOLDER,
        h5_file=CONFIG.H5_FILE,
        max_move_sequence_length=CONFIG.MAX_MOVE_SEQUENCE_LENGTH,
        move_sequence_notation=CONFIG.MOVE_SEQUENCE_NOTATION,
        expected_rows=CONFIG.EXPECTED_ROWS,
    )
