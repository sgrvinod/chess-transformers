import os
import chess
import random
import chess.pgn
import tables as tb
from config import *
from tqdm import tqdm


random.seed(1234)


def parse_pgn_data():

    # Create table description for HDF5 file
    class ChessTable(tb.IsDescription):
        board_position = tb.StringCol(64)
        turn = tb.StringCol(1)
        white_kingside_castling_rights = tb.BoolCol()
        white_queenside_castling_rights = tb.BoolCol()
        black_kingside_castling_rights = tb.BoolCol()
        black_queenside_castling_rights = tb.BoolCol()
        can_claim_draw = tb.BoolCol()
        output_sequence = tb.StringCol(
            shape=(MAX_MOVE_SEQUENCE_LENGTH), itemsize=8, dflt="<pad>"
        )  # "dflt" doesn't work for some reason

    # Delete HDF5 file if it already exists; start anew
    if os.path.exists(os.path.join(DATA_FOLDER, H5_FILE)):
        os.remove(os.path.join(DATA_FOLDER, H5_FILE))

    # Create new HDF5 file
    h5_file = tb.open_file(
        os.path.join(DATA_FOLDER, H5_FILE), mode="w", title="data file"
    )

    # Create table in HDF5 file
    table = h5_file.create_table("/", "data", ChessTable, expectedrows=15000000)

    # Create pointer to next row in this table
    row = table.row

    # Get names of PGN files
    pgn_files = [f for f in os.listdir(DATA_FOLDER) if f.endswith(".pgn")]
    print("There are %d PGN files to parse." % len(pgn_files))

    # Parse PGN files
    for pgn_file in sorted(pgn_files):
        pgn_file_path = os.path.join(DATA_FOLDER, pgn_file)

        # Read, extract, parse games
        games = read_pgn_file(pgn_file_path)
        input_and_output_sequences = list()
        for game in tqdm(games, desc="Parsing this file"):
            input_and_output_sequences.extend(
                get_input_and_output_sequences(
                    game,
                    max_output_sequence_length=MAX_MOVE_SEQUENCE_LENGTH,
                    output_sequence_notation=MOVE_SEQUENCE_NOTATION,
                )
            )
        del games

        # Write parsed datapoints to the HDF5 file
        for datapoint in input_and_output_sequences:
            row["board_position"] = datapoint["input_sequence"]["board_position"]
            row["turn"] = datapoint["input_sequence"]["turn"]
            row["white_kingside_castling_rights"] = datapoint["input_sequence"][
                "white_kingside_castling_rights"
            ]
            row["white_queenside_castling_rights"] = datapoint["input_sequence"][
                "white_queenside_castling_rights"
            ]
            row["black_kingside_castling_rights"] = datapoint["input_sequence"][
                "black_kingside_castling_rights"
            ]
            row["black_queenside_castling_rights"] = datapoint["input_sequence"][
                "black_queenside_castling_rights"
            ]
            row["can_claim_draw"] = datapoint["input_sequence"]["can_claim_draw"]
            row["output_sequence"] = datapoint["output_sequence"] + ["<pad>"] * (
                MAX_MOVE_SEQUENCE_LENGTH - len(datapoint["output_sequence"])
            )
            row.append()
        table.flush()
        print("A total of %d datapoints have been saved to disk." % table.nrows)
        del input_and_output_sequences

    h5_file.close()
    print("Done.")


def read_pgn_file(pgn_file_path):
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
    # Get current board position
    ranks = str(board).splitlines()
    ranks.reverse()
    board_position = " ".join(ranks).replace(" ", "")

    # If there is a square that can be attacked with a legal en-passant move, indicate in board position
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


def get_input_and_output_sequences(game):

    # Get all moves from this game, and board status before each move
    board = game.board()
    all_moves = list()
    board_status_before_moves = list()
    for move_number, move in enumerate(game.mainline_moves()):

        # Get current board status (i.e., before the next move is made)
        board_status_before_moves.append(get_board_status(board))

        # Get this move in the desired output sequence notation
        if MOVE_SEQUENCE_NOTATION == "uci":
            all_moves.append(
                board.uci(move).replace("+", "").replace("#", "").replace("x", "")
            )
        elif MOVE_SEQUENCE_NOTATION == "lan":
            all_moves.append(
                board.lan(move).replace("+", "").replace("#", "").replace("x", "")
            )
        elif MOVE_SEQUENCE_NOTATION == "san":
            all_moves.append(
                board.san(move).replace("+", "").replace("#", "").replace("x", "")
            )
        else:
            raise NotImplementedError

        # Make the move on the board
        board.push(move)

    # If game ended in checkmate, denote the loss as a "move" by the losing player
    # This is to condition the network to recognize the checkmate and to end the sequence with this move (like an "end generation" token)
    if board.is_checkmate():
        all_moves.append("<loss>")
        board_status_before_moves.append(
            get_board_status(board)
        )  # get board state at this "move"

    # Find winner because we want input and output sequences for the winner's moves only
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

    # Create input sequences and output sequences
    input_and_output_sequences = list()
    for move_number in range(len(all_moves)):
        # Consider only the winner's moves
        if board_status_before_moves[move_number]["turn"] == winner:
            input_and_output_sequences.append(
                {
                    "input_sequence": board_status_before_moves[move_number],
                    "output_sequence": all_moves[
                        move_number : move_number + MAX_MOVE_SEQUENCE_LENGTH
                    ],
                }
            )

    return input_and_output_sequences


if __name__ == "__main__":
    parse_pgn_data(data_folder=DATA_FOLDER, h5_file=H5_FILE)
