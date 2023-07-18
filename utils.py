import chess.pgn
import random
import chess
import json
import os
from tqdm import tqdm
from collections import Counter
import tables as tb

random.seed(1234)


def parse_pgn_data(
    pgn_data_path, max_output_sequence_length=10, output_sequence_notation="uci"
):

    # Check output sequence notation
    output_sequence_notation = output_sequence_notation.lower()
    assert output_sequence_notation in [
        "lan",
        "san",
        "uci",
    ], "Output sequences must be in LAN, SAN, or UCI notations!"

    # Create table description for HDF5 file
    class ChessTable(tb.IsDescription):
        board_position = tb.StringCol(64)
        turn = tb.StringCol(1)
        white_kingside_castling_rights = tb.BoolCol()
        white_queenside_castling_rights = tb.BoolCol()
        black_kingside_castling_rights = tb.BoolCol()
        black_queenside_castling_rights = tb.BoolCol()
        can_claim_draw = tb.BoolCol()

    # Add table columns for output sequence
    for i in range(max_output_sequence_length):
        ChessTable.columns["output_sequence_" + str(i)] = tb.StringCol(8)

    # Delete HDF5 file if it already exists; start anew
    if os.path.exists(os.path.join(pgn_data_path, "data.h5")):
        os.remove(os.path.join(pgn_data_path, "data.h5"))

    # Create new HDF5 file
    h5file = tb.open_file(
        os.path.join(pgn_data_path, "data.h5"), mode="w", title="data file"
    )

    # Create table in HDF5 file
    table = h5file.create_table("/", "data", ChessTable, expectedrows=15000000)

    # Create pointer to next row in this table
    row = table.row

    # Get names of PGN files
    pgn_files = [f for f in os.listdir(pgn_data_path) if f.endswith(".pgn")]
    print("There are %d PGN files to parse." % len(pgn_files))

    # Parse PGN files
    for pgn_file in sorted(pgn_files):
        pgn_file_path = os.path.join(pgn_data_path, pgn_file)

        # Read, extract, parse games
        games = read_pgn_file(pgn_file_path)
        input_and_output_sequences = list()
        for game in tqdm(games, desc="Parsing this file"):
            input_and_output_sequences.extend(
                get_input_and_output_sequences(
                    game,
                    max_output_sequence_length=max_output_sequence_length,
                    output_sequence_notation=output_sequence_notation,
                )
            )
        del games

        # Write parsed datapoints to the HDF5 file
        for datapoint in input_and_output_sequences:
            row["board_position"] = datapoint["input_sequence"]["board_position"]
            row["turn"] = datapoint["input_sequence"]["turn"]
            row["white_kingside_castling_rights"] = datapoint["input_sequence"][
                "castling_rights"
            ]["white_kingside"]
            row["white_queenside_castling_rights"] = datapoint["input_sequence"][
                "castling_rights"
            ]["white_queenside"]
            row["black_kingside_castling_rights"] = datapoint["input_sequence"][
                "castling_rights"
            ]["black_kingside"]
            row["black_queenside_castling_rights"] = datapoint["input_sequence"][
                "castling_rights"
            ]["black_queenside"]
            row["can_claim_draw"] = datapoint["input_sequence"]["can_claim_draw"]
            for i in range(max_output_sequence_length):
                try:
                    row["output_sequence_" + str(i)] = datapoint["output_sequence"][i]
                except IndexError:  # when approaching the end of the game
                    break
            row.append()
        table.flush()
        print("A total of %d datapoints have been saved to disk." % table.nrows)
        del input_and_output_sequences

    h5file.close()
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


def view_game(game):
    board = game.board()
    for move_number, move in enumerate(game.mainline_moves()):
        print("\n")
        print("Move #%d" % move_number)
        print("LAN:", board.lan(move))
        print("SAN:", board.san(move))
        print("UCI:", board.uci(move))
        board.push(move)
        print(board)


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
        "castling_rights": {
            "white_kingside": white_can_castle_kingside,
            "white_queenside": white_can_castle_queenside,
            "black_kingside": black_can_castle_kingside,
            "black_queenside": black_can_castle_queenside,
        },
        "can_claim_draw": can_claim_draw,
    }

    return board_status


def get_input_and_output_sequences(
    game, max_output_sequence_length=10, output_sequence_notation="uci"
):

    # Check output sequence notation
    output_sequence_notation = output_sequence_notation.lower()
    assert output_sequence_notation in [
        "lan",
        "san",
        "uci",
    ], "Output sequences must be in LAN, SAN, or UCI notations!"

    # Get all moves from this game, and board status before each move
    board = game.board()
    all_moves = list()
    board_status_before_moves = list()
    for move_number, move in enumerate(game.mainline_moves()):

        # Get current board status (i.e., before the next move is made)
        board_status_before_moves.append(get_board_status(board))

        # Get this move in the desired output sequence notation
        if output_sequence_notation == "uci":
            all_moves.append(
                board.uci(move).replace("+", "").replace("#", "").replace("x", "")
            )
        elif output_sequence_notation == "lan":
            all_moves.append(
                board.lan(move).replace("+", "").replace("#", "").replace("x", "")
            )
        elif output_sequence_notation == "san":
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

    # TODO: figure out if check (+) checkmate notation (#) should be removed from moves

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
                        move_number : move_number + max_output_sequence_length
                    ],
                }
            )

    return input_and_output_sequences


def build_vocabulary(data_folder, h5_file, vocabulary_file):
    # Open table in H5 file
    h5_file = tb.open_file(os.path.join(data_folder, h5_file), mode="r")
    table = h5_file.root.data

    # Get columns containing output (move) sequences
    output_columns = list()
    for column in table.colnames:
        if column.startswith("output_sequence_"):
            output_columns.append(column)

    # Create move, board position, turn vocabularies (with indices in order of most to least common moves)
    # For moves, an empty string ("") indicates a "no move" after a draw or loss is declared
    move_count = Counter()
    board_position_count = Counter()
    turn_count = Counter()
    for i in tqdm(range(table.nrows), "Accumulating moves"):
        move_count.update(list(table[i][output_columns]))
        board_position_count.update(
            list(table[i]["board_position"].decode())
        )  # strings are bytestrings in H5 files, not unicode
        turn_count.update([table[i]["turn"]])

    vocabulary = {
        "output_sequence": {},
        "board_position": {},
        "turn": {},
        "white_kingside_castling_rights": {False: 0, True: 1},
        "white_queenside_castling_rights": {False: 0, True: 1},
        "black_kingside_castling_rights": {False: 0, True: 1},
        "black_queenside_castling_rights": {False: 0, True: 1},
        "can_claim_draw": {False: 0, True: 1},
    }
    for i, move in enumerate(dict(move_count.most_common()).keys()):
        vocabulary["output_sequence"][move.decode()] = i
    for i, board_position in enumerate(dict(board_position_count.most_common()).keys()):
        vocabulary["board_position"][board_position] = i
    for i, turn in enumerate(dict(turn_count.most_common()).keys()):
        vocabulary["turn"][turn.decode()] = i

    print(
        "There are %d moves in the vocabulary, not including loss or draw declarations. Mathematically, there are only 1968 possible UCI moves."
        % (len(vocabulary["output_sequence"]) - 3)
    )

    # Save vocabulary to file
    with open(os.path.join(data_folder, vocabulary_file), "w") as j:
        json.dump(vocabulary, j, indent=4)

    print("\nSaved to file.\n")

    # Close H5 file
    h5_file.close()
