import chess.pgn
import random
import chess
import json
import os
from tqdm import tqdm
from multiprocessing import Pool

random.seed(1234)


def parse_pgn_data(pgn_data_path):

    # Get names of PGN files
    pgn_files = [f for f in os.listdir(pgn_data_path) if f.endswith(".pgn")]
    print("There are %d PGN files to parse." % len(pgn_files))

    for pgn_file in sorted(pgn_files):

        pgn_file_path = os.path.join(pgn_data_path, pgn_file)
        parse_pgn_file(pgn_file_path)


def parse_pgn_file(
    pgn_file_path, max_output_sequence_length=7, output_sequence_notation="lan"
):

    # Check output sequence notation
    output_sequence_notation = output_sequence_notation.lower()
    assert output_sequence_notation in [
        "lan",
        "san",
        "uci",
    ], "Output sequences must be in LAN, SAN, or UCI notations!"

    # Path of output JSON file
    parsed_json_file_path = pgn_file_path.replace(".pgn", ".json")

    # Only proceed if this PGN file hasn't already been parsed
    if not os.path.exists(parsed_json_file_path):
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

        # Save to a JSON file
        with open(parsed_json_file_path, "w") as j:
            json.dump(input_and_output_sequences, j, indent=4)
        print(
            "%d games were parsed from %s, and saved to %s"
            % (len(games), pgn_file_path, parsed_json_file_path)
        )

    else:
        print("Skipping %s - already parsed" % pgn_file_path)


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
            "black_kingslide": black_can_castle_kingside,
            "black_queenside": black_can_castle_queenside,
        },
        "can_claim_draw": can_claim_draw,
    }

    return board_status


def get_input_and_output_sequences(
    game, max_output_sequence_length, output_sequence_notation
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
            all_moves.append(board.uci(move))
        elif output_sequence_notation == "lan":
            all_moves.append(board.lan(move))
        elif output_sequence_notation == "san":
            all_moves.append(board.san(move))
        else:
            raise NotImplementedError

        # Make the move on the board
        board.push(move)

    # If game ended in checkmate, denote the loss as a "move" by the losing player
    # This is to condition the network to recognize the checkmate and to end the sequence with this move (like an "end generation" token)
    if "#" in all_moves[-1]:
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

    return input_and_output_sequences, winner
