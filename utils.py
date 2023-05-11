import chess.pgn
import random
import chess
import os
from tqdm import tqdm
from multiprocessing import Pool

random.seed(1234)


def parse_pgn_data(pgn_data_path):
    pgn_files = [f for f in os.listdir(pgn_data_path) if f.endswith(".pgn")]
    pgn_file_paths = [os.path.join(pgn_data_path, pgn_file) for pgn_file in pgn_files]

    print("There are %d PGN files to parse." % len(pgn_files))

    games = list()
    for pgn_file in tqdm(sorted(pgn_files), desc="Parsing"):
        pgn_file_path = os.path.join(pgn_data_path, pgn_file)
        games.extend(parse_pgn_file(pgn_file_path))

    print(
        "A total of %d games were parsed and extracted from these files." % len(games)
    )

    return games


def parse_pgn_file(pgn_file_path):
    pgn_file_contents = open(pgn_file_path)
    games = list()
    there_are_more_games = True
    while there_are_more_games:
        game = chess.pgn.read_game(pgn_file_contents)
        there_are_more_games = game is not None
        if there_are_more_games:
            games.append(game)

    print(
        "%d games were parsed and extracted from %s."
        % (len(games), pgn_file_path.split("/")[-1])
    )

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
        # Get board position before this move
        ranks = str(board).splitlines()
        ranks.reverse()
        board_position = " ".join(ranks).replace(" ", "").split()

        # If there is a square that can be attacked with a legal en-passant move, indicate in board position
        if board.ep_square is not None and board.has_legal_en_passant():
            board_position[board.ep_square] = ","

        # Get player who should move in the current turn
        turn = "w" if board.turn else "b"

        # Get castling rights
        white_can_castle_kingside = board.has_kingside_castling_rights(chess.WHITE)
        white_can_castle_queenside = board.has_queenside_castling_rights(chess.WHITE)
        black_can_castle_kingside = board.has_kingside_castling_rights(chess.BLACK)
        black_can_castle_queenside = board.has_queenside_castling_rights(chess.BLACK)

        # Check if a draw can be claim in the current turn
        can_claim_draw = (
            board.can_claim_fifty_moves() or board.can_claim_threefold_repetition()
        )

        # Compile board status
        board_status_before_moves.append(
            {
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
        )

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

    # Find winner because we want input and output sequences for the winner's moves only
    result = game.headers["Result"]
    if result == "1-0":
        winner = "w"
    elif result == "0-1":
        winner = "b"
    elif result == "1/2-1/2":
        winner = random.choice("w", "b")
        all_moves.append("<draw>")  # assumes player at that turn offered draw
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


if __name__ == "__main__":
    # games = parse_pgn_data("/media/sgr/SSD/lichess data/")
    games = parse_pgn_file("/media/sgr/SSD/lichess data/lichess_elite_2013-09.pgn")

    game = games[0]
    view_game(games[0])
    sequences, winner = get_input_and_output_sequences(
        games[0], max_output_sequence_length=7, output_sequence_notation="lan"
    )
