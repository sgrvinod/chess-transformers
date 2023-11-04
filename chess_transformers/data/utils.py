import os
import regex
from collections import Counter


FILE_NAMES = ["a", "b", "c", "d", "e", "f", "g", "h"]
RANK_NAMES = ["1", "2", "3", "4", "5", "6", "7", "8"]


def replace_number(match):
    """
    Replaces numbers in a string with as many periods.

    For example, "3" will be replaced by "...".

    Args:

        match (regex.match): A RegEx match for a number.

    Returns:

        str: The replacement string.
    """
    return int(match.group()) * "."


def square_index(square):
    """
    Gets the index of a chessboard square, counted from the top-left
    corner (a8) of the chessboard.

    Args:

        square (str): The square.

    Returns:

        int: The index for this square.
    """
    file = square[0]
    rank = square[1]

    return (7 - RANK_NAMES.index(rank)) * 8 + FILE_NAMES.index(file)


def assign_ep_square(board, ep_square):
    """
    Notate a board position with an En Passan square.

    Args:

        board (str): The board position.

        ep_square (str): The En Passant square.

    Returns:

        str: The modified board position.
    """
    i = square_index(ep_square)

    return board[:i] + "," + board[i + 1 :]


def get_castling_rights(castling_rights):
    """
    Get individual color/side castling rights from the FEN notation of
    castling rights.

    Args:

        castling_rights (str): The castling rights component of the FEN
        notation.

    Returns:

        bool: Can white castle kingside?

        bool: Can white castle queenside?

        bool: Can black castle kingside?

        bool: Can black castle queenside?
    """
    white_kingside = "K" in castling_rights
    white_queenside = "Q" in castling_rights
    black_kingside = "k" in castling_rights
    black_queenside = "q" in castling_rights

    return white_kingside, white_queenside, black_kingside, black_queenside


def parse_fen(fen):
    """
    Parse the FEN notation at a given board position.

    Args:

        fen (str): The FEN notation.

    Returns:

        str: The player to move next, one of "w" or "b".

        str: The board position.

        bool: Can white castle kingside?

        bool: Can white castle queenside?

        bool: Can black castle kingside?

        bool: Can black castle queenside?
    """
    board, turn, castling_rights, ep_square, _, __ = fen.split()
    board = regex.sub(r"\d", replace_number, board.replace("/", ""))
    if ep_square != "-":
        board = assign_ep_square(board, ep_square)
    (
        white_kingside,
        white_queenside,
        black_kingside,
        black_queenside,
    ) = get_castling_rights(castling_rights)

    return turn, board, white_kingside, white_queenside, black_kingside, black_queenside


def create_vocabulary(data_folder, moves_files):
    """
    Create a vocabulary for all categorical variables.

    Note that only creating the move vocabulary is non-trivial; the rest
    can be written manually.

    Args:

        data_folder (str): The folder containing the move files.

        moves_files (str): The names of the move files.

    Returns:

        dict: The constructed vocabulary.
    """

    # Vocabulary
    vocabulary = {
        "move_sequence": {},
        "board_position": dict(
            zip(
                [".", ",", "P", "p", "R", "r", "N", "n", "B", "b", "Q", "q", "K", "k"],
                range(14),
            )
        ),
        "turn": {"b": 0, "w": 1},
        "white_kingside_castling_rights": {False: 0, True: 1},
        "white_queenside_castling_rights": {False: 0, True: 1},
        "black_kingside_castling_rights": {False: 0, True: 1},
        "black_queenside_castling_rights": {False: 0, True: 1},
    }

    # Create move vocabulary
    move_count = Counter()
    for i in range(len(moves_files)):
        move_count.update(
            "\n".join(
                open(os.path.join(data_folder, moves_files[i]), "r")
                .read()
                .split("\n\n")
            ).split("\n")
        )
    del move_count["1-0"], move_count["0-1"], move_count[""]
    for i, move in enumerate(dict(move_count.most_common()).keys()):
        vocabulary["move_sequence"][move] = i
    vocabulary["move_sequence"]["<move>"] = i + 1
    vocabulary["move_sequence"]["<loss>"] = i + 2
    vocabulary["move_sequence"]["<pad>"] = i + 3

    return vocabulary


def encode(item, vocabulary):
    """
    Encode an item with its index in the vocabulary its from.

    Args:

        item (list, str, bool): The item.

        vocabulary (dict): The vocabulary.

    Raises:

        NotImplementedError: If the item is not one of the types
        specified above.

    Returns:

        list, str: The item, encoded.
    """
    if isinstance(item, list):  # move sequence
        return [vocabulary[it] for it in item]
    elif isinstance(item, str):  # turn or board position
        return (
            vocabulary[item] if item in vocabulary else [vocabulary[it] for it in item]
        )
    elif isinstance(item, bool):  # castling rights
        return vocabulary[item]
    else:
        raise NotImplementedError
