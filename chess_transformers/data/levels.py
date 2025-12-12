"""
Mappings for chess board representation and moves.

This module defines constants and mappings used to represent the chess board,
pieces, turns, and moves in a format suitable for the chess transformer model.

Key Features:
    - Board coordinates: FILES (a-h) and RANKS (1-8).
    - Piece-to-integer mappings: PIECES dictionary for encoding pieces.
    - Turn indicators: TURN dictionary for encoding whose turn it is.
    - Square mappings: SQUARES dictionary mapping algebraic notation to indices.
    - UCI move mappings: UCI_MOVES dictionary for encoding all legal moves.
    - Boolean mapping: BOOL dictionary for binary flags.

Notes:
    The UCI_MOVES mapping includes all possible moves in UCI format plus
    three special tokens: <move>, <loss>, and <pad>.
"""

from typing import Dict

FILES = ["a", "b", "c", "d", "e", "f", "g", "h"]
RANKS = ["1", "2", "3", "4", "5", "6", "7", "8"]
PIECES = {
    ".": 0,
    ",": 1,
    "P": 2,
    "p": 3,
    "R": 4,
    "r": 5,
    "N": 6,
    "n": 7,
    "B": 8,
    "b": 9,
    "Q": 10,
    "q": 11,
    "K": 12,
    "k": 13,
}
TURN = {"b": 0, "w": 1}


def _generate_squares() -> Dict[str, int]:
    """Generate the SQUARES mapping.

    Returns:
        A dictionary mapping chess squares (e.g., "a1", "e4") to
        integer indices from 0 to 63.
    """
    squares = {}
    for rank_index in range(7, -1, -1):  # ranks 8 to 1
        for file_index in range(8):  # files a to h
            square = f"{FILES[file_index]}{RANKS[rank_index]}"
            squares[square] = len(squares)
    return squares


SQUARES = _generate_squares()


def _generate_uci_moves() -> Dict[str, int]:
    """Generate the UCI_MOVES mapping.

    The mapping includes:
        1. Queen moves (sliding) + King + Pawn non-promotion moves.
        2. Knight moves.
        3. Promotion moves.
        4. Special tokens: <move>, <loss>, <pad>.

    Returns:
        A dictionary mapping UCI moves (e.g., "e2e4", "a7a8q") to
        integer indices.
    """
    moves = []

    # 1. Queen Moves (Sliding) + King + Pawn non-promotion
    # Source: File major (a1..a8, b1..b8...)
    for source_file_index in range(8):
        for source_rank_index in range(8):
            source_square = f"{FILES[source_file_index]}{RANKS[source_rank_index]}"

            # Destination: Rank 7..0, File 7..0
            for destination_rank_index in range(7, -1, -1):
                for destination_file_index in range(7, -1, -1):
                    if (
                        source_file_index == destination_file_index
                        and source_rank_index == destination_rank_index
                    ):
                        continue

                    file_difference = abs(destination_file_index - source_file_index)
                    rank_difference = abs(destination_rank_index - source_rank_index)

                    # Check Queen move (horizontal, vertical, diagonal)
                    if (
                        file_difference == 0
                        or rank_difference == 0
                        or file_difference == rank_difference
                    ):
                        moves.append(
                            f"{source_square}"
                            f"{FILES[destination_file_index]}"
                            f"{RANKS[destination_rank_index]}"
                        )

    # 2. Knight Moves
    for source_file_index in range(8):
        for source_rank_index in range(8):
            source_square = f"{FILES[source_file_index]}{RANKS[source_rank_index]}"

            for destination_rank_index in range(7, -1, -1):
                for destination_file_index in range(7, -1, -1):
                    file_difference = abs(destination_file_index - source_file_index)
                    rank_difference = abs(destination_rank_index - source_rank_index)

                    if file_difference * rank_difference == 2:
                        moves.append(
                            f"{source_square}"
                            f"{FILES[destination_file_index]}"
                            f"{RANKS[destination_rank_index]}"
                        )

    # 3. Promotions
    # Source: File a..h, Rank 7 (White) then Rank 2 (Black) -> Indices 6 and 1
    for source_file_index in range(8):
        # Rank 7 (index 6) and Rank 2 (index 1)
        for source_rank_index in [6, 1]:
            source_square = f"{FILES[source_file_index]}{RANKS[source_rank_index]}"
            direction = 1 if source_rank_index == 6 else -1
            destination_rank_index = source_rank_index + direction

            possible_destinations = []

            # Captures: File descending
            for destination_file_index in range(7, -1, -1):
                if abs(destination_file_index - source_file_index) == 1:
                    possible_destinations.append(
                        f"{FILES[destination_file_index]}"
                        f"{RANKS[destination_rank_index]}"
                    )

            # Push
            possible_destinations.append(
                f"{FILES[source_file_index]}{RANKS[destination_rank_index]}"
            )

            for destination_square in possible_destinations:
                for promotion_piece in ["q", "r", "b", "n"]:
                    moves.append(
                        f"{source_square}{destination_square}" f"{promotion_piece}"
                    )

    uci_dict = {m: i for i, m in enumerate(moves)}

    # Add special tokens: <move>, <loss>, <pad>
    special_start = len(moves)
    uci_dict["<move>"] = special_start
    uci_dict["<loss>"] = special_start + 1
    uci_dict["<pad>"] = special_start + 2

    return uci_dict


UCI_MOVES = _generate_uci_moves()

BOOL = {False: 0, True: 1}
