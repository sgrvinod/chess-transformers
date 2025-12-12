"""Process chess game data into LMDB format for training.

This module reads chess games from pre-processed .moves and .fens files,
encodes them into a compact binary format, and stores them in an LMDB
database for efficient training data access.

The data pipeline expects games that ended in checkmate, with the winner's
moves extracted for supervised learning. Each datapoint consists of a board
position (encoded from FEN) and the winning player's next move (as from/to
squares).

Usage:
    Set the CT_DATA_FOLDER environment variable to the parent folder
    containing dataset subfolders, then run::

        python -m chess_transformers.data.process <config_name>

    The script will look for .moves and .fens files in
    CT_DATA_FOLDER/<config_name>/ and create the LMDB there.

Example:
    >>> export CT_DATA_FOLDER=/path/to/data
    >>> python -m chess_transformers.data.process LE25ct
    # Reads from /path/to/data/LE25ct/*.moves and *.fens
    # Creates /path/to/data/LE25ct/LE25ct.lmdb

Key Features:
    - Encodes board positions as 64-byte arrays (one byte per square).
    - Optionally stores legal move masks for label smoothing during training.
    - Supports train/validation splitting at game boundaries.

Notes:
    When legal_mask_mode is enabled, each record grows by 16 bytes to store
    two 64-bit masks (one for legal from-squares, one for legal to-squares).
"""

import os
import chess
import struct
import argparse

from tqdm import tqdm
from pathlib import Path
from typing import Optional, Tuple, Union

from chess_transformers.utilities.lmdb import ChessLMDB
from chess_transformers.utilities.loggers import setup_logger
from chess_transformers.utilities.configs import import_config
from chess_transformers.data.configs.base import LegalMaskMode
from chess_transformers.data.levels import RANKS, FILES, TURN, PIECES, SQUARES, BOOL


# Logger
logger = setup_logger(__file__)

# Precompiled patterns and lookup tables for performance
_DIGIT_EXPAND = str.maketrans({str(i): "." * i for i in range(1, 9)})
_FILE_INDEX = {f: i for i, f in enumerate(FILES)}
_RANK_INDEX = {r: i for i, r in enumerate(RANKS)}

# Struct formats for record packing
# Base: 64 bytes (board) + 7 bytes (turn, castling, from, to) = 71 bytes
# With masks: + 16 bytes (two 64-bit masks) = 87 bytes
_RECORD_STRUCT_BASE = struct.Struct("64B B B B B B B B")
_RECORD_STRUCT_WITH_MASKS = struct.Struct("64B B B B B B B B Q Q")


def square_index(square: str) -> int:
    """Convert algebraic square notation to a linear board index.

    Converts a chess square in algebraic notation (e.g., "e4") to a
    0-based linear index where a8=0, b8=1, ..., h1=63. This matches
    the order of squares when reading a FEN board string from left
    to right, top to bottom.

    Args:
        square: A two-character string representing a chess square
            in algebraic notation (e.g., "a1", "e4", "h8").

    Returns:
        int: A linear index from 0 to 63, where 0 is a8 and 63 is h1.

    Example:
        >>> square_index("a8")
        0
        >>> square_index("h1")
        63
        >>> square_index("e4")
        36
    """
    return (7 - _RANK_INDEX[square[1]]) * 8 + _FILE_INDEX[square[0]]


def assign_ep_square(board: str, ep_square: str) -> str:
    """Mark the en passant square on a board string.

    Replaces the character at the en passant square position with a
    comma (",") to indicate that en passant capture is available at
    that square.

    Args:
        board: A 64-character string representing the board state,
            with pieces as letters and empty squares as dots.
        ep_square: A two-character string in algebraic notation
            indicating the en passant target square (e.g., "e3").

    Returns:
        str: The board string with the en passant square replaced
            by a comma character.

    Example:
        >>> board = "." * 64  # Empty board
        >>> assign_ep_square(board, "e3")  # Mark e3 as EP square
        '..........................................,....................'
    """
    i = square_index(ep_square)

    return board[:i] + "," + board[i + 1 :]  # noqa: E203


def get_castling_rights(castling_rights: str) -> Tuple[bool, bool, bool, bool]:
    """Parse FEN castling rights string into boolean flags.

    Extracts the four castling availability flags from the FEN
    castling rights field.

    Args:
        castling_rights: A string from FEN notation indicating available
            castling moves. Contains any combination of "K" (white kingside),
            "Q" (white queenside), "k" (black kingside), "q" (black queenside),
            or "-" if no castling is available.

    Returns:
        Tuple[bool, bool, bool, bool]: A tuple of four booleans:
            - white_kingside: True if white can castle kingside.
            - white_queenside: True if white can castle queenside.
            - black_kingside: True if black can castle kingside.
            - black_queenside: True if black can castle queenside.

    Example:
        >>> get_castling_rights("KQkq")
        (True, True, True, True)
        >>> get_castling_rights("Kq")
        (True, False, False, True)
        >>> get_castling_rights("-")
        (False, False, False, False)
    """
    white_kingside = "K" in castling_rights
    white_queenside = "Q" in castling_rights
    black_kingside = "k" in castling_rights
    black_queenside = "q" in castling_rights

    return white_kingside, white_queenside, black_kingside, black_queenside


def parse_fen(fen: str) -> Tuple[str, str, bool, bool, bool, bool]:
    """Parse a FEN string into board state components.

    Extracts and processes all relevant fields from a FEN (Forsyth-Edwards
    Notation) string, converting the board representation to a 64-character
    string and extracting castling rights and turn information.

    Args:
        fen: A complete FEN string with all six fields:
            "<board> <turn> <castling> <ep_square> <halfmove> <fullmove>".

    Returns:
        tuple: A 6-tuple containing:
            - turn (str): "w" for white to move, "b" for black.
            - board (str): 64-character string with pieces as letters,
                empty squares as ".", and en passant square as ",".
            - white_kingside (bool): White can castle kingside.
            - white_queenside (bool): White can castle queenside.
            - black_kingside (bool): Black can castle kingside.
            - black_queenside (bool): Black can castle queenside.

    Example:
        >>> fen = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"
        >>> turn, board, wk, wq, bk, bq = parse_fen(fen)
        >>> turn
        'b'
        >>> len(board)
        64
    """
    board, turn, castling_rights, ep_square, _, __ = fen.split()
    board = board.replace("/", "").translate(_DIGIT_EXPAND)
    if ep_square != "-":
        board = assign_ep_square(board, ep_square)
    (
        white_kingside,
        white_queenside,
        black_kingside,
        black_queenside,
    ) = get_castling_rights(castling_rights)

    return turn, board, white_kingside, white_queenside, black_kingside, black_queenside


def _compute_legal_masks(
    fen: str,
    from_sq: int,
    mode: LegalMaskMode,
) -> Tuple[int, int]:
    """Compute legal move masks for a given position.

    Args:
        fen: The FEN string representing the board position.
        from_sq: The ground-truth from-square index (0-63).
        mode: The legal mask mode ("disjoint" or "conditional").

    Returns:
        Tuple[int, int]: Two 64-bit integers representing:
            - legal_from_mask: Bitmask of squares with legal moves.
            - legal_to_mask: Bitmask of legal destination squares.
              If mode is "conditional", only destinations for from_sq.
              If mode is "disjoint", all legal destinations.

    Note:
        The bitmask uses bit index matching the square index: bit 0 = a8,
        bit 63 = h1. This matches the square_index() convention and the
        SQUARES dict in levels.py.
    """
    # Indexing convention: bit 0 = a8, bit 63 = h1
    # This matches square_index() and the SQUARES dict in levels.py
    board = chess.Board(fen)
    legal_moves = list(board.legal_moves)

    legal_from_mask = 0
    legal_to_mask = 0

    for move in legal_moves:
        # python-chess uses 0=a1, 63=h8, but we use 0=a8, 63=h1
        # Convert using: our_sq = (7 - rank) * 8 + file
        # where rank = chess_sq // 8 and file = chess_sq % 8
        chess_from = move.from_square
        chess_to = move.to_square

        # Convert to our indexing
        our_from = (7 - (chess_from // 8)) * 8 + (chess_from % 8)
        our_to = (7 - (chess_to // 8)) * 8 + (chess_to % 8)

        # Set from-square bit
        legal_from_mask |= 1 << our_from

        # Set to-square bit based on mode
        # - disjoint: all legal destinations
        # - conditional: only destinations for the target from-square
        if mode == "disjoint" or (mode == "conditional" and our_from == from_sq):
            legal_to_mask |= 1 << our_to

    return legal_from_mask, legal_to_mask


def process_data(
    data_folder: Union[str, Path],
    output_filename: str,
    expected_rows: int,
    val_split_fraction: Optional[float],
    legal_mask_mode: Optional[LegalMaskMode] = None,
) -> None:
    """Process chess game data files into an LMDB database.

    Reads .moves and .fens files from the data folder, extracts winning
    player's moves from checkmate games, encodes board positions and moves
    into a compact binary format, and stores them in an LMDB database.

    Each record is 71 bytes without masks, or 87 bytes with legal masks:
    64 bytes for the board, 7 bytes for metadata (turn, castling, from/to),
    and optionally 16 bytes for two 64-bit legal move masks.

    The function only processes moves made by the winning player, as the
    dataset is intended for learning from winning play.

    Args:
        data_folder: Path to the directory containing .moves and .fens
            files. Files should be named consistently so that sorting
            pairs them correctly (e.g., chunk_001.moves, chunk_001.fens).
        output_filename: Name of the LMDB file to create (e.g., "data.lmdb").
            The file will be created inside data_folder.
        expected_rows: Estimated number of datapoints, used to calculate
            the LMDB map size. The actual map size is set to
            expected_rows * record_size * 1.5 bytes.
        val_split_fraction: Fraction (0.0 to 1.0) at which the validation
            split begins. For example, 0.98 means training data is indices
            0 to ~98%, and validation is ~98% to 100%. The actual split
            split point is adjusted to fall on a game boundary.
        legal_mask_mode: Mode for storing legal move masks. Options are:
            None (no masks), "disjoint" (all legal from/to squares), or
            "conditional" (legal to-squares for target from-square only).

    Raises:
        FileNotFoundError: If no .moves files are found in data_folder.
        FileExistsError: If the output LMDB file already exists.
        ValueError: If the number of .moves and .fens files don't match,
            or if a chunk has mismatched game counts between moves and FENs.

    Note:
        Games are excluded if:
        - The number of moves doesn't match the number of FENs.
        - The recorded result doesn't match the move count parity
          (indicates data corruption).
    """
    # Get names of files/chunks containing moves and FENs
    data_folder = Path(data_folder)
    moves_files = sorted(data_folder.glob("*.moves"))
    fens_files = sorted(data_folder.glob("*.fens"))
    if not moves_files:
        logger.critical(f"No .moves files found in {data_folder}")
        raise FileNotFoundError(f"No .moves files found in {data_folder}")
    if len(moves_files) != len(fens_files):
        logger.critical("Moves and FENs are not stored in the same number of chunks")
        raise ValueError("Moves and FENs are not stored in the same number of chunks")
    logger.info(f"Moves and FENs are stored in {len(moves_files)} chunks")

    # Determine record size based on whether we're storing legal masks
    # Base: 71 bytes, With masks: 87 bytes (adds two 64-bit masks)
    record_size = 87 if legal_mask_mode else 71
    record_struct = (
        _RECORD_STRUCT_WITH_MASKS if legal_mask_mode else _RECORD_STRUCT_BASE
    )

    # LMDB map size (allow for some overhead)
    map_size = int(expected_rows * record_size * 1.5)

    # Open LMDB environment using the helper class
    db_path = data_folder / output_filename
    if db_path.exists():
        logger.critical(f"LMDB file {db_path} already exists")
        raise FileExistsError(f"LMDB file {db_path} already exists")

    lmdb_db = ChessLMDB(
        path=db_path,
        readonly=False,
        map_size=map_size,
    )

    # Keep track of global index
    global_index = 0
    new_game_indices = []

    # Keep track of errors
    n_wrong_results = 0
    n_move_fen_mismatches = 0
    n_malformed_moves = 0

    # Use the batch writer for efficient writes with periodic commits
    with lmdb_db.write(commit_every=1000000) as write:
        # Iterate through chunks
        for i in range(len(moves_files)):
            logger.info(
                f"Now reading {moves_files[i].name} and {fens_files[i].name}..."
            )

            # Read moves and FENs in this chunk
            all_moves = moves_files[i].read_text().split("\n\n")[:-1]
            all_fens = fens_files[i].read_text().split("\n\n")[:-1]

            if len(all_moves) != len(all_fens):
                logger.critical(
                    "Moves and FENs are not stored in the same number of games"
                )
                raise ValueError(
                    "Moves and FENs are not stored in the same number of games"
                )
            logger.info(f"There are {len(all_moves)} games in this chunk")

            # Iterate through games in this chunk
            for j in tqdm(
                range(len(all_moves)), desc=f"Adding rows to LMDB (Chunk {i+1})"
            ):
                moves = all_moves[j].split("\n")
                result = moves.pop(-1)
                moves = [move.lower() for move in moves]
                fens = all_fens[j].split("\n")

                # Ignore game if there is a mismatch between moves and FENs
                # There should be one more FEN than moves because:
                # - FENs include the starting position and position after each move
                # - The final FEN is the checkmate position (no move follows it)
                if len(fens) != len(moves) + 1:
                    n_move_fen_mismatches += 1
                    continue

                start_index = 0 if result == "1-0" else 1

                # Ignore this game if the wrong result is recorded in the source file
                if len(moves) % 2 == start_index:
                    n_wrong_results += 1
                    continue

                # Iterate through moves in this game
                # We only process the turns of the winner
                game_start_index = global_index

                for k in range(start_index, len(moves), 2):
                    t, b, wk, wq, bk, bq = parse_fen(fens[k])

                    # Parse move (e.g. "e2e4" -> from="e2", to="e4")
                    # Promotion is usually 5 chars e.g. "a7a8q", but ignore promotion piece
                    move = moves[k]
                    from_sq_str = move[:2]
                    to_sq_str = move[2:4]

                    # Validate squares and skip malformed moves
                    try:
                        from_sq = SQUARES[from_sq_str]
                        to_sq = SQUARES[to_sq_str]
                    except KeyError:
                        logger.warning(f"Malformed move '{move}', skipping")
                        n_malformed_moves += 1
                        continue

                    # Encode and pack data
                    if legal_mask_mode:
                        # Compute legal move masks
                        legal_from_mask, legal_to_mask = _compute_legal_masks(
                            fens[k], from_sq, legal_mask_mode
                        )
                        packed_data = record_struct.pack(
                            *[PIECES[c] for c in b],  # board (64 bytes)
                            TURN[t],  # turn
                            BOOL[wk],  # white kingside
                            BOOL[wq],  # white queenside
                            BOOL[bk],  # black kingside
                            BOOL[bq],  # black queenside
                            from_sq,  # from square
                            to_sq,  # to square
                            legal_from_mask,  # 64-bit from mask
                            legal_to_mask,  # 64-bit to mask
                        )
                    else:
                        packed_data = record_struct.pack(
                            *[PIECES[c] for c in b],  # board (64 bytes)
                            TURN[t],  # turn
                            BOOL[wk],  # white kingside
                            BOOL[wq],  # white queenside
                            BOOL[bk],  # black kingside
                            BOOL[bq],  # black queenside
                            from_sq,  # from square
                            to_sq,  # to square
                        )

                    # Write to LMDB via batch writer
                    write(global_index, packed_data)
                    global_index += 1

                new_game_indices.append(game_start_index)

    logger.info(f"A total of {global_index} datapoints have been saved to {db_path}")

    if n_move_fen_mismatches > 0:
        logger.warning(
            f"{n_move_fen_mismatches} game(s) excluded because number of moves "
            "and FENs did not match"
        )
    if n_wrong_results > 0:
        logger.warning(
            f"{n_wrong_results} game(s) excluded that had the wrong result recorded"
        )
    if n_malformed_moves > 0:
        logger.warning(f"{n_malformed_moves} move(s) skipped due to malformed data")

    # Compute split index
    val_split_index = None
    if val_split_fraction is not None:
        target_count = int(global_index * val_split_fraction)
        # Find the game boundary closest to the target count
        # We want to ensure we don't split in the middle of a game, although
        # with "random row access" training, strictly speaking, it matters less if games are shuffled,
        # but keeping games distinct in train/val is good practice to avoid leakage.

        # Since we processed sequentially, new_game_indices are sorted.
        # We can just find the index in new_game_indices that is closest to target_count.

        # Simple search
        for idx in new_game_indices:
            if idx >= target_count:
                val_split_index = idx
                break

        if val_split_index is None:
            val_split_index = global_index  # Use all for train if split is at end

        if global_index > 0:
            val_pct = 100.0 * val_split_index / global_index
            logger.info(
                f"The training set will start at index 0, the validation set "
                f"at index {val_split_index} ({val_pct:.2f}%)."
            )

    # Save metadata **inside** the LMDB under a reserved key
    if legal_mask_mode:
        schema = (
            "board(64B), turn(1B), wk(1B), wq(1B), bk(1B), bq(1B), "
            "from(1B), to(1B), legal_from(8B), legal_to(8B)"
        )
    else:
        schema = (
            "board(64B), turn(1B), wk(1B), wq(1B), bk(1B), bq(1B), " "from(1B), to(1B)"
        )
    metadata = {
        "length": global_index,
        "val_split_index": val_split_index,
        "legal_mask_mode": legal_mask_mode,
        "schema": schema,
    }
    lmdb_db.write_metadata(metadata)

    # Close the database
    lmdb_db.close()
    logger.info(f"LMDB database created successfully at {db_path}")


if __name__ == "__main__":
    # Get configuration
    parser = argparse.ArgumentParser()
    parser.add_argument("config_name", type=str, help="Name of configuration file.")
    args = parser.parse_args()

    # Import config module
    config = import_config(args.config_name)

    # Get actual data folder path from env var + config name
    data_root = os.environ.get("CT_DATA_FOLDER")
    if not data_root:
        logger.critical("Environment variable CT_DATA_FOLDER not set")
        raise EnvironmentError("Environment variable CT_DATA_FOLDER not set")

    data_folder = Path(data_root) / config.name
    if not data_folder.is_dir():
        logger.critical(f"Data folder does not exist: {data_folder}")
        raise FileNotFoundError(f"Data folder does not exist: {data_folder}")

    # Process data
    process_data(
        data_folder=data_folder,
        output_filename=config.lmdb_filename,
        expected_rows=config.expected_rows,
        val_split_fraction=config.val_split_fraction,
        legal_mask_mode=config.legal_mask_mode,
    )
