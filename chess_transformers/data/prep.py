import os
import json
import argparse
import tables as tb
from tqdm import tqdm

from chess_transformers.configs import import_config
from chess_transformers.data.utils import create_vocabulary, encode, parse_fen


def prepare_data(
    data_folder,
    h5_file,
    max_move_sequence_length,
    expected_rows,
    vocab_file,
    splits_file=None,
    val_split_fraction=None,
):
    """
    Transform raw data (FENs and moves) into a form that can be consumed
    by a neural network.

    This is stored in two tables in an H5 file -- one in its natural
    form for human reference, and the other encoded to indices in a
    vocabulary for consumption by a neural network.

    This also creates the vocabulary and train/val splits, and saves
    these to JSON files. The splits are saved in the form of the index
    to split the H5 data at, so that it is split into the training and
    validation data. The games in each split should be mutually
    exclusive.

    Test sets are created separately from all of the raw data by
    setting either "splits_file" or "val_split_fraction" to None. This
    is because a single test set should be used for all models
    regardless of the training or validation data.

    Args:

        data_folder (str): The folder containing the FEN and move files.

        h5_file (str): The H5 file to be saved.

        max_move_sequence_length (int): The maximum number of future
        moves to save for each FEN.

        expected_rows (int): The expected number of rows in the H5
        tables, for internally optimizing write/read access.

        vocab_file (str): The vocabulary file to be saved.

        splits_file (str): The splits file to be saved. If None, no
        splits will be computed. Defaults to None.

        val_split_fraction (float): The fraction, approximately, at
        which the validation data should begin. If None, no splits will
        be computed. Defaults to None.
    """

    # Get names of files/chunks containing moves and FENs
    moves_files = sorted([f for f in os.listdir(data_folder) if f.endswith(".moves")])
    fens_files = sorted([f for f in os.listdir(data_folder) if f.endswith(".fens")])
    assert len(moves_files) == len(fens_files)
    print("\nMoves and FENs are stored in %d chunk(s).\n" % len(moves_files))

    # Create vocabulary and save to file
    vocabulary = create_vocabulary(data_folder, moves_files)
    with open(os.path.join(data_folder, vocab_file), "w") as j:
        json.dump(vocabulary, j, indent=4)
    print(
        "There are %d moves in the move vocabulary, not including '<move>', '<loss>', or '<pad>' tokens. Mathematically, there are only 1968 possible UCI moves.\n"
        % (len(vocabulary["move_sequence"]) - 3)
    )

    # Create table description for H5 file
    class ChessTable(tb.IsDescription):
        board_position = tb.StringCol(64)
        turn = tb.StringCol(1)
        white_kingside_castling_rights = tb.BoolCol()
        white_queenside_castling_rights = tb.BoolCol()
        black_kingside_castling_rights = tb.BoolCol()
        black_queenside_castling_rights = tb.BoolCol()
        move_sequence = tb.StringCol(
            shape=(max_move_sequence_length + 1), itemsize=8, dflt="<pad>"
        )  # "dflt" doesn't work for some reason
        move_sequence_length = tb.Int8Col()

    # Create table description for HDF5 file
    class EncodedChessTable(tb.IsDescription):
        board_position = tb.Int8Col(shape=(64))
        turn = tb.Int8Col()
        white_kingside_castling_rights = tb.Int8Col()
        white_queenside_castling_rights = tb.Int8Col()
        black_kingside_castling_rights = tb.Int8Col()
        black_queenside_castling_rights = tb.Int8Col()
        move_sequence = tb.Int16Col(shape=(max_move_sequence_length + 1))
        move_sequence_length = tb.Int8Col()

    # Delete H5 file if it already exists; start anew
    if os.path.exists(os.path.join(data_folder, h5_file)):
        os.remove(os.path.join(data_folder, h5_file))

    # Create new H5 file
    h5_file = tb.open_file(
        os.path.join(data_folder, h5_file), mode="w", title="data file"
    )

    # Create table in H5 file
    table = h5_file.create_table("/", "data", ChessTable, expectedrows=expected_rows)

    # Create encoded table in H5 file
    encoded_table = h5_file.create_table(
        "/", "encoded_data", EncodedChessTable, expectedrows=table.nrows
    )

    # Create pointer to next row in these tables
    row = table.row
    encoded_row = encoded_table.row

    # Keep track of row numbers where new games begin
    new_game_index = 0
    new_game_indices = list()

    # Iterate through chunks
    for i in range(len(moves_files)):
        print("Now reading %s and %s...\n" % (moves_files[i], fens_files[i]))

        # Read moves and FENs in this chunk
        all_moves = open(os.path.join(data_folder, moves_files[i]), "r").read()
        all_fens = open(os.path.join(data_folder, fens_files[i]), "r").read()
        all_moves = all_moves.split("\n\n")[:-1]
        all_fens = all_fens.split("\n\n")[:-1]
        assert len(all_moves) == len(all_fens)
        print("There are %d games.\n" % len(all_moves))

        # Iterate through games in this chunk
        for j in tqdm(range(len(all_moves)), desc="Adding rows to table"):
            moves = all_moves[j].split("\n")
            result = moves.pop(-1)
            moves.append("<loss>")  # like an EOS token
            fens = all_fens[j].split("\n")
            assert len(moves) == len(fens)
            start_index = 0 if result == "1-0" else 1

            # Iterate through moves in this game
            for k in range(start_index, len(moves), 2):
                t, b, wk, wq, bk, bq = parse_fen(fens[k])
                ms = (
                    ["<move>"]
                    + moves[k : k + max_move_sequence_length]
                    + ["<pad>"] * ((k + max_move_sequence_length) - len(moves))
                )
                msl = len([m for m in ms if m != "<pad>"]) - 1

                # Board position
                row["board_position"] = b
                encoded_row["board_position"] = encode(b, vocabulary["board_position"])

                # Turn
                row["turn"] = t
                encoded_row["turn"] = encode(t, vocabulary["turn"])

                # Castling rights
                row["white_kingside_castling_rights"] = wk
                row["white_queenside_castling_rights"] = wq
                row["black_kingside_castling_rights"] = bk
                row["black_queenside_castling_rights"] = bq
                encoded_row["white_kingside_castling_rights"] = encode(
                    wk,
                    vocabulary["white_kingside_castling_rights"],
                )
                encoded_row["white_queenside_castling_rights"] = encode(
                    wq,
                    vocabulary["white_queenside_castling_rights"],
                )
                encoded_row["black_kingside_castling_rights"] = encode(
                    bk,
                    vocabulary["black_kingside_castling_rights"],
                )
                encoded_row["black_queenside_castling_rights"] = encode(
                    bq,
                    vocabulary["black_queenside_castling_rights"],
                )

                # Move sequence
                row["move_sequence"] = ms
                encoded_row["move_sequence"] = encode(
                    ms,
                    vocabulary["move_sequence"],
                )

                # Move sequence lengths
                row["move_sequence_length"] = msl
                encoded_row["move_sequence_length"] = msl

                # Add row
                row.append()
                encoded_row.append()
            new_game_index += k + 1
            new_game_indices.append(new_game_index)

        table.flush()
        print("\nA total of %d datapoints have been saved to disk.\n" % table.nrows)

    print("...done.\n")

    # Get indices to split at
    if splits_file is not None and val_split_fraction is not None:
        val_split_index = None
        for i in new_game_indices:
            if val_split_index is None:
                if i / table.nrows >= val_split_fraction:
                    val_split_index = i
            else:
                break
        print(
            "The training set will start at index 0, the validation set at index %d (%2.6f%%).\n"
            % (val_split_index, 100.0 * val_split_index / table.nrows)
        )
        with open(os.path.join(data_folder, splits_file), "w") as j:
            json.dump(
                {"val_split_index": val_split_index},
                j,
                indent=4,
            )

    # Close H5 file
    h5_file.close()


if __name__ == "__main__":
    # Get configuration
    parser = argparse.ArgumentParser()
    parser.add_argument("config_name", type=str, help="Name of configuration file.")
    args = parser.parse_args()
    CONFIG = import_config(args.config_name)

    # Prepare data
    prepare_data(
        data_folder=CONFIG.DATA_FOLDER,
        h5_file=CONFIG.H5_FILE,
        max_move_sequence_length=CONFIG.MAX_MOVE_SEQUENCE_LENGTH,
        expected_rows=CONFIG.EXPECTED_ROWS,
        vocab_file=CONFIG.VOCAB_FILE,
        splits_file=CONFIG.SPLITS_FILE,
        val_split_fraction=CONFIG.VAL_SPLIT_FRACTION,
    )
