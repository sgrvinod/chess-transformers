import os
import json
import argparse
import numpy as np
import tables as tb
from tqdm import tqdm
from importlib import import_module


def encode(item, vocabulary):
    """
    Encode an item with its index in the vocabulary its from.

    Args:

        item (list, np.bytes_, str, np.bool_): The item.

        vocabulary (dict): The vocabulary.

    Raises:

        NotImplementedError: If the item is not one of the types
        specified above.

    Returns:

        list, str: The item, encoded.
    """
    if isinstance(item, list):  # move sequence
        return [vocabulary[it.decode()] for it in item]
    elif isinstance(item, np.bytes_) or isinstance(item, str):  # turn or board position
        item = item.decode() if isinstance(item, np.bytes_) else item
        return (
            vocabulary[item] if item in vocabulary else [vocabulary[it] for it in item]
        )
    elif isinstance(item, np.bool_) or isinstance(
        item, bool
    ):  # castling rights and draw potential
        return vocabulary[str(item).lower()]
    else:
        raise NotImplementedError


def encode_data(data_folder, h5_file, vocab_file):
    """
    Encode the data in the H5 file.

    Args:

        data_folder (str): The folder with the H5 file.

        h5_file (str): The H5 file.

        vocab_file (str): The vocabulary file.
    """
    # Load vocabularies
    vocabulary = json.load(open(os.path.join(data_folder, vocab_file), "r"))

    # Open table in H5 file
    h5_file = tb.open_file(os.path.join(data_folder, h5_file), mode="a")
    table = h5_file.root.data
    print("\n")

    # Create table description for HDF5 file
    class EncodedChessTable(tb.IsDescription):
        board_position = tb.Int8Col(shape=(64))
        turn = tb.Int8Col()
        white_kingside_castling_rights = tb.Int8Col()
        white_queenside_castling_rights = tb.Int8Col()
        black_kingside_castling_rights = tb.Int8Col()
        black_queenside_castling_rights = tb.Int8Col()
        can_claim_draw = tb.Int8Col()
        move_sequence = tb.Int16Col(shape=(table[0]["move_sequence"].shape[0] + 1))
        move_sequence_length = tb.Int8Col()

    # Create table in encoded HDF5 file
    encoded_table = h5_file.create_table(
        "/", "encoded_data", EncodedChessTable, expectedrows=table.nrows
    )

    # Create pointer to next row in this table
    row = encoded_table.row

    # Write parsed datapoints to the HDF5 file
    for i in tqdm(range(table.nrows), desc="Encoding"):
        row["board_position"] = encode(
            table[i]["board_position"], vocabulary=vocabulary["board_position"]
        )
        row["turn"] = encode(table[i]["turn"], vocabulary=vocabulary["turn"])
        row["white_kingside_castling_rights"] = encode(
            table[i]["white_kingside_castling_rights"],
            vocabulary=vocabulary["white_kingside_castling_rights"],
        )
        row["white_queenside_castling_rights"] = encode(
            table[i]["white_queenside_castling_rights"],
            vocabulary=vocabulary["white_queenside_castling_rights"],
        )
        row["black_kingside_castling_rights"] = encode(
            table[i]["black_kingside_castling_rights"],
            vocabulary=vocabulary["black_kingside_castling_rights"],
        )
        row["black_queenside_castling_rights"] = encode(
            table[i]["black_queenside_castling_rights"],
            vocabulary=vocabulary["black_queenside_castling_rights"],
        )
        row["can_claim_draw"] = encode(
            table[i]["can_claim_draw"], vocabulary=vocabulary["can_claim_draw"]
        )
        row["move_sequence"] = encode(
            [b"<move>"] + table[i]["move_sequence"].tolist(),
            vocabulary=vocabulary["move_sequence"],
        )
        row["move_sequence_length"] = len(
            [o for o in table[i]["move_sequence"].tolist() if o != b"<pad>"]
        )
        row.append()
    encoded_table.flush()
    assert encoded_table.nrows == table.nrows

    print("\nA total of %d datapoints have been saved to disk.\n" % encoded_table.nrows)

    h5_file.close()


if __name__ == "__main__":
    # Get configuration
    parser = argparse.ArgumentParser()
    parser.add_argument("config_name", type=str, help="Name of configuration file.")
    args = parser.parse_args()
    CONFIG = import_module("configs.{}".format(args.config_name))

    # Encode data
    encode_data(
        data_folder=CONFIG.DATA_FOLDER,
        h5_file=CONFIG.H5_FILE,
        vocab_file=CONFIG.VOCAB_FILE,
    )
