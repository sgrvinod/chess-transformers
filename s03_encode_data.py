import os
import json
import numpy as np
import tables as tb
from tqdm import tqdm


def encode(item, vocabulary):
    if isinstance(item, list):  # output sequence
        return [vocabulary[it.decode()] for it in item]
    elif isinstance(item, np.bytes_):  # turn or board position
        item = item.decode()
        return (
            vocabulary[item] if item in vocabulary else [vocabulary[it] for it in item]
        )
    elif isinstance(item, np.bool_):  # castling rights and draw potential
        return vocabulary[str(item).lower()]
    else:
        raise NotImplementedError


def encode_data(data_folder, h5_file, vocabulary_file):
    # Load vocabularies
    vocabulary = json.load(open(os.path.join(data_folder, vocabulary_file), "r"))

    # Open table in H5 file
    h5_file = tb.open_file(os.path.join(data_folder, h5_file), mode="a")
    table = h5_file.root.data

    # Create table description for HDF5 file
    class EncodedChessTable(tb.IsDescription):
        board_position = tb.Int8Col(shape=(64))
        turn = tb.Int8Col()
        white_kingside_castling_rights = tb.Int8Col()
        white_queenside_castling_rights = tb.Int8Col()
        black_kingside_castling_rights = tb.Int8Col()
        black_queenside_castling_rights = tb.Int8Col()
        can_claim_draw = tb.Int8Col()
        output_sequence = tb.Int16Col(shape=(table[0]["output_sequence"].shape[0] + 1))
        output_sequence_length = tb.Int8Col()

    # Create table in encoded HDF5 file
    encoded_table = h5_file.create_table(
        "/", "encoded_data", EncodedChessTable, expectedrows=15000000
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
        row["output_sequence"] = encode(
            [b"<move>"] + table[i]["output_sequence"].tolist(),
            vocabulary=vocabulary["output_sequence"],
        )
        row["output_sequence_length"] = len(
            [o for o in table[i]["output_sequence"].tolist() if o != b"<pad>"]
        )
        row.append()
    encoded_table.flush()
    assert encoded_table.nrows == table.nrows

    print("A total of %d datapoints have been saved to disk." % encoded_table.nrows)

    h5_file.close()

    print("Done.")


if __name__ == "__main__":
    encode_data(
        data_folder="/media/sgr/SSD/lichess data (copy)/",
        h5_file="data.h5",
        vocabulary_file="vocabulary.json",
    )
