import os
import json
import argparse
import tables as tb
from tqdm import tqdm
from collections import Counter
from importlib import import_module


def build_vocabulary(data_folder, h5_file, vocab_file):
    """
    Build vocabularies (token-to-index mappings) for all tokens and
    token-types in the H5 data and save to file.

    Args:

        data_folder (str): The folder with the H5 file.

        h5_file (str): The H5 file.

        vocab_file (str): The vocabulary file to be saved.
    """
    # Open table in H5 file
    h5_file = tb.open_file(os.path.join(data_folder, h5_file), mode="r")
    table = h5_file.root.data
    print("\n")

    # Create move, board position vocabularies (with indices in order of
    # most to least common moves) For moves, an empty string ("")
    # indicates a "no move" after a draw or loss is declared
    move_count = Counter()
    board_position_count = Counter()
    for i in tqdm(range(table.nrows), "Creating vocabulary"):
        move_count.update(list(table[i]["move_sequence"]))
        board_position_count.update(
            list(table[i]["board_position"].decode())
        )  # strings are bytestrings in H5 files, not unicode

    vocabulary = {
        "move_sequence": {},
        "board_position": {},
        "turn": {"b": 0, "w": 1},
        "white_kingside_castling_rights": {False: 0, True: 1},
        "white_queenside_castling_rights": {False: 0, True: 1},
        "black_kingside_castling_rights": {False: 0, True: 1},
        "black_queenside_castling_rights": {False: 0, True: 1},
        "can_claim_draw": {False: 0, True: 1},
    }
    for i, move in enumerate(dict(move_count.most_common()).keys()):
        vocabulary["move_sequence"][move.decode()] = i
    vocabulary["move_sequence"]["<move>"] = i + 1
    for i, board_position in enumerate(dict(board_position_count.most_common()).keys()):
        vocabulary["board_position"][board_position] = i

    print(
        "\nThere are %d moves in the vocabulary, not including loss or draw declarations. Mathematically, there are only 1968 possible UCI moves."
        % (len(vocabulary["move_sequence"]) - 4)
    )

    # Save vocabulary to file
    with open(os.path.join(data_folder, vocab_file), "w") as j:
        json.dump(vocabulary, j, indent=4)

    print("\nSaved to file.\n")

    # Close H5 file
    h5_file.close()


if __name__ == "__main__":
    # Get configuration
    parser = argparse.ArgumentParser()
    parser.add_argument("config_name", type=str, help="Name of configuration file.")
    args = parser.parse_args()
    CONFIG = import_module("configs.{}".format(args.config_name))

    # Build vocabulary
    build_vocabulary(
        data_folder=CONFIG.DATA_FOLDER,
        h5_file=CONFIG.H5_FILE,
        vocab_file=CONFIG.VOCAB_FILE,
    )
