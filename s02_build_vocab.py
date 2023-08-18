import os
import json
import tables as tb
from config import *
from tqdm import tqdm
from collections import Counter


def build_vocabulary():
    # Open table in H5 file
    h5_file = tb.open_file(os.path.join(DATA_FOLDER, H5_FILE), mode="r")
    table = h5_file.root.data

    # Create move, board position vocabularies (with indices in order of most to least common moves)
    # For moves, an empty string ("") indicates a "no move" after a draw or loss is declared
    move_count = Counter()
    board_position_count = Counter()
    for i in tqdm(range(table.nrows), "Creating vocabulary"):
        move_count.update(list(table[i]["output_sequence"]))
        board_position_count.update(
            list(table[i]["board_position"].decode())
        )  # strings are bytestrings in H5 files, not unicode

    vocabulary = {
        "output_sequence": {},
        "board_position": {},
        "turn": {"b": 0, "w": 1},
        "white_kingside_castling_rights": {False: 0, True: 1},
        "white_queenside_castling_rights": {False: 0, True: 1},
        "black_kingside_castling_rights": {False: 0, True: 1},
        "black_queenside_castling_rights": {False: 0, True: 1},
        "can_claim_draw": {False: 0, True: 1},
    }
    for i, move in enumerate(dict(move_count.most_common()).keys()):
        vocabulary["output_sequence"][move.decode()] = i
    vocabulary["output_sequence"]["<move>"] = i + 1
    for i, board_position in enumerate(dict(board_position_count.most_common()).keys()):
        vocabulary["board_position"][board_position] = i

    print(
        "There are %d moves in the vocabulary, not including loss or draw declarations. Mathematically, there are only 1968 possible UCI moves."
        % (len(vocabulary["output_sequence"]) - 4)
    )

    # Save vocabulary to file
    with open(os.path.join(DATA_FOLDER, VOCAB_FILE), "w") as j:
        json.dump(vocabulary, j, indent=4)

    print("\nSaved to file.\n")

    # Close H5 file
    h5_file.close()


if __name__ == "__main__":
    build_vocabulary()
