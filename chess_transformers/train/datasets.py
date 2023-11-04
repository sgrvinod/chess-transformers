import os
import json
import torch
import argparse
import tables as tb
from importlib import import_module
from torch.utils.data import Dataset


class ChessDataset(Dataset):
    def __init__(self, data_folder, h5_file, splits_file, split, n_moves=None):
        """
        Init.

        Args:

            data_folder (str): The folder containing the H5 and splits
            files.

            h5_file (str): The H5 file.

            splits_file (str): The splits file. Defaults to None, which
            means that all datapoints will be included.

            split (str): The data split. One of "train", "val", None.
            Defaults to None, which means that all datapoints will be
            included.

            n_moves (int, optional): Number of moves into the future to
            return. Defaults to None, which means that all moves in the
            H5 data column will be returned.
        """
        if n_moves is not None:
            assert n_moves > 0

        # Open table in H5 file
        self.h5_file = tb.open_file(os.path.join(data_folder, h5_file), mode="r")
        self.encoded_table = self.h5_file.root.encoded_data

        # Load splits
        if splits_file is not None:
            self.splits = json.load(open(os.path.join(data_folder, splits_file), "r"))

        # Create indices
        if split == "train":
            self.indices = list(range(0, self.splits["val_split_index"]))
        elif split == "val":
            self.indices = list(
                range(self.splits["val_split_index"], self.encoded_table.nrows)
            )
        elif split is None:
            self.indices = list(range(0, self.encoded_table.nrows))
        else:
            raise NotImplementedError

        # How many moves should be returned per row? Remember, there's a
        # "<move>" token prepended to each row's move sequence in the H5
        # data -- subtract 1 to get MAX_MOVE_SEQUENCE_LENGTH
        if n_moves is not None:
            # This is the same as min(MAX_MOVE_SEQUENCE_LENGTH, n_moves)
            self.n_moves = min(
                len(self.encoded_table[self.indices[0]]["move_sequence"]) - 1, n_moves
            )
        else:
            self.n_moves = len(self.encoded_table[self.indices[0]]["move_sequence"]) - 1

    def __getitem__(self, i):
        turns = torch.IntTensor([self.encoded_table[self.indices[i]]["turn"]])
        white_kingside_castling_rights = torch.IntTensor(
            [self.encoded_table[self.indices[i]]["white_kingside_castling_rights"]]
        )  # (1)
        white_queenside_castling_rights = torch.IntTensor(
            [self.encoded_table[self.indices[i]]["white_queenside_castling_rights"]]
        )  # (1)
        black_kingside_castling_rights = torch.IntTensor(
            [self.encoded_table[self.indices[i]]["black_kingside_castling_rights"]]
        )  # (1)
        black_queenside_castling_rights = torch.IntTensor(
            [self.encoded_table[self.indices[i]]["black_queenside_castling_rights"]]
        )  # (1)
        board_positions = torch.IntTensor(
            self.encoded_table[self.indices[i]]["board_position"]
        )  # (64)
        moves = torch.LongTensor(
            self.encoded_table[self.indices[i]]["move_sequence"][: self.n_moves + 1]
        )  # (n_moves + 1)
        lengths = torch.LongTensor(
            [self.encoded_table[self.indices[i]]["move_sequence_length"]]
        ).clamp(
            max=self.n_moves
        )  # (1), value <= n_moves

        return {
            "turns": turns,
            "white_kingside_castling_rights": white_kingside_castling_rights,
            "white_queenside_castling_rights": white_queenside_castling_rights,
            "black_kingside_castling_rights": black_kingside_castling_rights,
            "black_queenside_castling_rights": black_queenside_castling_rights,
            "board_positions": board_positions,
            "moves": moves,
            "lengths": lengths,
        }

    def __len__(self):
        return len(self.indices)


if __name__ == "__main__":
    # Get configuration
    parser = argparse.ArgumentParser()
    parser.add_argument("config_name", type=str, help="Name of configuration file.")
    args = parser.parse_args()
    CONFIG = import_module(
        "chess_transformers.configs.models.{}".format(args.config_name)
    )

    # Dataset
    dataset = ChessDataset(
        data_folder=CONFIG.DATA_FOLDER,
        h5_file=CONFIG.H5_FILE,
        splits_file=CONFIG.SPLITS_FILE,
        split="train",
        n_moves=5,
    )
    print(len(dataset))
    print(dataset[17])
