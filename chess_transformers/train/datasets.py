import os
import json
import torch
import argparse
import tables as tb
from torch.utils.data import Dataset

from chess_transformers.configs import import_config


class ChessDataset(Dataset):
    def __init__(self, data_folder, h5_file, split, n_moves=None, **unused):
        """
        Init.

        Args:

            datasets (list): A list of tuples, each containing:

                (dict): A data configuration representing the
                dataset.

                (str): The data split. One of "train", "val", or
                None, which means that all datapoints will be
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
        self.split = split

        # Create indices
        if split == "train":
            self.first_index = 0
        elif split == "val":
            self.first_index = self.encoded_table.attrs.val_split_index
        elif split is None:
            self.first_index = 0
        else:
            raise NotImplementedError

        # How many moves should be returned per row? Remember, there's a
        # "<move>" token prepended to each row's move sequence in the H5
        # data -- subtract 1 to get MAX_MOVE_SEQUENCE_LENGTH
        if n_moves is not None:
            # This is the same as min(MAX_MOVE_SEQUENCE_LENGTH, n_moves)
            self.n_moves = min(
                len(self.encoded_table[self.first_index]["moves"]) - 1, n_moves
            )
        else:
            self.n_moves = len(self.encoded_table[self.first_index]["moves"]) - 1

    def __getitem__(self, i):
        turns = torch.IntTensor([self.encoded_table[self.first_index + i]["turn"]])
        white_kingside_castling_rights = torch.IntTensor(
            [self.encoded_table[self.first_index + i]["white_kingside_castling_rights"]]
        )  # (1)
        white_queenside_castling_rights = torch.IntTensor(
            [
                self.encoded_table[self.first_index + i][
                    "white_queenside_castling_rights"
                ]
            ]
        )  # (1)
        black_kingside_castling_rights = torch.IntTensor(
            [self.encoded_table[self.first_index + i]["black_kingside_castling_rights"]]
        )  # (1)
        black_queenside_castling_rights = torch.IntTensor(
            [
                self.encoded_table[self.first_index + i][
                    "black_queenside_castling_rights"
                ]
            ]
        )  # (1)
        board_position = torch.IntTensor(
            self.encoded_table[self.first_index + i]["board_position"]
        )  # (64)
        moves = torch.LongTensor(
            self.encoded_table[self.first_index + i]["moves"][: self.n_moves + 1]
        )  # (n_moves + 1)
        length = torch.LongTensor(
            [self.encoded_table[self.first_index + i]["length"]]
        ).clamp(
            max=self.n_moves
        )  # (1), value <= n_moves

        return {
            "turns": turns,
            "white_kingside_castling_rights": white_kingside_castling_rights,
            "white_queenside_castling_rights": white_queenside_castling_rights,
            "black_kingside_castling_rights": black_kingside_castling_rights,
            "black_queenside_castling_rights": black_queenside_castling_rights,
            "board_positions": board_position,
            "moves": moves,
            "lengths": length,
        }

    def __len__(self):
        if self.split == "train":
            return self.encoded_table.attrs.val_split_index
        elif self.split == "val":
            return self.encoded_table.nrows - self.encoded_table.attrs.val_split_index
        elif self.split is None:
            return self.encoded_table.nrows
        else:
            raise NotImplementedError


class ChessDatasetFT(Dataset):
    def __init__(self, data_folder, h5_file, split, **unused):
        """
        Init.

        Args:

            data_folder (str): The folder containing the H5 and splits
            files.

            h5_file (str): The H5 file.

            split (str): The data split. One of "train", "val", None.
            Defaults to None, which means that all datapoints will be
            included.
        """
        # Open table in H5 file
        self.h5_file = tb.open_file(os.path.join(data_folder, h5_file), mode="r")
        self.encoded_table = self.h5_file.root.encoded_data
        self.split = split

        # Create indices
        if split == "train":
            self.first_index = 0
        elif split == "val":
            self.first_index = self.encoded_table.attrs.val_split_index
        elif split is None:
            self.first_index = 0
        else:
            raise NotImplementedError

    def __getitem__(self, i):
        turns = torch.IntTensor([self.encoded_table[self.first_index + i]["turn"]])
        white_kingside_castling_rights = torch.IntTensor(
            [self.encoded_table[self.first_index + i]["white_kingside_castling_rights"]]
        )  # (1)
        white_queenside_castling_rights = torch.IntTensor(
            [self.encoded_table[self.first_index + i]["white_queenside_castling_rights"]]
        )  # (1)
        black_kingside_castling_rights = torch.IntTensor(
            [self.encoded_table[self.first_index + i]["black_kingside_castling_rights"]]
        )  # (1)
        black_queenside_castling_rights = torch.IntTensor(
            [self.encoded_table[self.first_index + i]["black_queenside_castling_rights"]]
        )  # (1)
        board_position = torch.IntTensor(
            self.encoded_table[self.first_index + i]["board_position"]
        )  # (64)
        from_square = torch.LongTensor(
            [self.encoded_table[self.first_index + i]["from_square"]]
        )  # (1)
        to_square = torch.LongTensor(
            [self.encoded_table[self.first_index + i]["to_square"]]
        )  # (1)
        length = torch.LongTensor([1])
        return {
            "turns": turns,
            "white_kingside_castling_rights": white_kingside_castling_rights,
            "white_queenside_castling_rights": white_queenside_castling_rights,
            "black_kingside_castling_rights": black_kingside_castling_rights,
            "black_queenside_castling_rights": black_queenside_castling_rights,
            "board_positions": board_position,
            "from_squares": from_square,
            "to_squares": to_square,
            "lengths": length,
        }

    def __len__(self):
        if self.split == "train":
            return self.encoded_table.attrs.val_split_index
        elif self.split == "val":
            return self.encoded_table.nrows - self.encoded_table.attrs.val_split_index
        elif self.split is None:
            return self.encoded_table.nrows
        else:
            raise NotImplementedError


if __name__ == "__main__":
    # Get configuration
    parser = argparse.ArgumentParser()
    parser.add_argument("config_name", type=str, help="Name of configuration file.")
    args = parser.parse_args()
    CONFIG = import_config(args.config_name)

    # Dataset
    dataset = CONFIG.DATASET(
        data_folder=CONFIG.DATA_FOLDER,
        h5_file=CONFIG.H5_FILE,
        split="train",
        n_moves=5,
    )
    print(len(dataset))
    print(dataset[17])
    dataset.h5_file.close()
