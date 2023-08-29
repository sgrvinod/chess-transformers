import os
import json
import torch
import argparse
import tables as tb
from importlib import import_module
from torch.utils.data import Dataset, DataLoader


class ChessDataset(Dataset):
    def __init__(self, data_folder, h5_file, splits_file, split):
        assert split.lower() in ["train", "val", "test"]

        # Open table in H5 file
        self.h5_file = tb.open_file(os.path.join(data_folder, h5_file), mode="r")
        self.encoded_table = self.h5_file.root.encoded_data

        # Load splits
        self.splits = json.load(open(os.path.join(data_folder, splits_file), "r"))

        # Create indices
        if split == "train":
            self.indices = list(range(0, self.splits["val_split_index"]))
        elif split == "val":
            self.indices = list(
                range(self.splits["val_split_index"], self.splits["test_split_index"])
            )
        else:
            self.indices = list(
                range(self.splits["test_split_index"], self.encoded_table.nrows)
            )

    def __getitem__(self, i):

        turn = torch.IntTensor([self.encoded_table[self.indices[i]]["turn"]])
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
        can_claim_draw = torch.IntTensor(
            [self.encoded_table[self.indices[i]]["can_claim_draw"]]
        )  # (1)
        board_position = torch.IntTensor(
            self.encoded_table[self.indices[i]]["board_position"]
        )  # (64)
        move_sequence = torch.LongTensor(
            self.encoded_table[self.indices[i]]["move_sequence"]
        )  # (m)
        move_sequence_length = torch.LongTensor(
            [self.encoded_table[self.indices[i]]["move_sequence_length"]]
        )  # (1), value <= m - 1

        return (
            turn,
            white_kingside_castling_rights,
            white_queenside_castling_rights,
            black_kingside_castling_rights,
            black_queenside_castling_rights,
            can_claim_draw,
            board_position,
            move_sequence,
            move_sequence_length,
        )

    def __len__(self):
        return len(self.indices)


if __name__ == "__main__":
    # Get configuration
    parser = argparse.ArgumentParser()
    parser.add_argument("config_name", type=str, help="Name of configuration file.")
    args = parser.parse_args()
    CONFIG = import_module("configs.{}".format(args.config_name))

    # Dataset
    dataset = ChessDataset(
        data_folder=CONFIG.DATA_FOLDER,
        h5_file=CONFIG.H5_FILE,
        splits_file=CONFIG.SPLITS_FILE,
        split="train",
    )
    print(len(dataset))
    print(dataset[17])