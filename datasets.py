import os
import json
import time
import torch
import random
import tables as tb
from tqdm import tqdm
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
        )
        white_queenside_castling_rights = torch.IntTensor(
            [self.encoded_table[self.indices[i]]["white_queenside_castling_rights"]]
        )
        black_kingside_castling_rights = torch.IntTensor(
            [self.encoded_table[self.indices[i]]["black_kingside_castling_rights"]]
        )
        black_queenside_castling_rights = torch.IntTensor(
            [self.encoded_table[self.indices[i]]["black_queenside_castling_rights"]]
        )
        can_claim_draw = torch.IntTensor(
            [self.encoded_table[self.indices[i]]["can_claim_draw"]]
        )
        board_position = torch.IntTensor(
            self.encoded_table[self.indices[i]]["board_position"]
        )
        output_sequence = torch.IntTensor(
            self.encoded_table[self.indices[i]]["output_sequence"]
        )
        output_sequence_length = torch.IntTensor(
            [self.encoded_table[self.indices[i]]["output_sequence_length"]]
        )

        return (
            turn,
            white_kingside_castling_rights,
            white_queenside_castling_rights,
            black_kingside_castling_rights,
            black_queenside_castling_rights,
            can_claim_draw,
            board_position,
            output_sequence,
            output_sequence_length,
        )

    def __len__(self):
        return len(self.indices)


if __name__ == "__main__":
    dataset = ChessDataset(
        data_folder="/media/sgr/SSD/lichess data (copy)/",
        h5_file="data.h5",
        splits_file="splits.json",
        split="train",
    )
    print(len(dataset))
    print(dataset[17])
    dataloader = DataLoader(dataset, batch_size=5)
    for d in dataloader:
        print(d)
        break
