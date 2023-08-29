import os
import json
import argparse
import tables as tb
from tqdm import tqdm
from importlib import import_module


def split_data(
    data_folder, h5_file, splits_file, val_split_fraction, test_split_fraction
):
    """
    Find indices to split the H5 data at, so that it is split into the
    training, validation, and test data. The games in each split should
    be mutually exclusive.

    The training data will begin at index 0 (or a fraction of 0.0).

    Args:

        data_folder (str): The folder containing the H5 file.

        h5_file (str): The H5 file.

        splits_file (str): The splits file to be saved.

        val_split_fraction (float): The fraction, approximately, at
        which the validation data should begin.

        test_split_fraction (float): The fraction, approximately, at
        which the test data should begin.
    """

    assert val_split_fraction < test_split_fraction < 1.0
    print(
        "\nIt is desired that the validation set will start %2.6f%% into the data, and the test set at %2.6f%%.\n"
        % (val_split_fraction * 100.0, test_split_fraction * 100.0)
    )

    # Open table in H5 file
    h5_file = tb.open_file(os.path.join(data_folder, h5_file), mode="r")
    encoded_table = h5_file.root.encoded_data

    # Find indices where the we know for sure new games begin and
    # closely match the desired split fractions A new game begins for
    # sure when the winner (represented by "turn") changes
    found_val_split_index = False
    found_test_split_index = False
    winner = ""
    for i in tqdm(range(encoded_table.nrows), desc="Finding desired splits"):
        if encoded_table[i]["turn"] != winner:
            if not found_val_split_index:
                if i / encoded_table.nrows >= val_split_fraction:
                    val_split_index = i
                    found_val_split_index = True
            elif not found_test_split_index:
                if i / encoded_table.nrows >= test_split_fraction:
                    test_split_index = i
                    found_test_split_index = True
            else:
                break
            winner = encoded_table[i]["turn"]
    assert (
        encoded_table[val_split_index]["turn"]
        != encoded_table[val_split_index - 1]["turn"]
        and encoded_table[test_split_index]["turn"]
        != encoded_table[test_split_index - 1]["turn"]
    )
    print(
        "\nThe training set will start at index 0, the validation set at index %d (%2.6f%%), and the test set at index %d (%2.6f%%)."
        % (
            val_split_index,
            100.0 * val_split_index / encoded_table.nrows,
            test_split_index,
            100.0 * test_split_index / encoded_table.nrows,
        )
    )

    # Write to file
    with open(os.path.join(data_folder, splits_file), "w") as j:
        json.dump(
            {"val_split_index": val_split_index, "test_split_index": test_split_index},
            j,
            indent=4,
        )
    print("\nSplit indices saved to file.\n")

    h5_file.close()


if __name__ == "__main__":
    # Get configuration
    parser = argparse.ArgumentParser()
    parser.add_argument("config_name", type=str, help="Name of configuration file.")
    args = parser.parse_args()
    CONFIG = import_module("configs.{}".format(args.config_name))

    # Split data
    split_data(
        data_folder=CONFIG.DATA_FOLDER,
        h5_file=CONFIG.H5_FILE,
        splits_file=CONFIG.SPLITS_FILE,
        val_split_fraction=CONFIG.VAL_SPLIT_FRACTION,
        test_split_fraction=CONFIG.TEST_SPLIT_FRACTION,
    )
