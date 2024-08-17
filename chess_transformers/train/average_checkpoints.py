import os
import torch
import argparse
from collections import OrderedDict

from chess_transformers.configs import import_config


def average_checkpoints(
    checkpoint_folder, checkpoint_avg_prefix, checkpoint_avg_suffix, final_checkpoint
):
    """
    Averages the states of multiple checkpoints into a single
    checkpoint.

    Args:

        checkpoint_folder (str): The folder containing the checkpoints
        to be averaged.

        checkpoint_avg_prefix (str): A pattern for matching the
        beginnings of the names of these checkpoints.

        checkpoint_avg_suffix (str): A pattern for matching the ends of
        the names of these checkpoints.

        final_checkpoint (str): The final, averaged checkpoint to be
        saved.
    """

    # Get list of checkpoint names
    checkpoint_names = [
        f
        for f in os.listdir(checkpoint_folder)
        if f.startswith(checkpoint_avg_prefix) and f.endswith(checkpoint_avg_suffix)
    ]
    assert len(checkpoint_names) > 0, "Did not find any checkpoints!"

    # Average parameters from checkpoints
    averaged_params = OrderedDict()
    for c in checkpoint_names:
        checkpoint = torch.load(os.path.join(checkpoint_folder, c), weights_only=True)
        checkpoint_params = checkpoint["model_state_dict"]
        checkpoint_param_names = checkpoint_params.keys()
        for param_name in checkpoint_param_names:
            if param_name not in averaged_params:
                averaged_params[param_name] = (
                    checkpoint_params[param_name].clone() * 1 / len(checkpoint_names)
                )
            else:
                averaged_params[param_name] += (
                    checkpoint_params[param_name] * 1 / len(checkpoint_names)
                )
    # If a compiled model was saved, keys may be prepended with
    # "_orig_mod."
    for param_name in list(averaged_params.keys()):
        if param_name.startswith("_orig_mod."):
            averaged_params[param_name.replace("_orig_mod.", "")] = averaged_params[
                param_name
            ]
            del averaged_params[param_name]

    # Save averaged model
    torch.save(
        {"model_state_dict": averaged_params},
        os.path.join(checkpoint_folder, final_checkpoint),
    )
    print("\nCheckpoints averaged and saved to file.\n")


if __name__ == "__main__":
    # Get configuration
    parser = argparse.ArgumentParser()
    parser.add_argument("config_name", type=str, help="Name of configuration file.")
    args = parser.parse_args()
    CONFIG = import_config(args.config_name)

    # Average checkpoints
    average_checkpoints(
        checkpoint_folder=CONFIG.CHECKPOINT_FOLDER,
        checkpoint_avg_prefix=CONFIG.CHECKPOINT_AVG_PREFIX,
        checkpoint_avg_suffix=CONFIG.CHECKPOINT_AVG_SUFFIX,
        final_checkpoint=CONFIG.FINAL_CHECKPOINT,
    )
