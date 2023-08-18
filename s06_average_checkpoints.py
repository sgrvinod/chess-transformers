import os
import torch
from config import *
from collections import OrderedDict

def average_checkpoints():
    """
    Averages the states of multiple checkpoints into a single
    checkpoint.
    """

    # Get list of checkpoint names
    checkpoint_names = [
        f
        for f in os.listdir(CHECKPOINT_FOLDER)
        if f.startswith(CHECKPOINT_AVG_PREFIX) and f.endswith(CHECKPOINT_AVG_SUFFIX)
    ]
    assert len(checkpoint_names) > 0, "Did not find any checkpoints!"

    # Average parameters from checkpoints
    averaged_params = OrderedDict()
    for c in checkpoint_names:
        checkpoint = torch.load(os.path.join(CHECKPOINT_FOLDER, c))
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
    torch.save({"model_state_dict": averaged_params}, os.path.join(CHECKPOINT_FOLDER, FINAL_CHECKPOINT))
    print("\nCheckpoints averaged and saved to file.\n")


if __name__ == "__main__":
    average_checkpoints()
