import os
import torch
from collections import OrderedDict

# Model parameters
d_model = 512  # size of vectors throughout the transformer model
n_heads = 8  # number of heads in the multi-head attention
d_queries = 64  # size of query vectors (and also the size of the key vectors) in the multi-head attention
d_values = 64  # size of value vectors in the multi-head attention
d_inner = 2048  # an intermediate size in the position-wise FC
n_layers = 6  # number of layers in the Encoder and Decoder
dropout = 0.1  # dropout probability
max_move_sequence_length = 10  # expected maximum length of move sequences


def average_checkpoints(
    checkpoint_folder,
    starts_with,
    ends_with,
    averaged_checkpoint,
):
    """
    Averages the states of multiple checkpoints into a single
    checkpoint.

    Args:

        checkpoint_folder (str): folder containing checkpoints that need
        to be averaged

        starts_with (str): checkpoints' names begin with this string

        ends_with (str): checkpoints' names end with this string

        averaged_checkpoint (str): name of averaged checkpoint file to
        be created
    """

    # Get list of checkpoint names
    checkpoint_names = [
        f
        for f in os.listdir(checkpoint_folder)
        if f.startswith(starts_with) and f.endswith(ends_with)
    ]
    assert len(checkpoint_names) > 0, "Did not find any checkpoints!"

    # Average parameters from checkpoints
    averaged_params = OrderedDict()
    for c in checkpoint_names:
        checkpoint = torch.load(os.path.join(checkpoint_folder, c))
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
    torch.save({"model_state_dict": averaged_params}, os.path.join(checkpoint_folder, averaged_checkpoint))
    print("\nCheckpoints averaged and saved to file.\n")


if __name__ == "__main__":
    average_checkpoints(
        checkpoint_folder="./",
        starts_with="step",
        ends_with=".pt",
        averaged_checkpoint="averaged_transformer_checkpoint.pt",
    )
