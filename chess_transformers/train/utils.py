import os
import math
import json
import torch
import pathlib
import torch.nn.functional as F


def get_vocab_sizes(data_folder, vocab_file):
    """
    Get sizes of all vocabularies in the vocabulary file.

    Args:

        data_folder (str): The folder containing all data files.

        vocab_file (str): The vocabulary file.

    Returns:

        dict: A dictionary containing the sizes of each vocabulary in
        the vocabulary file.
    """
    vocabulary = json.load(open(os.path.join(data_folder, vocab_file), "r"))
    vocab_sizes = dict()
    for k in vocabulary:
        vocab_sizes[k] = len(vocabulary[k])

    return vocab_sizes


def get_lr(step, d_model, warmup_steps):
    """
    The LR schedule. This version below is twice the definition in the
    paper, as used in the official T2T repository.

    Args:

        step (int): Training step number.

        d_model (int): Size of vectors throughout the transformer model.

        warmup_steps (int): Number of warmup steps where learning rate
        is increased linearly; twice the value in the paper, as in the
        official T2T repo.

    Returns:

        float: Updated learning rate.
    """
    lr = (
        2.0
        * math.pow(d_model, -0.5)
        * min(math.pow(step, -0.5), step * math.pow(warmup_steps, -1.5))
    )

    return lr


def save_checkpoint(epoch, model, optimizer, config_name, checkpoint_folder, prefix=""):
    """
    Checkpoint saver. Each save overwrites any previous save.

    Args:

        epoch (int): The epoch number (0-indexed).

        model (torch.nn.Module): The transformer model.

        optimizer (torch.optim.adam.Adam): The optimizer.

        config_name (str): The configuration name.

        checkpoint_folder (str): The folder where checkpoints must be
        saved.

        prefix (str, optional): The checkpoint filename prefix. Defaults
        to "".
    """
    state = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    pathlib.Path(checkpoint_folder).mkdir(parents=True, exist_ok=True)
    filename = prefix + config_name + ".pt"
    torch.save(state, os.path.join(checkpoint_folder, filename))
    print("Checkpoint saved.\n")


def change_lr(optimizer, new_lr):
    """
    Change learning rate to a specified value.

    Args:

        optimizer (torch.optim.adam.Adam): Optimizer whose learning rate
        must be changed.

        new_lr (float): New learning rate.
    """
    for param_group in optimizer.param_groups:
        param_group["lr"] = new_lr


class AverageMeter(object):
    """
    Keeps track of most recent, average, sum, and count of a metric.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def topk_accuracy(logits, targets, other_logits=None, other_targets=None, k=[1, 3, 5]):
    """
    Compute "top-k" accuracies for multiple values of "k".

    Optionally, a second set of logits and targets, for a second
    predicted variable, can be provided. In this case, probabilities
    associated with both sets of logits are combined to arrive at the
    best combinations of both predicted variables. A correct prediction
    occurs when the combination of the targets is present in the top "k"
    predicted combinations.

    Args:

        logits (torch.FloatTensor): Predicted logits, of size (N,
        vocab_size).

        targets (torch.LongTensor): Actual targets, of size (N).

        other_logits (torch.FloatTensor, optional): Predicted logits for a
        second predicted variable, if any, of size (N, other_vocab_size).
        Defaults to None.

        other_targets (torch.LongTensor, optional): Actual targets for a second
        predicted variable, if any, of size (N). Defaults to None.

        k (list, optional): Values of "k". Defaults to [1, 3, 5].

    Returns:

        list: "Top-k" accuracies.
    """
    with torch.no_grad():
        batch_size = logits.shape[0]
        if other_logits is not None:
            # Get indices corresponding to top-max(k) scores
            probabilities = F.softmax(logits, dim=-1).unsqueeze(2)  # (N, vocab_size, 1)
            other_probabilities = F.softmax(other_logits, dim=-1).unsqueeze(
                1
            )  # (N, 1, other_vocab_size)
            combined_probabilities = torch.bmm(probabilities, other_probabilities).view(
                batch_size, -1
            )  # (N, vocab_size * other_vocab_size)
            _, flattened_indices = combined_probabilities.topk(
                k=max(k), dim=1
            )  # (N, max(k))
            indices = flattened_indices // other_logits.shape[-1]  # (N, max(k))
            other_indices = flattened_indices % other_logits.shape[-1]  # (N, max(k))

            # Expand targets to the same shape
            targets = targets.unsqueeze(1).expand_as(indices)  # (N, max(k))
            other_targets = other_targets.unsqueeze(1).expand_as(
                other_indices
            )  # (N, max(k))

            # Get correct predictions
            correct_predictions = (indices == targets) * (
                other_indices == other_targets
            )  # (N, max(k))

        else:
            # Get indices corresponding to top-max(k) scores
            _, indices = logits.topk(k=max(k), dim=1)  # (N, max(k))

            # Expand targets to the same shape
            targets = targets.unsqueeze(1).expand_as(indices)  # (N, max(k))

            # Get correct predictions
            correct_predictions = indices == targets  # (N, max(k))

        # Calculate top-k accuracies
        topk_accuracies = [
            correct_predictions[:, :k_value].sum().item() / batch_size for k_value in k
        ]

        return topk_accuracies
