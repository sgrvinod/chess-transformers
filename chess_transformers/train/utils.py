import os
import math
import json
import torch


def view_game(game):
    """
    View the progression of a chess game.

    Args:

        game (chess.pgn.Game): The game to be viewed.
    """
    board = game.board()
    for move_number, move in enumerate(game.mainline_moves()):
        print("\n")
        print("Move #%d" % move_number)
        print("LAN:", board.lan(move))
        print("SAN:", board.san(move))
        print("UCI:", board.uci(move))
        board.push(move)
        print(board)


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

        checkpoint_folder (str): The folder where checkpoints must be saved.

        prefix (str, optional): The checkpoint filename prefix. Defaults to
        "".
    """
    state = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    if not os.path.exists(checkpoint_folder):
        os.makedirs(checkpoint_folder)
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


def topk_accuracy(logits, targets, k=[1, 3, 5]):
    """
    Compute "top-k" accuracies for multiple values of "k".

    Args:

        logits (torch.FloatTensor): Predicted next-move probabilities,
        of size (N, move_vocab_size).

        targets (torch.LongTensor): Actual moves made by the winner of
        this game, of size (N).

        k (list, optional): Values of "k". Defaults to [1, 3, 5].

    Returns:

        list: "Top-k" accuracies.
    """
    with torch.no_grad():
        batch_size = logits.shape[0]

        # Get move indices corresponding to top-max(k) scores
        _, indices = logits.topk(k=max(k), dim=1)  # (N, max(k))

        # Expand actual (target) moves to the same shape
        targets = targets.unsqueeze(1).expand_as(indices)  # (N, max(k))

        # Calculate top-k accuracies
        correct_predictions = indices == targets
        topk_accuracies = [
            correct_predictions[:, :k_value].sum().item() / batch_size for k_value in k
        ]

        return topk_accuracies

