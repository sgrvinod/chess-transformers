import math
import torch


def view_game(game):
    board = game.board()
    for move_number, move in enumerate(game.mainline_moves()):
        print("\n")
        print("Move #%d" % move_number)
        print("LAN:", board.lan(move))
        print("SAN:", board.san(move))
        print("UCI:", board.uci(move))
        board.push(move)
        print(board)


def get_lr(step, d_model, warmup_steps):
    """
    The LR schedule. This version below is twice the definition in the paper, as used in the official T2T repository.

    Args:

        step (int): training step number

        d_model (int): size of vectors throughout the transformer model

        warmup_steps (int): number of warmup steps where learning rate is increased linearly; twice the value in the paper, as in the official T2T repo

    Returns:

        float: updated learning rate
    """
    lr = (
        2.0
        * math.pow(d_model, -0.5)
        * min(math.pow(step, -0.5), step * math.pow(warmup_steps, -1.5))
    )

    return lr


def save_checkpoint(epoch, model, optimizer, prefix=""):
    """
    Checkpoint saver. Each save overwrites previous save.

    Args:

        epoch (int): epoch number (0-indexed)

        model (torch.nn.Module): transformer model

        optimizer (torch.optim.adam.Adam): optimizer

        prefix (str, optional): checkpoint filename prefix. Defaults to "".
    """
    state = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    filename = prefix + "transformer_checkpoint.pth.tar"
    torch.save(state, filename)


def change_lr(optimizer, new_lr):
    """
    Change learning rate to a specified value.

    Args:

        optimizer (torch.optim.adam.Adam): optimizer whose learning rate must be changed

        new_lr (float): new learning rate
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
