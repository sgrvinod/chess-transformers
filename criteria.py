import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)  # CPU isn't really practical here


class LabelSmoothedCE(torch.nn.Module):
    """
    Cross Entropy loss with label-smoothing as a form of regularization.

    See "Rethinking the Inception Architecture for Computer Vision",
    https://arxiv.org/abs/1512.00567
    """

    def __init__(self, eps=0.1):
        """
        Init.

        Args:

            eps (float, optional): Smoothing co-efficient. Defaults to
            0.1.
        """
        super(LabelSmoothedCE, self).__init__()
        self.eps = eps

    def forward(self, moves, actual_moves, lengths):
        """
        Forward prop.

        Args:

            moves (torch.FloatTensor): predicted next-move
            probabilities, of size (N, max_move_sequence_length,
            move_vocab_size)

            actual_moves (torch.LongTensor): actual moves made by the
            winner of this game, of size (N, move_vocab_size)

            lengths (torch.LongTensor): true lengths of move sequences,
            not including <move> and <pad> tokens, of size (N, 1)

        Returns:

            torch.Tensor: mean label-smoothed cross-entropy loss, a
            scalar
        """
        # Remove pad-positions and flatten
        moves, _, _, _ = pack_padded_sequence(
            input=moves, lengths=lengths.cpu(), batch_first=True, enforce_sorted=False
        )  # (sum(lengths), vocab_size)
        actual_moves, _, _, _ = pack_padded_sequence(
            input=actual_moves,
            lengths=lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )  # (sum(lengths))

        # "Smoothed" one-hot vectors for the gold sequences
        target_vector = (
            torch.zeros_like(moves)
            .scatter(dim=1, index=actual_moves.unsqueeze(1), value=1.0)
            .to(DEVICE)
        )  # (sum(lengths), move_vocab_size), one-hot
        target_vector = target_vector * (
            1.0 - self.eps
        ) + self.eps / target_vector.size(
            1
        )  # (sum(lengths), move_vocab_size), "smoothed" one-hot

        # Compute smoothed cross-entropy loss
        loss = (-1 * target_vector * F.log_softmax(moves, dim=1)).sum(
            dim=1
        )  # (sum(lengths))

        # Compute mean loss
        loss = torch.mean(loss)

        return loss
