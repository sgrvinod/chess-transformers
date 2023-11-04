import torch
import torch.nn.functional as F

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)  # CPU isn't really practical here


class LabelSmoothedCE(torch.nn.Module):
    """
    Cross Entropy loss with label-smoothing as a form of regularization.

    See "Rethinking the Inception Architecture for Computer Vision",
    https://arxiv.org/abs/1512.00567
    """

    def __init__(self, eps, n_moves):
        """
        Init.

        Args:

            eps (float, optional): Smoothing co-efficient. Defaults to
            0.1.
        """
        super(LabelSmoothedCE, self).__init__()
        self.eps = eps
        self.indices = torch.arange(n_moves).unsqueeze(0).to(DEVICE)  # (1, n_moves)
        self.indices.requires_grad = False

    def forward(self, predicted_moves, target_moves, lengths):
        """
        Forward prop.

        Args:

            predicted_moves (torch.FloatTensor): The predicted next-move
            probabilities, of size (N, n_moves, move_vocab_size).

            target_moves (torch.LongTensor): The actual next-moves made
            in this game, of size (N, n_moves).

            lengths (torch.LongTensor): The true lengths of the move
            sequences, not including <move> and <pad> tokens, of size
            (N, 1).

        Returns:

            torch.Tensor: The mean label-smoothed cross-entropy loss, a
            scalar.
        """
        # Remove pad-positions and flatten
        predicted_moves = predicted_moves[
            self.indices < lengths
        ]  # (sum(lengths), vocab_size)
        target_moves = target_moves[self.indices < lengths]  # (sum(lengths))

        # "Smoothed" one-hot vectors for the gold sequences
        target_vector = (
            torch.zeros_like(predicted_moves)
            .scatter(dim=1, index=target_moves.unsqueeze(1), value=1.0)
            .to(DEVICE)
        )  # (sum(lengths), move_vocab_size), one-hot
        target_vector = target_vector * (
            1.0 - self.eps
        ) + self.eps / target_vector.size(
            1
        )  # (sum(lengths), move_vocab_size), "smoothed" one-hot

        # Compute smoothed cross-entropy loss
        loss = (-1 * target_vector * F.log_softmax(predicted_moves, dim=1)).sum(
            dim=1
        )  # (sum(lengths))

        # Compute mean loss
        loss = torch.mean(loss)

        return loss
