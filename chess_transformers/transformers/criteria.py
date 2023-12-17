import torch
import torch.nn.functional as F

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


class LabelSmoothedCE(torch.nn.Module):
    """
    Cross Entropy loss with label-smoothing as a form of regularization.

    See "Rethinking the Inception Architecture for Computer Vision",
    https://arxiv.org/abs/1512.00567
    """

    def __init__(self, eps, n_predictions):
        """
        Init.

        Args:

            eps (float): Smoothing co-efficient. 

            n_predictions (int): Number of predictions expected per
            datapoint, or length of the predicted sequence.
        """
        super(LabelSmoothedCE, self).__init__()
        self.eps = eps
        self.indices = torch.arange(n_predictions).unsqueeze(0).to(DEVICE)  # (1, n_predictions)
        self.indices.requires_grad = False

    def forward(self, predicted, targets, lengths):
        """
        Forward prop.

        Args:

            predicted (torch.FloatTensor): The predicted probabilities,
            of size (N, n_predictions, vocab_size).

            targets (torch.LongTensor): The actual targets, of size (N,
            n_predictions).

            lengths (torch.LongTensor): The true lengths of the
            prediction sequences, not including special tokens, of size
            (N, 1).

        Returns:

            torch.Tensor: The mean label-smoothed cross-entropy loss, a
            scalar.
        """
        # Remove pad-positions and flatten
        predicted = predicted[
            self.indices < lengths
        ]  # (sum(lengths), vocab_size)
        targets = targets[self.indices < lengths]  # (sum(lengths))

        # "Smoothed" one-hot vectors for the gold sequences
        target_vector = (
            torch.zeros_like(predicted)
            .scatter(dim=1, index=targets.unsqueeze(1), value=1.0)
            .to(DEVICE)
        )  # (sum(lengths), vocab_size), one-hot
        target_vector = target_vector * (
            1.0 - self.eps
        ) + self.eps / target_vector.size(
            1
        )  # (sum(lengths), vocab_size), "smoothed" one-hot

        # Compute smoothed cross-entropy loss
        loss = (-1 * target_vector * F.log_softmax(predicted, dim=1)).sum(
            dim=1
        )  # (sum(lengths))

        # Compute mean loss
        loss = torch.mean(loss)

        return loss
