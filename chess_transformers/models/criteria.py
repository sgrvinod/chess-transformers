"""Loss functions for training the chess transformer model.

This module provides custom loss functions designed for the twin-head
chess transformer architecture, which predicts both a 'From' square and
a 'To' square for each move. The loss functions handle both prediction
heads and return a combined scalar loss value.

Key Features:
    - DualLabelSmoothedCE: Cross entropy with label smoothing for both heads,
      which helps prevent overconfident predictions and improves generalization.
    - DualFocalLoss: Focal loss for both heads, which addresses class imbalance
      by down-weighting well-classified examples and focusing on hard cases.
    - LegalMoveSmoothing: Label smoothing that distributes probability mass
      only among legal moves, using bitmasks from the data pipeline.

Notes:
    All loss functions expect logits with shape (batch, 1, 64) for each head,
    where 64 represents the number of squares on a chess board. The singleton
    dimension is squeezed internally before computing the loss.
"""

import torch
import torch.nn as nn
import torch.nn.functional as f


class DualLabelSmoothedCE(nn.Module):
    """
    Cross Entropy loss with label-smoothing for twin-head (From/To) prediction.

    Sums the label-smoothed cross entropy loss for both the 'From' square
    and 'To' square predictions independently.

    See "Rethinking the Inception Architecture for Computer Vision",
    https://arxiv.org/abs/1512.00567
    """

    def __init__(
        self,
        eps: float = 0.1,
        from_weight: float = 1.0,
        to_weight: float = 1.0,
    ) -> None:
        """
        Initialize the loss function.

        Args:
            eps (float): Smoothing coefficient. A value of 0.1 means
                the correct class gets probability 0.9, and the remaining 0.1
                is distributed uniformly among all other classes.
            from_weight (float): Weight for the 'From' square loss.
            to_weight (float): Weight for the 'To' square loss.
        """
        super().__init__()
        self.eps = eps
        self.from_weight = from_weight
        self.to_weight = to_weight

    def _smoothed_loss(
        self, logits: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute label-smoothed cross entropy for a single head.

        Args:
            logits: Predicted logits of size (batch, num_classes).
            target: Target class indices of size (batch).

        Returns:
            Scalar loss value.

        Note:
            This uses the standard label smoothing formulation where the
            smoothing mass (eps / K) is distributed to ALL classes, including
            the true class. The true class receives (1 - eps) + (eps / K),
            while all other classes receive (eps / K).
        """
        # Number of classes (64 squares)
        n_classes = logits.size(-1)

        # Create smoothed targets
        # One-hot encode the target: (batch, n_classes)
        with torch.no_grad():
            target_one_hot = torch.zeros_like(logits).scatter(
                dim=1, index=target.unsqueeze(1), value=1.0
            )

            # Apply label smoothing via linear interpolation
            smoothed_target = target_one_hot * (1.0 - self.eps) + self.eps / n_classes

        # Compute log probabilities
        log_probs = f.log_softmax(logits, dim=-1)

        # Compute cross entropy: - sum(target * log_prob)
        loss = (-smoothed_target * log_probs).sum(dim=-1)

        return loss.mean()

    def forward(
        self,
        from_logits: torch.Tensor,
        to_logits: torch.Tensor,
        from_targets: torch.Tensor,
        to_targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Calculate total loss.

        Args:
            from_logits (torch.Tensor): Logits for 'From' squares (batch, 1, 64).
            to_logits (torch.Tensor): Logits for 'To' squares (batch, 1, 64).
            from_targets (torch.Tensor): Target 'From' indices (batch,).
            to_targets (torch.Tensor): Target 'To' indices (batch,).

        Returns:
            torch.Tensor: Weighted sum of losses for From and To heads.
        """
        # Remove the singleton dimension (batch, 1, 64) -> (batch, 64)
        from_logits = from_logits.squeeze(1)
        to_logits = to_logits.squeeze(1)

        loss_from = self._smoothed_loss(from_logits, from_targets)
        loss_to = self._smoothed_loss(to_logits, to_targets)

        return self.from_weight * loss_from + self.to_weight * loss_to


class DualFocalLoss(nn.Module):
    """
    Focal Loss for twin-head (From/To) prediction.

    Addresses class imbalance by down-weighting well-classified examples.
    Formula: FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)

    See "Focal Loss for Dense Object Detection",
    https://arxiv.org/abs/1708.02002
    """

    def __init__(self, alpha: float = 1.0, gamma: float = 2.0) -> None:
        """
        Initialize Focal Loss.

        Args:
            alpha (float): Balancing factor.
            gamma (float): Focusing parameter. Higher values strictly focus
                training on hard, misclassified examples.
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def _focal_loss(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute focal loss for a single head.

        Args:
            logits: Predicted logits of size (batch, num_classes).
            target: Target class indices of size (batch).

        Returns:
            Scalar loss value.
        """
        # Compute cross entropy (without reduction) to get -log(p_t)
        ce_loss = f.cross_entropy(logits, target, reduction="none")

        # Compute p_t from cross entropy: p_t = exp(-ce_loss)
        p_t = torch.exp(-ce_loss)

        # Apply focal weighting: (1 - p_t)^gamma
        focal_term = (1.0 - p_t).pow(self.gamma)

        # Combine with alpha scaling
        loss = self.alpha * focal_term * ce_loss

        return loss.mean()

    def forward(
        self,
        from_logits: torch.Tensor,
        to_logits: torch.Tensor,
        from_targets: torch.Tensor,
        to_targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Calculate total focal loss.

        Args:
            from_logits (torch.Tensor): Logits for 'From' squares (batch, 1, 64).
            to_logits (torch.Tensor): Logits for 'To' squares (batch, 1, 64).
            from_targets (torch.Tensor): Target 'From' indices (batch,).
            to_targets (torch.Tensor): Target 'To' indices (batch,).

        Returns:
            torch.Tensor: Sum of losses for From and To heads.
        """
        # Remove the singleton dimension
        from_logits = from_logits.squeeze(1)
        to_logits = to_logits.squeeze(1)

        loss_from = self._focal_loss(from_logits, from_targets)
        loss_to = self._focal_loss(to_logits, to_targets)

        return loss_from + loss_to


class LegalMoveSmoothing(nn.Module):
    """
    Cross Entropy loss with label smoothing distributed among legal moves only.

    Unlike standard label smoothing which distributes the smoothing mass
    uniformly across all 64 squares, this loss function distributes it
    only among squares that correspond to legal moves. This provides a
    more principled regularization signal by only considering valid
    chess moves.

    The target (ground truth) square receives (1 - eps) probability, while
    the remaining eps is distributed uniformly among the other legal
    squares. If a mask contains only the target square, no smoothing is
    applied (equivalent to standard cross-entropy).

    For the "from" head, legal squares are those from which any legal move
    originates. For the "to" head, legal squares depend on the mask mode
    used during data processing:
    - "disjoint": All squares that are legal destinations from any piece.
    - "conditional": Only squares that are legal destinations from the
      actual from-square of the target move.
    """

    def __init__(
        self,
        eps: float = 0.1,
        from_weight: float = 1.0,
        to_weight: float = 1.0,
    ) -> None:
        """
        Initialize the loss function.

        Args:
            eps: Smoothing coefficient. A value of 0.1 means the correct
                class gets probability 0.9, and the remaining 0.1 is
                distributed uniformly among other legal squares.
            from_weight: Weight for the 'From' square loss.
            to_weight: Weight for the 'To' square loss.
        """
        super().__init__()
        self.eps = eps
        self.from_weight = from_weight
        self.to_weight = to_weight

    def _smoothed_loss(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        legal_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute label-smoothed cross entropy with legal move constraints.

        Args:
            logits: Predicted logits of size (batch, 64).
            target: Target class indices of size (batch,).
            legal_mask: 64-bit integer masks of size (batch,), where bit i
                is set if square i is a legal option.

        Returns:
            Scalar loss value averaged over the batch.

        Note:
            For each sample, the smoothing mass (eps / n_legal) is distributed
            to ALL legal squares, including the target. The target square gets
            (1 - eps) + (eps / n_legal), while other legal squares each get
            (eps / n_legal). This matches the formulation in DualLabelSmoothedCE.
        """
        batch_size = logits.size(0)
        n_classes = logits.size(-1)  # 64
        device = logits.device
        dtype = logits.dtype

        # Expand legal_mask bits into a float tensor of shape (batch, 64)
        # legal_mask is (batch,) with int64, each bit represents a square
        bit_positions = torch.arange(n_classes, device=device, dtype=torch.int64)
        # Shape: (1, 64) for broadcasting
        bit_positions = bit_positions.unsqueeze(0)
        # Shape: (batch, 1) for broadcasting
        legal_mask = legal_mask.unsqueeze(1)
        # Boolean mask: (batch, 64)
        legal_bool = ((legal_mask >> bit_positions) & 1).to(dtype)

        # Count how many legal squares per sample (batch,)
        n_legal = legal_bool.sum(dim=1).clamp(min=1)  # Avoid division by zero

        # Create one-hot target: (batch, 64)
        target_one_hot = torch.zeros(batch_size, n_classes, device=device, dtype=dtype)
        target_one_hot.scatter_(1, target.unsqueeze(1), 1.0)

        # Build smoothed targets using linear interpolation (same as DualLabelSmoothedCE):
        # smoothed = one_hot * (1 - eps) + (eps / n_legal) * legal_mask  # noqa: E800
        #
        # This gives:
        # - Target square: (1 - eps) + eps/n_legal
        # - Other legal squares: eps/n_legal
        # - Illegal squares: 0
        eps_per_legal = (self.eps / n_legal).unsqueeze(1)  # (batch, 1)
        smoothed_target = target_one_hot * (1.0 - self.eps) + eps_per_legal * legal_bool

        # Compute log probabilities
        log_probs = f.log_softmax(logits, dim=-1)

        # Compute cross entropy: - sum(smoothed_target * log_prob)
        loss = (-smoothed_target * log_probs).sum(dim=-1)

        return loss.mean()

    def forward(
        self,
        from_logits: torch.Tensor,
        to_logits: torch.Tensor,
        from_targets: torch.Tensor,
        to_targets: torch.Tensor,
        legal_from_mask: torch.Tensor,
        legal_to_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Calculate total loss with legal move smoothing.

        Args:
            from_logits: Logits for 'From' squares (batch, 1, 64).
            to_logits: Logits for 'To' squares (batch, 1, 64).
            from_targets: Target 'From' indices (batch,).
            to_targets: Target 'To' indices (batch,).
            legal_from_mask: 64-bit masks indicating legal from-squares (batch,).
            legal_to_mask: 64-bit masks indicating legal to-squares (batch,).

        Returns:
            Weighted sum of losses for From and To heads.
        """
        # Remove the singleton dimension (batch, 1, 64) -> (batch, 64)
        from_logits = from_logits.squeeze(1)
        to_logits = to_logits.squeeze(1)

        loss_from = self._smoothed_loss(from_logits, from_targets, legal_from_mask)
        loss_to = self._smoothed_loss(to_logits, to_targets, legal_to_mask)

        return self.from_weight * loss_from + self.to_weight * loss_to
