"""Optimized transformer modules for chess move prediction.

This module provides building blocks for a transformer-based chess model with
architectural choices optimized for the chess domain:

- **RMSNorm**: Root Mean Square normalization, faster than LayerNorm as it skips
  mean centering. Used in modern architectures like LLaMA and Mistral.

- **SwiGLU FFN**: Feed-forward network with gated linear units and SiLU activation.
  More expressive than standard ReLU FFN due to multiplicative gating.

- **DisentangledAttention**: Multi-head attention with 2D relative position embeddings,
  inspired by DeBERTa. Decomposes attention into content-to-content, content-to-position,
  and position-to-content components.

- **TransformerEncoderLayer**: Pre-norm transformer layer combining attention and FFN.

- **BoardEncoder**: Full transformer encoder that processes chess board state into
  contextualized embeddings.

Example:
    >>> encoder = BoardEncoder(
    ...     vocab_sizes={"turn": 2, "white_kingside_castling_rights": 2, ...},
    ...     d_model=256,
    ...     n_heads=8,
    ...     d_ff=512,
    ...     n_layers=6,
    ... )
    >>> output = encoder(turns, wk, wq, bk, bq, board_positions)
    >>> output.shape
    torch.Size([batch, 69, 256])

Attributes:
    N_SQUARES (int): Number of squares on the board (64).
    N_METADATA (int): Number of metadata tokens (5: turn + 4 castling rights).
    N_TOKENS (int): Total sequence length (69 = 5 metadata + 64 squares).
    BOARD_SIZE (int): Board dimension (8x8).
    MAX_REL_DIST (int): Number of relative distance bins (15, for -7 to +7 offset by 7).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812

# Board constants
N_SQUARES = 64
N_METADATA = 5  # turn + 4 castling rights
N_TOKENS = N_METADATA + N_SQUARES  # 69 total
BOARD_SIZE = 8
MAX_REL_DIST = 15  # -7 to +7, offset by 7


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.

    Normalizes inputs by their root mean square, without mean centering.
    This is faster than LayerNorm while achieving similar performance.
    Used in LLaMA, Mistral, and other modern transformer architectures.

    The normalization is computed as:
        output = x / RMS(x) * weight
        where RMS(x) = sqrt(mean(x^2) + eps)

    Attributes:
        eps (float): Small constant for numerical stability.
        weight (nn.Parameter): Learnable scale parameter of shape (d_model,).
    """

    def __init__(self, d_model: int, eps: float = 1e-6) -> None:
        """Initialize RMSNorm.

        Args:
            d_model: The dimension of the input features (last dimension).
            eps: Small constant added to denominator for numerical stability.
                Defaults to 1e-6.
        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply RMS normalization.

        Args:
            x: Input tensor of shape (batch, seq, d_model) or any shape
                where the last dimension is d_model.

        Returns:
            Normalized tensor of the same shape and dtype as input.
        """
        dtype = x.dtype
        x = x.float()  # upcast to float32 for numerical stability
        rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (x * rms).to(dtype) * self.weight


class SwiGLUFFN(nn.Module):
    """Feed-forward network with SwiGLU activation.

    Implements the gated linear unit variant used in modern transformers:
        SwiGLU(x) = (SiLU(x @ W_gate) * (x @ W_up)) @ W_down

    The multiplicative gating mechanism allows the network to learn which
    features to pass through, making it more expressive than standard ReLU FFN.
    Used in LLaMA, PaLM, and other modern architectures.

    Note:
        The effective parameter count is higher than a standard 2-layer FFN
        due to the additional gate projection. For equivalent parameter count,
        use d_ff = (2/3) * standard_d_ff.

    Attributes:
        w_gate (nn.Linear): Gate projection (d_model -> d_ff).
        w_up (nn.Linear): Up projection (d_model -> d_ff).
        w_down (nn.Linear): Down projection (d_ff -> d_model).
        dropout (nn.Dropout): Dropout layer applied to output.
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0) -> None:
        """Initialize SwiGLU feed-forward network.

        Args:
            d_model: Input and output dimension.
            d_ff: Hidden dimension (intermediate size).
            dropout: Dropout probability applied to output. Defaults to 0.0.
        """
        super().__init__()
        self.w_gate = nn.Linear(d_model, d_ff, bias=False)
        self.w_up = nn.Linear(d_model, d_ff, bias=False)
        self.w_down = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply SwiGLU feed-forward transformation.

        Args:
            x: Input tensor of shape (batch, seq, d_model).

        Returns:
            Output tensor of shape (batch, seq, d_model).
        """
        gate = F.silu(self.w_gate(x))  # (batch, seq, d_ff)
        up = self.w_up(x)  # (batch, seq, d_ff)
        return self.dropout(self.w_down(gate * up))  # (batch, seq, d_model)


class DisentangledAttention(nn.Module):
    """Multi-head attention with 2D disentangled relative position embeddings.

    Implements a variant of DeBERTa-style disentangled attention adapted for
    chess boards. The attention score between positions i and j is decomposed
    into three components:

    - **C2C (content-to-content)**: Standard Q·K attention based on token content.
    - **C2P (content-to-position)**: Q_c · K_r where K_r encodes the relative
      position of the query with respect to the key.
    - **P2C (position-to-content)**: Q_r · K_c where Q_r encodes the relative
      position of the key with respect to the query.

    For board squares (token positions 5-68), all three components are used.
    For metadata tokens (positions 0-4: turn and castling rights), only C2C
    is applied since relative position is not meaningful for these tokens.

    The relative positions are decomposed into row and column components,
    providing a 2D inductive bias appropriate for the chess board structure.

    Attributes:
        d_model (int): Model dimension.
        n_heads (int): Number of attention heads.
        d_head (int): Dimension per head (d_model // n_heads).
        scale (float): Attention scaling factor (1 / sqrt(d_head)).
        dropout_p (float): Dropout probability.
        w_q (nn.Linear): Query projection.
        w_k (nn.Linear): Key projection.
        w_v (nn.Linear): Value projection.
        w_out (nn.Linear): Output projection.
        w_kr (nn.Linear): Relative position key projection for C2P.
        w_qr (nn.Linear): Relative position query projection for P2C.
        E_rel_row (nn.Embedding): Row relative position embeddings.
        E_rel_col (nn.Embedding): Column relative position embeddings.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.0,
        rel_pos_embeddings: tuple = None,
    ) -> None:
        """Initialize disentangled attention layer.

        Args:
            d_model: Model dimension. Must be divisible by n_heads.
            n_heads: Number of attention heads.
            dropout: Dropout probability for attention weights. Defaults to 0.0.
            rel_pos_embeddings: Optional tuple of (E_rel_row, E_rel_col) embedding
                layers for sharing across multiple attention layers. If None,
                creates new per-layer embeddings.

        Raises:
            AssertionError: If d_model is not divisible by n_heads.
        """
        super().__init__()

        assert (  # noqa: S101
            d_model % n_heads == 0
        ), "d_model must be divisible by n_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.scale = 1.0 / math.sqrt(self.d_head)
        self.dropout_p = dropout

        # Content projections (Q, K, V for standard attention)
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_out = nn.Linear(d_model, d_model, bias=False)

        # Relative position projections
        self.w_kr = nn.Linear(d_model, d_model, bias=False)  # For C2P: projects P(i→j)
        self.w_qr = nn.Linear(d_model, d_model, bias=False)  # For P2C: projects P(j→i)

        # Relative position embeddings (per-layer or shared)
        if rel_pos_embeddings is not None:
            # Use shared embeddings (don't register as submodules to avoid double counting)
            self.E_rel_row = rel_pos_embeddings[0]
            self.E_rel_col = rel_pos_embeddings[1]
        else:
            # Create per-layer embeddings
            self.E_rel_row = nn.Embedding(MAX_REL_DIST, d_model)
            self.E_rel_col = nn.Embedding(MAX_REL_DIST, d_model)

        # Pre-compute relative position indices (static, 64x64)
        self._register_rel_pos_indices()

    def _register_rel_pos_indices(self) -> None:
        """Pre-compute and register relative position index matrices.

        Creates four index tensors for looking up relative position embeddings:
        - rel_row_idx_ij: Row offset of square i relative to square j
        - rel_col_idx_ij: Column offset of square i relative to square j
        - rel_row_idx_ji: Row offset of square j relative to square i
        - rel_col_idx_ji: Column offset of square j relative to square i

        All indices are offset by 7 to convert the range [-7, 7] to [0, 14].
        Tensors are registered as non-persistent buffers (not saved in state_dict).
        """
        # Compute row and column for each of the 64 squares
        # Square index i corresponds to row i//8, column i%8
        rows = (
            torch.arange(BOARD_SIZE)
            .unsqueeze(1)
            .expand(BOARD_SIZE, BOARD_SIZE)
            .reshape(N_SQUARES)
        )
        cols = (
            torch.arange(BOARD_SIZE)
            .unsqueeze(0)
            .expand(BOARD_SIZE, BOARD_SIZE)
            .reshape(N_SQUARES)
        )

        # Relative position of query i w.r.t. key j: Δr = row_i - row_j, Δc = col_i - col_j
        # Offset by 7 to get indices in [0, 14]
        rel_row_ij = rows.unsqueeze(1) - rows.unsqueeze(0) + 7  # (64, 64)
        rel_col_ij = cols.unsqueeze(1) - cols.unsqueeze(0) + 7  # (64, 64)

        # Relative position of key j w.r.t. query i (for P2C): negation gives 14 - idx
        rel_row_ji = 14 - rel_row_ij
        rel_col_ji = 14 - rel_col_ij

        # Register as non-persistent buffers (not needed in state_dict)
        self.register_buffer("rel_row_idx_ij", rel_row_ij, persistent=False)
        self.register_buffer("rel_col_idx_ij", rel_col_ij, persistent=False)
        self.register_buffer("rel_row_idx_ji", rel_row_ji, persistent=False)
        self.register_buffer("rel_col_idx_ji", rel_col_ji, persistent=False)

    def _compute_rel_pos_bias(
        self, Q_board: torch.Tensor, K_board: torch.Tensor  # noqa: N803
    ) -> torch.Tensor:
        """
        Compute C2P + P2C relative position bias for board squares.

        Args:
            Q_board: Query vectors for board squares, (batch, n_heads, 64, d_head)
            K_board: Key vectors for board squares, (batch, n_heads, 64, d_head)

        Returns:
            Relative position bias, (batch, n_heads, 64, 64), pre-scaled by 1/sqrt(d_head)
        """
        # Look up relative position embeddings
        # P_ij[i,j] encodes position of i relative to j
        P_ij = self.E_rel_row(self.rel_row_idx_ij) + self.E_rel_col(  # noqa: N806
            self.rel_col_idx_ij
        )  # (64, 64, d_model)

        # P_ji[i,j] encodes position of j relative to i (for P2C)
        P_ji = self.E_rel_row(self.rel_row_idx_ji) + self.E_rel_col(  # noqa: N806
            self.rel_col_idx_ji
        )  # (64, 64, d_model)

        # Project relative positions
        # K_r[i,j] encodes "where is i relative to j?" (for C2P term)
        K_r = self.w_kr(P_ij)  # noqa: N806  # (64, 64, d_model)
        K_r = K_r.view(N_SQUARES, N_SQUARES, self.n_heads, self.d_head)  # noqa: N806
        K_r = K_r.permute(2, 0, 1, 3)  # noqa: N806  # (n_heads, 64, 64, d_head)

        # Q_r[i,j] encodes "where is j relative to i?" (for P2C term)
        Q_r = self.w_qr(P_ji)  # noqa: N806  # (64, 64, d_model)
        Q_r = Q_r.view(N_SQUARES, N_SQUARES, self.n_heads, self.d_head)  # noqa: N806
        Q_r = Q_r.permute(2, 0, 1, 3)  # noqa: N806  # (n_heads, 64, 64, d_head)

        # C2P: Q_c[i] · K_r[i,j]
        # Q_board: (batch, n_heads, 64, d_head), K_r: (n_heads, 64, 64, d_head)
        c2p = torch.einsum("bhid,hijd->bhij", Q_board, K_r)  # (batch, n_heads, 64, 64)

        # P2C: Q_r[i,j] · K_c[j]
        # Q_r: (n_heads, 64, 64, d_head), K_board: (batch, n_heads, 64, d_head)
        p2c = torch.einsum("hijd,bhjd->bhij", Q_r, K_board)  # (batch, n_heads, 64, 64)

        # Scale by 1/sqrt(d_head) to match SDPA scaling
        return (c2p + p2c) * self.scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor, (batch, 69, d_model) where 69 = 5 metadata + 64 board squares

        Returns:
            Output tensor, (batch, 69, d_model)
        """
        batch_size, seq_len, _ = x.shape

        # Project to Q, K, V
        Q = self.w_q(x).view(  # noqa: N806
            batch_size, seq_len, self.n_heads, self.d_head
        )
        K = self.w_k(x).view(  # noqa: N806
            batch_size, seq_len, self.n_heads, self.d_head
        )
        V = self.w_v(x).view(  # noqa: N806
            batch_size, seq_len, self.n_heads, self.d_head
        )

        # Transpose to (batch, n_heads, seq, d_head)
        Q = Q.transpose(1, 2)  # noqa: N806
        K = K.transpose(1, 2)  # noqa: N806
        V = V.transpose(1, 2)  # noqa: N806

        # Extract board square Q and K for relative position bias
        Q_board = Q[:, :, N_METADATA:, :]  # noqa: N806  # (batch, n_heads, 64, d_head)
        K_board = K[:, :, N_METADATA:, :]  # noqa: N806  # (batch, n_heads, 64, d_head)

        # Compute relative position bias for board squares
        rel_pos_bias = self._compute_rel_pos_bias(
            Q_board, K_board
        )  # (batch, n_heads, 64, 64)

        # Build full attention bias matrix (69 x 69)
        # Only board↔board positions get the relative position bias
        attn_bias = x.new_zeros(batch_size, self.n_heads, seq_len, seq_len)
        attn_bias[:, :, N_METADATA:, N_METADATA:] = rel_pos_bias

        # Scaled dot-product attention with additive bias
        # SDPA computes: softmax(Q @ K.T / sqrt(d) + attn_mask) @ V
        dropout_p = self.dropout_p if self.training else 0.0
        attn_output = F.scaled_dot_product_attention(
            Q, K, V, attn_mask=attn_bias, dropout_p=dropout_p
        )  # (batch, n_heads, seq, d_head)

        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.d_model)

        return self.w_out(attn_output)


class TransformerEncoderLayer(nn.Module):
    """Single transformer encoder layer with pre-norm architecture.

    Implements the pre-normalization variant of a transformer encoder layer,
    which applies layer normalization before (rather than after) each sub-layer.
    This architecture is more stable for deep networks and is used in GPT-2,
    LLaMA, and other modern transformers.

    Structure::

        x = x + Attention(RMSNorm(x))
        x = x + FFN(RMSNorm(x))

    Attributes:
        norm1 (RMSNorm): Normalization before attention.
        attention (DisentangledAttention): Multi-head attention with relative positions.
        norm2 (RMSNorm): Normalization before feed-forward.
        ffn (SwiGLUFFN): Feed-forward network with SwiGLU activation.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.0,
        rel_pos_embeddings: tuple = None,
    ) -> None:
        """Initialize transformer encoder layer.

        Args:
            d_model: Model dimension.
            n_heads: Number of attention heads.
            d_ff: Feed-forward hidden dimension.
            dropout: Dropout probability for attention and FFN. Defaults to 0.0.
            rel_pos_embeddings: Optional tuple of (E_rel_row, E_rel_col) for
                sharing relative position embeddings across layers.
        """
        super().__init__()

        self.norm1 = RMSNorm(d_model)
        self.attention = DisentangledAttention(
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout,
            rel_pos_embeddings=rel_pos_embeddings,
        )

        self.norm2 = RMSNorm(d_model)
        self.ffn = SwiGLUFFN(d_model=d_model, d_ff=d_ff, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply transformer encoder layer.

        Args:
            x: Input tensor of shape (batch, seq, d_model).

        Returns:
            Output tensor of shape (batch, seq, d_model).
        """
        # Pre-norm attention
        x = x + self.attention(self.norm1(x))
        # Pre-norm FFN
        x = x + self.ffn(self.norm2(x))
        return x


class BoardEncoder(nn.Module):
    """Transformer encoder for chess board state representation.

    Processes a chess board state (turn, castling rights, and piece positions)
    into contextualized embeddings. The output can be used for downstream tasks
    like move prediction or position evaluation.

    The input sequence has 69 tokens:
        - 5 metadata tokens: turn, white kingside castling, white queenside
          castling, black kingside castling, black queenside castling
        - 64 board square tokens: pieces at each square (a1, b1, ..., h8)

    Positional Embeddings:
        - Metadata tokens use independent learned embeddings (one per token type).
        - Board squares use decomposed file + rank embeddings, providing 2D
          inductive bias. This means a1 and a2 share the same file embedding,
          while a1 and b1 share the same rank embedding.

    Attributes:
        d_model (int): Model dimension.
        n_layers (int): Number of transformer layers.
        turn_embedding (nn.Embedding): Embedding for turn indicator.
        wk_castling_embedding (nn.Embedding): White kingside castling embedding.
        wq_castling_embedding (nn.Embedding): White queenside castling embedding.
        bk_castling_embedding (nn.Embedding): Black kingside castling embedding.
        bq_castling_embedding (nn.Embedding): Black queenside castling embedding.
        piece_embedding (nn.Embedding): Embedding for piece types.
        metadata_pos_embedding (nn.Embedding): Positional embeddings for metadata.
        file_embedding (nn.Embedding): File (column) embeddings for board squares.
        rank_embedding (nn.Embedding): Rank (row) embeddings for board squares.
        layers (nn.ModuleList): Stack of transformer encoder layers.
        final_norm (RMSNorm): Final layer normalization.
        dropout (nn.Dropout): Dropout layer for embeddings.

    Example:
        >>> vocab_sizes = {
        ...     "turn": 2,
        ...     "white_kingside_castling_rights": 2,
        ...     "white_queenside_castling_rights": 2,
        ...     "black_kingside_castling_rights": 2,
        ...     "black_queenside_castling_rights": 2,
        ...     "board_position": 13,  # empty + 6 white + 6 black pieces
        ... }
        >>> encoder = BoardEncoder(vocab_sizes, d_model=256, n_heads=8, d_ff=512, n_layers=6)
        >>> # Input shapes: (batch, 1) for metadata, (batch, 64) for board
        >>> output = encoder(turns, wk, wq, bk, bq, board_positions)
        >>> output.shape
        torch.Size([batch, 69, 256])
    """

    def __init__(
        self,
        vocab_sizes: dict,
        d_model: int,
        n_heads: int,
        d_ff: int,
        n_layers: int,
        dropout: float = 0.0,
        share_rel_pos_embeddings: bool = False,
    ) -> None:
        """Initialize the board encoder.

        Args:
            vocab_sizes: Dictionary mapping token type names to vocabulary sizes.
                Required keys:

                - ``"turn"``: 2 (white=0, black=1)
                - ``"white_kingside_castling_rights"``: 2 (no=0, yes=1)
                - ``"white_queenside_castling_rights"``: 2 (no=0, yes=1)
                - ``"black_kingside_castling_rights"``: 2 (no=0, yes=1)
                - ``"black_queenside_castling_rights"``: 2 (no=0, yes=1)
                - ``"board_position"``: Number of piece types (typically 13:
                  empty + 6 white pieces + 6 black pieces)

            d_model: Model/embedding dimension. Must be divisible by n_heads.
            n_heads: Number of attention heads in each layer.
            d_ff: Hidden dimension of the feed-forward network.
            n_layers: Number of transformer encoder layers.
            dropout: Dropout probability applied to embeddings and attention.
                Defaults to 0.0.
            share_rel_pos_embeddings: If True, share relative position embeddings
                across all transformer layers (reduces parameters). If False,
                each layer has its own embeddings. Defaults to False.

        Raises:
            KeyError: If vocab_sizes is missing required keys.
            AssertionError: If d_model is not divisible by n_heads.
        """
        super().__init__()

        self.d_model = d_model
        self.n_layers = n_layers

        # Content embeddings (what is at each position)
        self.turn_embedding = nn.Embedding(vocab_sizes["turn"], d_model)
        self.wk_castling_embedding = nn.Embedding(
            vocab_sizes["white_kingside_castling_rights"], d_model
        )
        self.wq_castling_embedding = nn.Embedding(
            vocab_sizes["white_queenside_castling_rights"], d_model
        )
        self.bk_castling_embedding = nn.Embedding(
            vocab_sizes["black_kingside_castling_rights"], d_model
        )
        self.bq_castling_embedding = nn.Embedding(
            vocab_sizes["black_queenside_castling_rights"], d_model
        )
        self.piece_embedding = nn.Embedding(vocab_sizes["board_position"], d_model)

        # Absolute positional embeddings
        # Metadata tokens: independent learned embeddings
        self.metadata_pos_embedding = nn.Embedding(N_METADATA, d_model)
        # Board squares: decomposed file (column) + rank (row) embeddings
        self.file_embedding = nn.Embedding(BOARD_SIZE, d_model)  # a-h
        self.rank_embedding = nn.Embedding(BOARD_SIZE, d_model)  # 1-8

        # Pre-compute file and rank indices for each square
        # Square i: file = i % 8, rank = i // 8
        files = (
            torch.arange(BOARD_SIZE)
            .unsqueeze(0)
            .expand(BOARD_SIZE, BOARD_SIZE)
            .reshape(N_SQUARES)
        )
        ranks = (
            torch.arange(BOARD_SIZE)
            .unsqueeze(1)
            .expand(BOARD_SIZE, BOARD_SIZE)
            .reshape(N_SQUARES)
        )
        self.register_buffer("file_indices", files)
        self.register_buffer("rank_indices", ranks)

        # Shared relative position embeddings (if enabled)
        self.share_rel_pos_embeddings = share_rel_pos_embeddings
        if share_rel_pos_embeddings:
            self.E_rel_row = nn.Embedding(MAX_REL_DIST, d_model)
            self.E_rel_col = nn.Embedding(MAX_REL_DIST, d_model)
            rel_pos_embeds = (self.E_rel_row, self.E_rel_col)
        else:
            rel_pos_embeds = None

        # Transformer layers
        self.layers = nn.ModuleList(
            [
                TransformerEncoderLayer(
                    d_model=d_model,
                    n_heads=n_heads,
                    d_ff=d_ff,
                    dropout=dropout,
                    rel_pos_embeddings=rel_pos_embeds,
                )
                for _ in range(n_layers)
            ]
        )

        # Final normalization
        self.final_norm = RMSNorm(d_model)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        turns: torch.Tensor,
        white_kingside_castling_rights: torch.Tensor,
        white_queenside_castling_rights: torch.Tensor,
        black_kingside_castling_rights: torch.Tensor,
        black_queenside_castling_rights: torch.Tensor,
        board_positions: torch.Tensor,
    ) -> torch.Tensor:
        """Encode a batch of chess board states.

        Args:
            turns: Current turn indicator of shape (batch, 1).
                Values: 0 for white to move, 1 for black to move.
            white_kingside_castling_rights: White kingside castling availability
                of shape (batch, 1). Values: 0 for unavailable, 1 for available.
            white_queenside_castling_rights: White queenside castling availability
                of shape (batch, 1). Values: 0 for unavailable, 1 for available.
            black_kingside_castling_rights: Black kingside castling availability
                of shape (batch, 1). Values: 0 for unavailable, 1 for available.
            black_queenside_castling_rights: Black queenside castling availability
                of shape (batch, 1). Values: 0 for unavailable, 1 for available.
            board_positions: Piece type at each of the 64 squares, shape (batch, 64).
                Squares are ordered a1, b1, c1, ..., h8. Values are indices into
                the piece vocabulary (e.g., 0=empty, 1-6=white pieces, 7-12=black).

        Returns:
            Contextualized embeddings of shape (batch, 69, d_model).
            The sequence is ordered as: [turn, wk_castling, wq_castling,
            bk_castling, bq_castling, a1, b1, ..., h8].
        """
        # Content embeddings for metadata tokens
        turn_emb = self.turn_embedding(turns)  # (batch, 1, d_model)
        wk_emb = self.wk_castling_embedding(white_kingside_castling_rights)
        wq_emb = self.wq_castling_embedding(white_queenside_castling_rights)
        bk_emb = self.bk_castling_embedding(black_kingside_castling_rights)
        bq_emb = self.bq_castling_embedding(black_queenside_castling_rights)

        # Content embeddings for board squares
        piece_emb = self.piece_embedding(board_positions)  # (batch, 64, d_model)

        # Absolute positional embeddings for metadata
        metadata_pos = self.metadata_pos_embedding.weight.unsqueeze(
            0
        )  # (1, 5, d_model)

        # Absolute positional embeddings for board squares (file + rank)
        file_emb = self.file_embedding(self.file_indices)  # (64, d_model)
        rank_emb = self.rank_embedding(self.rank_indices)  # (64, d_model)
        board_pos = (file_emb + rank_emb).unsqueeze(0)  # (1, 64, d_model)

        # Combine content and positional embeddings
        metadata = torch.cat(
            [turn_emb, wk_emb, wq_emb, bk_emb, bq_emb], dim=1
        )  # (batch, 5, d_model)
        metadata = metadata + metadata_pos  # Add positional embeddings

        board = piece_emb + board_pos  # (batch, 64, d_model)

        # Concatenate metadata and board
        x = torch.cat([metadata, board], dim=1)  # (batch, 69, d_model)

        # Scale embeddings
        x = x * math.sqrt(self.d_model)

        # Dropout
        x = self.dropout(x)

        # Transformer layers
        for layer in self.layers:
            x = layer(x)

        # Final normalization
        x = self.final_norm(x)

        return x
