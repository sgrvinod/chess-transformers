import math
import torch
from torch import nn

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


class MultiHeadAttention(nn.Module):
    """
    The Multi-Head Attention sublayer.

    Reused from https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Machine-Translation.
    """

    def __init__(
        self, d_model, n_heads, d_queries, d_values, dropout, in_decoder=False
    ):
        """
        Init.

        Args:

            d_model (int): The size of vectors throughout the
            transformer model, i.e. input and output sizes for this
            sublayer.

            n_heads (int): The number of heads in the multi-head
            attention.

            d_queries (int): The size of query vectors (and also the
            size of the key vectors).

            d_values (int): The size of value vectors.

            dropout (float): The dropout probability.

            in_decoder (bool, optional): Is this Multi-Head Attention
            sublayer instance in the Decoder? Defaults to False.
        """
        super(MultiHeadAttention, self).__init__()

        self.d_model = d_model
        self.n_heads = n_heads

        self.d_queries = d_queries
        self.d_values = d_values
        self.d_keys = d_queries  # size of key vectors, same as of the query vectors to allow dot-products for similarity

        self.in_decoder = in_decoder

        # A linear projection to cast (n_heads sets of) queries from the
        # input query sequences
        self.cast_queries = nn.Linear(d_model, n_heads * d_queries)

        # A linear projection to cast (n_heads sets of) keys and values
        # from the input reference sequences
        self.cast_keys_values = nn.Linear(d_model, n_heads * (d_queries + d_values))

        # A linear projection to cast (n_heads sets of) computed
        # attention-weighted vectors to output vectors (of the same size
        # as input query vectors)
        self.cast_output = nn.Linear(n_heads * d_values, d_model)

        # Softmax layer
        self.softmax = nn.Softmax(dim=-1)

        # Layer-norm layer
        self.layer_norm = nn.LayerNorm(d_model)

        # Dropout layer
        self.apply_dropout = nn.Dropout(dropout)

    def forward(self, query_sequences, key_value_sequences, key_value_sequence_lengths):
        """
        Forward prop.

        Args:

            query_sequences (torch.FloatTensor): The input query
            sequences, of size (N, query_sequence_pad_length, d_model).

            key_value_sequences (torch.FloatTensor): The sequences to be
            queried against, of size (N, key_value_sequence_pad_length,
            d_model).

            key_value_sequence_lengths (torch.LongTensor): The true
            lengths of the key_value_sequences, to be able to ignore
            pads, of size (N).

        Returns:

            torch.FloatTensor: Attention-weighted output sequences for
            the query sequences, of size (N, query_sequence_pad_length,
            d_model).
        """
        batch_size = query_sequences.size(0)  # batch size (N) in number of sequences
        query_sequence_pad_length = query_sequences.size(1)
        key_value_sequence_pad_length = key_value_sequences.size(1)

        # Is this self-attention?
        self_attention = torch.equal(key_value_sequences, query_sequences)

        # Store input for adding later
        input_to_add = query_sequences.clone()

        # Apply layer normalization
        query_sequences = self.layer_norm(
            query_sequences
        )  # (N, query_sequence_pad_length, d_model)
        # If this is self-attention, do the same for the key-value
        # sequences (as they are the same as the query sequences) If
        # this isn't self-attention, they will already have been normed
        # in the last layer of the Encoder (from whence they came)
        if self_attention:
            key_value_sequences = self.layer_norm(
                key_value_sequences
            )  # (N, key_value_sequence_pad_length, d_model)

        # Project input sequences to queries, keys, values
        queries = self.cast_queries(
            query_sequences
        )  # (N, query_sequence_pad_length, n_heads * d_queries)
        keys, values = self.cast_keys_values(key_value_sequences).split(
            split_size=self.n_heads * self.d_keys, dim=-1
        )  # (N, key_value_sequence_pad_length, n_heads * d_keys), (N, key_value_sequence_pad_length, n_heads * d_values)

        # Split the last dimension by the n_heads subspaces
        queries = queries.contiguous().view(
            batch_size, query_sequence_pad_length, self.n_heads, self.d_queries
        )  # (N, query_sequence_pad_length, n_heads, d_queries)
        keys = keys.contiguous().view(
            batch_size, key_value_sequence_pad_length, self.n_heads, self.d_keys
        )  # (N, key_value_sequence_pad_length, n_heads, d_keys)
        values = values.contiguous().view(
            batch_size, key_value_sequence_pad_length, self.n_heads, self.d_values
        )  # (N, key_value_sequence_pad_length, n_heads, d_values)

        # Re-arrange axes such that the last two dimensions are the
        # sequence lengths and the queries/keys/values And then, for
        # convenience, convert to 3D tensors by merging the batch and
        # n_heads dimensions This is to prepare it for the batch matrix
        # multiplication (i.e. the dot product)
        queries = (
            queries.permute(0, 2, 1, 3)
            .contiguous()
            .view(-1, query_sequence_pad_length, self.d_queries)
        )  # (N * n_heads, query_sequence_pad_length, d_queries)
        keys = (
            keys.permute(0, 2, 1, 3)
            .contiguous()
            .view(-1, key_value_sequence_pad_length, self.d_keys)
        )  # (N * n_heads, key_value_sequence_pad_length, d_keys)
        values = (
            values.permute(0, 2, 1, 3)
            .contiguous()
            .view(-1, key_value_sequence_pad_length, self.d_values)
        )  # (N * n_heads, key_value_sequence_pad_length, d_values)

        # Perform multi-head attention

        # Perform dot-products
        attention_weights = torch.bmm(
            queries, keys.permute(0, 2, 1)
        )  # (N * n_heads, query_sequence_pad_length, key_value_sequence_pad_length)

        # Scale dot-products
        attention_weights = (
            1.0 / math.sqrt(self.d_keys)
        ) * attention_weights  # (N * n_heads, query_sequence_pad_length, key_value_sequence_pad_length)

        # Before computing softmax weights, prevent queries from
        # attending to certain keys

        # MASK 1: keys that are pads
        not_pad_in_keys = (
            torch.LongTensor(range(key_value_sequence_pad_length))
            .unsqueeze(0)
            .unsqueeze(0)
            .expand_as(attention_weights)
            .to(DEVICE)
        )  # (N * n_heads, query_sequence_pad_length, key_value_sequence_pad_length)
        not_pad_in_keys = (
            not_pad_in_keys
            < key_value_sequence_lengths.repeat_interleave(self.n_heads)
            .unsqueeze(1)
            .unsqueeze(2)
            .expand_as(attention_weights)
        )  # (N * n_heads, query_sequence_pad_length, key_value_sequence_pad_length)
        # Note: PyTorch auto-broadcasts singleton dimensions in
        # comparison operations (as well as arithmetic operations)

        # Mask away by setting such weights to a large negative number,
        # so that they evaluate to 0 under the softmax
        attention_weights = attention_weights.masked_fill(
            ~not_pad_in_keys, -float("inf")
        )  # (N * n_heads, query_sequence_pad_length, key_value_sequence_pad_length)

        # MASK 2: if this is self-attention in the Decoder, keys
        # chronologically ahead of queries
        if self.in_decoder and self_attention:
            # Therefore, a position [n, i, j] is valid only if j <= i
            # torch.tril(), i.e. lower triangle in a 2D matrix, sets j >
            # i to 0
            not_future_mask = (
                torch.ones_like(attention_weights).tril().bool().to(DEVICE)
            )  # (N * n_heads, query_sequence_pad_length, key_value_sequence_pad_length)

            # Mask away by setting such weights to a large negative
            # number, so that they evaluate to 0 under the softmax
            attention_weights = attention_weights.masked_fill(
                ~not_future_mask, -float("inf")
            )  # (N * n_heads, query_sequence_pad_length, key_value_sequence_pad_length)

        # Compute softmax along the key dimension
        attention_weights = self.softmax(
            attention_weights
        )  # (N * n_heads, query_sequence_pad_length, key_value_sequence_pad_length)

        # Apply dropout
        attention_weights = self.apply_dropout(
            attention_weights
        )  # (N * n_heads, query_sequence_pad_length, key_value_sequence_pad_length)

        # Calculate sequences as the weighted sums of values based on
        # these softmax weights
        sequences = torch.bmm(
            attention_weights, values
        )  # (N * n_heads, query_sequence_pad_length, d_values)

        # Unmerge batch and n_heads dimensions and restore original
        # order of axes
        sequences = (
            sequences.contiguous()
            .view(batch_size, self.n_heads, query_sequence_pad_length, self.d_values)
            .permute(0, 2, 1, 3)
        )  # (N, query_sequence_pad_length, n_heads, d_values)

        # Concatenate the n_heads subspaces (each with an output of size
        # d_values)
        sequences = sequences.contiguous().view(
            batch_size, query_sequence_pad_length, -1
        )  # (N, query_sequence_pad_length, n_heads * d_values)

        # Transform the concatenated subspace-sequences into a single
        # output of size d_model
        sequences = self.cast_output(
            sequences
        )  # (N, query_sequence_pad_length, d_model)

        # Apply dropout and residual connection
        sequences = (
            self.apply_dropout(sequences) + input_to_add
        )  # (N, query_sequence_pad_length, d_model)

        return sequences


class PositionWiseFCNetwork(nn.Module):
    """
    The Position-Wise Feed Forward Network sublayer.

    Reused from https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Machine-Translation.
    """

    def __init__(self, d_model, d_inner, dropout):
        """
        Init.

        Args:

            d_model (int): The size of vectors throughout the
            transformer model, i.e. input and output sizes for this
            sublayer.

            d_inner (int): An intermediate size.

            dropout (float): The dropout probability.
        """
        super(PositionWiseFCNetwork, self).__init__()

        self.d_model = d_model
        self.d_inner = d_inner

        # Layer-norm layer
        self.layer_norm = nn.LayerNorm(d_model)

        # A linear layer to project from the input size to an
        # intermediate size
        self.fc1 = nn.Linear(d_model, d_inner)

        # ReLU
        self.relu = nn.ReLU()

        # A linear layer to project from the intermediate size to the
        # output size (same as the input size)
        self.fc2 = nn.Linear(d_inner, d_model)

        # Dropout layer
        self.apply_dropout = nn.Dropout(dropout)

    def forward(self, sequences):
        """
        Forward prop.

        Args:

            sequences (torch.FloatTensor): The input sequences, of size
            (N, pad_length, d_model).

        Returns:

            torch.FloatTensor: The transformed output sequences, of size
            (N, pad_length, d_model).
        """

        # Store input for adding later
        input_to_add = sequences.clone()  # (N, pad_length, d_model)

        # Apply layer-norm
        sequences = self.layer_norm(sequences)  # (N, pad_length, d_model)

        # Transform position-wise
        sequences = self.apply_dropout(
            self.relu(self.fc1(sequences))
        )  # (N, pad_length, d_inner)
        sequences = self.fc2(sequences)  # (N, pad_length, d_model)

        # Apply dropout and residual connection
        sequences = (
            self.apply_dropout(sequences) + input_to_add
        )  # (N, pad_length, d_model)

        return sequences


class BoardEncoder(nn.Module):
    """
    The Board Encoder.

    Adapted from https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Machine-Translation.
    """

    def __init__(
        self,
        vocab_sizes,
        d_model,
        n_heads,
        d_queries,
        d_values,
        d_inner,
        n_layers,
        dropout,
    ):
        """
        Init.

        Args:

            vocab_sizes (dict): The vocabulary sizes of input sequence
            components.

            d_model (int): The size of vectors throughout the
            transformer model, i.e. input and output sizes for the
            Encoder.

            n_heads (int): The number of heads in the multi-head
            attention.

            d_queries (int): The size of query vectors (and also the
            size of the key vectors) in the multi-head attention.

            d_values (int): The size of value vectors in the multi-head
            attention.

            d_inner (int): An intermediate size in the position-wise FC.

            n_layers (int): The number of [multi-head attention +
            position-wise FC] layers in the Encoder.

            dropout (float): The dropout probability.
        """
        super(BoardEncoder, self).__init__()

        self.vocab_sizes = vocab_sizes
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_queries = d_queries
        self.d_values = d_values
        self.d_inner = d_inner
        self.n_layers = n_layers
        self.dropout = dropout

        # Embedding layers
        self.turn_embeddings = nn.Embedding(vocab_sizes["turn"], d_model)
        self.white_kingside_castling_rights_embeddings = nn.Embedding(
            vocab_sizes["white_kingside_castling_rights"], d_model
        )
        self.white_queenside_castling_rights_embeddings = nn.Embedding(
            vocab_sizes["white_queenside_castling_rights"], d_model
        )
        self.black_kingside_castling_rights_embeddings = nn.Embedding(
            vocab_sizes["black_kingside_castling_rights"], d_model
        )
        self.black_queenside_castling_rights_embeddings = nn.Embedding(
            vocab_sizes["black_queenside_castling_rights"], d_model
        )
        self.board_position_embeddings = nn.Embedding(
            vocab_sizes["board_position"], d_model
        )

        # Positional embedding layer
        self.positional_embeddings = nn.Embedding(
            69,
            d_model,
        )

        # Encoder layers
        self.encoder_layers = nn.ModuleList(
            [self.make_encoder_layer() for i in range(n_layers)]
        )

        # Dropout layer
        self.apply_dropout = nn.Dropout(dropout)

        # Layer-norm layer
        self.layer_norm = nn.LayerNorm(d_model)

    def make_encoder_layer(self):
        """
        Creates a single layer in the Encoder by combining a multi-head
        attention sublayer and a position-wise FC sublayer.
        """
        # A ModuleList of sublayers
        encoder_layer = nn.ModuleList(
            [
                MultiHeadAttention(
                    d_model=self.d_model,
                    n_heads=self.n_heads,
                    d_queries=self.d_queries,
                    d_values=self.d_values,
                    dropout=self.dropout,
                    in_decoder=False,
                ),
                PositionWiseFCNetwork(
                    d_model=self.d_model, d_inner=self.d_inner, dropout=self.dropout
                ),
            ]
        )

        return encoder_layer

    def forward(
        self,
        turns,
        white_kingside_castling_rights,
        white_queenside_castling_rights,
        black_kingside_castling_rights,
        black_queenside_castling_rights,
        board_positions,
    ):
        """
        Forward prop.

        Args:

            turns (torch.LongTensor): The current turn (w/b), of size
            (N, 1).

            white_kingside_castling_rights (torch.LongTensor): Whether
            white can castle kingside, of size (N, 1).

            white_queenside_castling_rights (torch.LongTensor): Whether
            white can castle queenside, of size (N, 1).

            black_kingside_castling_rights (torch.LongTensor): Whether
            black can castle kingside, of size (N, 1).

            black_queenside_castling_rights (torch.LongTensor): Whether
            black can castle queenside, of size (N, 1).

            board_positions (torch.LongTensor): The current board
            positions, of size (N, 64).

        Returns:

            torch.FloatTensor: The encoded board, of size (N,
            BOARD_STATUS_LENGTH, d_model).
        """
        batch_size = turns.size(0)  # N

        # Embeddings
        embeddings = torch.cat(
            [
                self.turn_embeddings(turns),
                self.white_kingside_castling_rights_embeddings(
                    white_kingside_castling_rights
                ),
                self.white_queenside_castling_rights_embeddings(
                    white_queenside_castling_rights
                ),
                self.black_kingside_castling_rights_embeddings(
                    black_kingside_castling_rights
                ),
                self.black_queenside_castling_rights_embeddings(
                    black_queenside_castling_rights
                ),
                self.board_position_embeddings(board_positions),
            ],
            dim=1,
        )  # (N, BOARD_STATUS_LENGTH, d_model)

        # Add positional embeddings
        boards = embeddings + self.positional_embeddings.weight.unsqueeze(
            0
        )  # (N, BOARD_STATUS_LENGTH, d_model)
        boards = boards * math.sqrt(self.d_model)  # (N, BOARD_STATUS_LENGTH, d_model)

        # Dropout
        boards = self.apply_dropout(boards)  # (N, BOARD_STATUS_LENGTH, d_model)

        # Encoder layers
        for encoder_layer in self.encoder_layers:
            # Sublayers
            boards = encoder_layer[0](
                query_sequences=boards,
                key_value_sequences=boards,
                key_value_sequence_lengths=torch.LongTensor([69] * batch_size).to(
                    DEVICE
                ),
            )  # (N, BOARD_STATUS_LENGTH, d_model)
            boards = encoder_layer[1](
                sequences=boards
            )  # (N, BOARD_STATUS_LENGTH, d_model)

        # Apply layer-norm
        boards = self.layer_norm(boards)  # (N, BOARD_STATUS_LENGTH, d_model)

        return boards


class MoveDecoder(nn.Module):
    """
    The Move Decoder.

    Adapted from https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Machine-Translation.
    """

    def __init__(
        self,
        vocab_size,
        n_moves,
        d_model,
        n_heads,
        d_queries,
        d_values,
        d_inner,
        n_layers,
        dropout,
    ):
        """
        Init.

        Args:

            vocab_size (int): The size of the output vocabulary.

            n_moves (int): The expected maximum length of output (move)
            sequences.

            d_model (int): The size of vectors throughout the
            transformer model, i.e. input and output sizes for the
            Decoder.

            n_heads (int): The number of heads in the multi-head
            attention.

            d_queries (int): The size of query vectors (and also the
            size of the key vectors) in the multi-head attention.

            d_values (int): The size of value vectors in the multi-head
            attention.

            d_inner (int): An intermediate size in the position-wise FC.

            n_layers (int): The number of [multi-head attention +
            multi-head attention + position-wise FC] layers in the
            Decoder.

            dropout (int): The dropout probability.
        """
        super(MoveDecoder, self).__init__()

        self.vocab_size = vocab_size
        self.n_moves = n_moves
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_queries = d_queries
        self.d_values = d_values
        self.d_inner = d_inner
        self.n_layers = n_layers
        self.dropout = dropout

        # Embedding layer
        self.embeddings = nn.Embedding(vocab_size, d_model)

        # Positional embedding layer
        self.positional_embeddings = nn.Embedding(n_moves, d_model)

        # Decoder layers
        self.decoder_layers = nn.ModuleList(
            [self.make_decoder_layer() for i in range(n_layers)]
        )

        # Dropout layer
        self.apply_dropout = nn.Dropout(dropout)

        # Layer-norm layer
        self.layer_norm = nn.LayerNorm(d_model)

        # Output linear layer that will compute logits for the
        # vocabulary
        self.fc = nn.Linear(d_model, vocab_size)

    def make_decoder_layer(self):
        """
        Creates a single layer in the Decoder by combining two
        multi-head attention sublayers and a position-wise FC sublayer.
        """
        # A ModuleList of sublayers
        decoder_layer = nn.ModuleList(
            [
                MultiHeadAttention(
                    d_model=self.d_model,
                    n_heads=self.n_heads,
                    d_queries=self.d_queries,
                    d_values=self.d_values,
                    dropout=self.dropout,
                    in_decoder=True,
                ),
                MultiHeadAttention(
                    d_model=self.d_model,
                    n_heads=self.n_heads,
                    d_queries=self.d_queries,
                    d_values=self.d_values,
                    dropout=self.dropout,
                    in_decoder=True,
                ),
                PositionWiseFCNetwork(
                    d_model=self.d_model, d_inner=self.d_inner, dropout=self.dropout
                ),
            ]
        )

        return decoder_layer

    def forward(
        self,
        moves,
        lengths,
        boards,
    ):
        """
        Forward prop.

        Args:

            moves (torch.LongTensor): The move sequences, of size (N,
            n_moves).

            lengths (torch.LongTensor): The true lengths of the move
            sequences, not including <move> and <pad> tokens, of size
            (N, 1).

            boards (torch.FloatTensor): The encoded boards, from the
            Encoder, of size (N, BOARD_STATUS_LENGTH, d_model).

        Returns:

            torch.FloatTensor: The decoded next-move probabilities, of
            size (N, n_moves, vocab_size).
        """
        batch_size = boards.size(0)  # N

        # Embeddings
        embeddings = self.embeddings(moves)  # (N, n_moves, d_model)

        # Add positional embeddings
        moves = embeddings + self.positional_embeddings.weight.unsqueeze(
            0
        )  # (N, n_moves, d_model)
        moves = moves * math.sqrt(self.d_model)  # (N, n_moves, d_model)

        # Dropout
        moves = self.apply_dropout(moves)

        # Decoder layers
        for decoder_layer in self.decoder_layers:
            # Sublayers
            moves = decoder_layer[0](
                query_sequences=moves,
                key_value_sequences=moves,
                key_value_sequence_lengths=lengths,
            )  # (N, n_moves, d_model)
            moves = decoder_layer[1](
                query_sequences=moves,
                key_value_sequences=boards,
                key_value_sequence_lengths=torch.LongTensor([69] * batch_size).to(
                    DEVICE
                ),
            )  # (N, n_moves, d_model)
            moves = decoder_layer[2](sequences=moves)  # (N, n_moves, d_model)

        # Apply layer-norm
        moves = self.layer_norm(moves)  # (N, n_moves, d_model)

        # Find logits over vocabulary
        moves = self.fc(moves)  # (N, n_moves, vocab_size)

        return moves
