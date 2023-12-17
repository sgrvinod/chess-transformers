import math
import torch
import argparse
from torch import nn

from chess_transformers.configs import import_config
from chess_transformers.transformers.modules import BoardEncoder, MoveDecoder

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


class ChessTransformer(nn.Module):
    """
    The Chess Transformer, for predicting the next move and sequence of
    moves likely to follow.
    """

    def __init__(
        self,
        CONFIG,
    ):
        """
        Init.

        Args:

            CONFIG (dict): The configuration, containing the following
            parameters for the model:

                VOCAB_SIZES (dict): The sizes of the vocabularies of the
                Encoder sequence components and the Decoder (move)
                sequence.

                N_MOVES (int): The expected maximum length of output
                (move) sequences.

                D_MODEL (int): The size of vectors throughout the
                transformer model, i.e. input and output sizes for the
                Encoder and Decoder.

                N_HEADS (int): The number of heads in the multi-head
                attention.

                D_QUERIES (int): The size of query vectors (and also the
                size of the key vectors) in the multi-head attention.

                D_VALUES (int): The size of value vectors in the
                multi-head attention.

                D_INNER (int): An intermediate size in the position-wise
                FC.

                N_LAYERS (int): The number of [multi-head attention +
                multi-head attention + position-wise FC] layers in the
                Encoder and Decoder.

                DROPOUT (int): The dropout probability.
        """
        super(ChessTransformer, self).__init__()

        self.code = "ED"

        self.vocab_sizes = CONFIG.VOCAB_SIZES
        self.n_moves = CONFIG.N_MOVES
        self.d_model = CONFIG.D_MODEL
        self.n_heads = CONFIG.N_HEADS
        self.d_queries = CONFIG.D_QUERIES
        self.d_values = CONFIG.D_VALUES
        self.d_inner = CONFIG.D_INNER
        self.n_layers = CONFIG.N_LAYERS
        self.dropout = CONFIG.DROPOUT

        # Encoder
        self.board_encoder = BoardEncoder(
            vocab_sizes=self.vocab_sizes,
            d_model=self.d_model,
            n_heads=self.n_heads,
            d_queries=self.d_queries,
            d_values=self.d_values,
            d_inner=self.d_inner,
            n_layers=self.n_layers,
            dropout=self.dropout,
        )

        # Decoder
        self.move_decoder = MoveDecoder(
            vocab_size=self.vocab_sizes["moves"],
            n_moves=self.n_moves,
            d_model=self.d_model,
            n_heads=self.n_heads,
            d_queries=self.d_queries,
            d_values=self.d_values,
            d_inner=self.d_inner,
            n_layers=self.n_layers,
            dropout=self.dropout,
        )

        # Initialize weights
        self.init_weights()

    def init_weights(self):
        """
        Initialize weights in the transformer model.
        """
        # Glorot uniform initialization with a gain of 1.
        for p in self.parameters():
            # Glorot initialization needs at least two dimensions on the
            # tensor
            if p.dim() > 1:
                nn.init.xavier_uniform_(p, gain=1.0)

        # For the embeddings, normal initialization with 0 mean and
        # 1/sqrt(d_model) S.D.
        nn.init.normal_(
            self.board_encoder.board_position_embeddings.weight,
            mean=0.0,
            std=math.pow(self.d_model, -0.5),
        )
        nn.init.normal_(
            self.board_encoder.turn_embeddings.weight,
            mean=0.0,
            std=math.pow(self.d_model, -0.5),
        )
        nn.init.normal_(
            self.board_encoder.white_kingside_castling_rights_embeddings.weight,
            mean=0.0,
            std=math.pow(self.d_model, -0.5),
        )
        nn.init.normal_(
            self.board_encoder.white_queenside_castling_rights_embeddings.weight,
            mean=0.0,
            std=math.pow(self.d_model, -0.5),
        )
        nn.init.normal_(
            self.board_encoder.black_kingside_castling_rights_embeddings.weight,
            mean=0.0,
            std=math.pow(self.d_model, -0.5),
        )
        nn.init.normal_(
            self.board_encoder.black_queenside_castling_rights_embeddings.weight,
            mean=0.0,
            std=math.pow(self.d_model, -0.5),
        )
        nn.init.normal_(
            self.board_encoder.positional_embeddings.weight,
            mean=0.0,
            std=math.pow(self.d_model, -0.5),
        )
        nn.init.normal_(
            self.move_decoder.embeddings.weight,
            mean=0.0,
            std=math.pow(self.d_model, -0.5),
        )
        nn.init.normal_(
            self.move_decoder.positional_embeddings.weight,
            mean=0.0,
            std=math.pow(self.d_model, -0.5),
        )

        # Share weights between the embedding layer in the Decoder and
        # the logit layer
        self.move_decoder.fc.weight = self.move_decoder.embeddings.weight

    def forward(self, batch):
        """
        Forward prop.

        Args:

            batch (dict): A single batch, containing the following keys:

                turns (torch.LongTensor): The current turn (w/b), of
                size (N, 1).

                white_kingside_castling_rights (torch.LongTensor):
                Whether white can castle kingside, of size (N, 1).

                white_queenside_castling_rights (torch.LongTensor):
                Whether white can castle queenside, of size (N, 1).

                black_kingside_castling_rights (torch.LongTensor):
                Whether black can castle kingside, of size (N, 1).

                black_queenside_castling_rights (torch.LongTensor):
                Whether black can castle queenside, of size (N, 1).

                board_positions (torch.LongTensor): The current board
                positions, of size (N, 64).

                moves (torch.LongTensor): The move sequences, of size
                (N, n_moves).

                lengths (torch.LongTensor): The true lengths of the move
                sequences, not including <move> and <pad> tokens, of
                size (N, 1).

        Returns:

            torch.FloatTensor: The decoded next-move probabilities, of
            size (N, n_moves, vocab_size).
        """
        # Encoder
        boards = self.board_encoder(
            batch["turns"],
            batch["white_kingside_castling_rights"],
            batch["white_queenside_castling_rights"],
            batch["black_kingside_castling_rights"],
            batch["black_queenside_castling_rights"],
            batch["board_positions"],
        )  # (N, BOARD_STATUS_LENGTH, d_model)

        # Decoder
        moves = self.move_decoder(
            batch["moves"][:, :-1], batch["lengths"].squeeze(1), boards
        )  # (N, n_moves, move_vocab_size)
        # Note: We don't pass the last move as it has no next-move

        return moves


class ChessTransformerEncoder(nn.Module):
    """
    The Chess Transformer (Encoder only), for predicting the next move.
    """

    def __init__(
        self,
        CONFIG,
    ):
        """
        Init.

        Args:

            CONFIG (dict): The configuration, containing the following
            parameters for the model:

                VOCAB_SIZES (dict): The sizes of the vocabularies of the
                Encoder sequence components.

                D_MODEL (int): The size of vectors throughout the
                transformer model, i.e. input and output sizes for the
                Encoder.

                N_HEADS (int): The number of heads in the multi-head
                attention.

                D_QUERIES (int): The size of query vectors (and also the
                size of the key vectors) in the multi-head attention.

                D_VALUES (int): The size of value vectors in the
                multi-head attention.

                D_INNER (int): An intermediate size in the position-wise
                FC.

                N_LAYERS (int): The number of [multi-head attention +
                multi-head attention + position-wise FC] layers in the
                Encoder.

                DROPOUT (int): The dropout probability.
        """
        super(ChessTransformerEncoder, self).__init__()

        self.code = "E"

        self.vocab_sizes = CONFIG.VOCAB_SIZES
        self.d_model = CONFIG.D_MODEL
        self.n_heads = CONFIG.N_HEADS
        self.d_queries = CONFIG.D_QUERIES
        self.d_values = CONFIG.D_VALUES
        self.d_inner = CONFIG.D_INNER
        self.n_layers = CONFIG.N_LAYERS
        self.dropout = CONFIG.DROPOUT

        # Encoder
        self.board_encoder = BoardEncoder(
            vocab_sizes=self.vocab_sizes,
            d_model=self.d_model,
            n_heads=self.n_heads,
            d_queries=self.d_queries,
            d_values=self.d_values,
            d_inner=self.d_inner,
            n_layers=self.n_layers,
            dropout=self.dropout,
        )

        # Output linear layer that will compute logits for the
        # vocabulary
        self.fc = nn.Linear(self.d_model, self.vocab_sizes["moves"])

        # Initialize weights
        self.init_weights()

    def init_weights(self):
        """
        Initialize weights in the transformer model.
        """
        # Glorot uniform initialization with a gain of 1.
        for p in self.parameters():
            # Glorot initialization needs at least two dimensions on the
            # tensor
            if p.dim() > 1:
                nn.init.xavier_uniform_(p, gain=1.0)

        # For the embeddings, normal initialization with 0 mean and
        # 1/sqrt(d_model) S.D.
        nn.init.normal_(
            self.board_encoder.board_position_embeddings.weight,
            mean=0.0,
            std=math.pow(self.d_model, -0.5),
        )
        nn.init.normal_(
            self.board_encoder.turn_embeddings.weight,
            mean=0.0,
            std=math.pow(self.d_model, -0.5),
        )
        nn.init.normal_(
            self.board_encoder.white_kingside_castling_rights_embeddings.weight,
            mean=0.0,
            std=math.pow(self.d_model, -0.5),
        )
        nn.init.normal_(
            self.board_encoder.white_queenside_castling_rights_embeddings.weight,
            mean=0.0,
            std=math.pow(self.d_model, -0.5),
        )
        nn.init.normal_(
            self.board_encoder.black_kingside_castling_rights_embeddings.weight,
            mean=0.0,
            std=math.pow(self.d_model, -0.5),
        )
        nn.init.normal_(
            self.board_encoder.black_queenside_castling_rights_embeddings.weight,
            mean=0.0,
            std=math.pow(self.d_model, -0.5),
        )
        nn.init.normal_(
            self.board_encoder.positional_embeddings.weight,
            mean=0.0,
            std=math.pow(self.d_model, -0.5),
        )

    def forward(self, batch):
        """
        Forward prop.

        Args:

            batch (dict): A single batch, containing the following keys:

                turns (torch.LongTensor): The current turn (w/b), of
                size (N, 1).

                white_kingside_castling_rights (torch.LongTensor):
                Whether white can castle kingside, of size (N, 1).

                white_queenside_castling_rights (torch.LongTensor):
                Whether white can castle queenside, of size (N, 1).

                black_kingside_castling_rights (torch.LongTensor):
                Whether black can castle kingside, of size (N, 1).

                black_queenside_castling_rights (torch.LongTensor):
                Whether black can castle queenside, of size (N, 1).

                board_positions (torch.LongTensor): The current board
                positions, of size (N, 64).

        Returns:

            torch.FloatTensor: The next-move logits, of size (N, 1,
            vocab_size).
        """
        # Encoder
        boards = self.board_encoder(
            batch["turns"],
            batch["white_kingside_castling_rights"],
            batch["white_queenside_castling_rights"],
            batch["black_kingside_castling_rights"],
            batch["black_queenside_castling_rights"],
            batch["board_positions"],
        )  # (N, BOARD_STATUS_LENGTH, d_model)

        # Find logits over vocabulary at the "turn" token
        moves = self.fc(boards[:, :1, :])  # (N, 1, vocab_size)

        return moves


class ChessTransformerEncoderFT(nn.Module):
    """
    The Chess Transformer (Encoder only), for predicting the "From"
    square and the "To" square separately for the next move, instead of
    the move in UCI notation.
    """

    def __init__(
        self,
        CONFIG,
    ):
        """
        Init.

        Args:

            CONFIG (dict): The configuration, containing the following
            parameters for the model:

                VOCAB_SIZES (dict): The sizes of the vocabularies of the
                Encoder sequence components.

                D_MODEL (int): The size of vectors throughout the
                transformer model, i.e. input and output sizes for the
                Encoder.

                N_HEADS (int): The number of heads in the multi-head
                attention.

                D_QUERIES (int): The size of query vectors (and also the
                size of the key vectors) in the multi-head attention.

                D_VALUES (int): The size of value vectors in the
                multi-head attention.

                D_INNER (int): An intermediate size in the position-wise
                FC.

                N_LAYERS (int): The number of [multi-head attention +
                multi-head attention + position-wise FC] layers in the
                Encoder.

                DROPOUT (int): The dropout probability.
        """
        super(ChessTransformerEncoderFT, self).__init__()

        self.code = "EFT"

        self.vocab_sizes = CONFIG.VOCAB_SIZES
        self.d_model = CONFIG.D_MODEL
        self.n_heads = CONFIG.N_HEADS
        self.d_queries = CONFIG.D_QUERIES
        self.d_values = CONFIG.D_VALUES
        self.d_inner = CONFIG.D_INNER
        self.n_layers = CONFIG.N_LAYERS
        self.dropout = CONFIG.DROPOUT

        # Encoder
        self.board_encoder = BoardEncoder(
            vocab_sizes=self.vocab_sizes,
            d_model=self.d_model,
            n_heads=self.n_heads,
            d_queries=self.d_queries,
            d_values=self.d_values,
            d_inner=self.d_inner,
            n_layers=self.n_layers,
            dropout=self.dropout,
        )

        # Output linear layers - for the "From" square and "To" square
        self.from_squares = nn.Linear(self.d_model, 1)
        self.to_squares = nn.Linear(self.d_model, 1)

        # Initialize weights
        self.init_weights()

    def init_weights(self):
        """
        Initialize weights in the transformer model.
        """
        # Glorot uniform initialization with a gain of 1.
        for p in self.parameters():
            # Glorot initialization needs at least two dimensions on the
            # tensor
            if p.dim() > 1:
                nn.init.xavier_uniform_(p, gain=1.0)

        # For the embeddings, normal initialization with 0 mean and
        # 1/sqrt(d_model) S.D.
        nn.init.normal_(
            self.board_encoder.board_position_embeddings.weight,
            mean=0.0,
            std=math.pow(self.d_model, -0.5),
        )
        nn.init.normal_(
            self.board_encoder.turn_embeddings.weight,
            mean=0.0,
            std=math.pow(self.d_model, -0.5),
        )
        nn.init.normal_(
            self.board_encoder.white_kingside_castling_rights_embeddings.weight,
            mean=0.0,
            std=math.pow(self.d_model, -0.5),
        )
        nn.init.normal_(
            self.board_encoder.white_queenside_castling_rights_embeddings.weight,
            mean=0.0,
            std=math.pow(self.d_model, -0.5),
        )
        nn.init.normal_(
            self.board_encoder.black_kingside_castling_rights_embeddings.weight,
            mean=0.0,
            std=math.pow(self.d_model, -0.5),
        )
        nn.init.normal_(
            self.board_encoder.black_queenside_castling_rights_embeddings.weight,
            mean=0.0,
            std=math.pow(self.d_model, -0.5),
        )
        nn.init.normal_(
            self.board_encoder.positional_embeddings.weight,
            mean=0.0,
            std=math.pow(self.d_model, -0.5),
        )

    def forward(self, batch):
        """
        Forward prop.

        Args:

            batch (dict): A single batch, containing the following keys:

                turns (torch.LongTensor): The current turn (w/b), of
                size (N, 1).

                white_kingside_castling_rights (torch.LongTensor):
                Whether white can castle kingside, of size (N, 1).

                white_queenside_castling_rights (torch.LongTensor):
                Whether white can castle queenside, of size (N, 1).

                black_kingside_castling_rights (torch.LongTensor):
                Whether black can castle kingside, of size (N, 1).

                black_queenside_castling_rights (torch.LongTensor):
                Whether black can castle queenside, of size (N, 1).

                board_positions (torch.LongTensor): The current board
                positions, of size (N, 64).

        Returns:

            torch.FloatTensor: The "From" square logits, of size (N,
            64).

            torch.FloatTensor: The "To" square logits, of size (N, 64).
        """
        # Encoder
        boards = self.board_encoder(
            batch["turns"],
            batch["white_kingside_castling_rights"],
            batch["white_queenside_castling_rights"],
            batch["black_kingside_castling_rights"],
            batch["black_queenside_castling_rights"],
            batch["board_positions"],
        )  # (N, BOARD_STATUS_LENGTH, d_model)

        # Find logits over vocabulary at the "turn" token
        from_squares = (
            self.from_squares(boards[:, 5:, :]).squeeze(2).unsqueeze(1)
        )  # (N, 1, 64)
        to_squares = (
            self.to_squares(boards[:, 5:, :]).squeeze(2).unsqueeze(1)
        )  # (N, 1, 64)

        return from_squares, to_squares


if __name__ == "__main__":
    # Get configuration
    parser = argparse.ArgumentParser()
    parser.add_argument("config_name", type=str, help="Name of configuration file.")
    args = parser.parse_args()
    CONFIG = import_config(args.config_name)

    # Model
    model = CONFIG.MODEL(CONFIG).to(DEVICE)
    print(
        "There are %d learnable parameters in this model."
        % sum([p.numel() for p in model.parameters()])
    )
