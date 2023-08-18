import os
import json
import chess
import torch.optim
import torch.utils.data
from utils import *
from config import *
from tqdm import tqdm
from datasets import ChessDataset
from model import ChessTransformer
from s03_encode_data import encode
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader
from s01_prepare_data import get_board_status

scaler = GradScaler(enabled=USE_AMP)

# Initialize model and load checkpoint
vocabulary = json.load(open(os.path.join(DATA_FOLDER, VOCAB_FILE), "r"))
vocab_sizes = dict()
for k in vocabulary:
    vocab_sizes[k] = len(vocabulary[k])
model = ChessTransformer(
    vocab_sizes=vocab_sizes,
    max_move_sequence_length=MAX_MOVE_SEQUENCE_LENGTH,
    d_model=D_MODEL,
    n_heads=N_HEADS,
    d_queries=D_QUERIES,
    d_values=D_VALUES,
    d_inner=D_INNER,
    n_layers=N_LAYERS,
    dropout=DROPOUT,
)
model = model.to(DEVICE)
checkpoint = torch.load(os.path.join(CHECKPOINT_FOLDER, FINAL_CHECKPOINT))
model.load_state_dict(checkpoint["model_state_dict"])
print("\nLoaded checkpoint.\n")

# Compile model
compiled_model = torch.compile(model, mode=COMPILE_MODE, dynamic=DYNAMIC_COMPILE)
compiled_model.eval()
# eval mode disables dropout

# "Move index to move" mapping
reverse_move_vocabulary = {v: k for k, v in vocabulary["output_sequence"].items()}

def make_model_move(board):
    board_status = get_board_status(board)
    encoded_board_status = dict()
    for status in board_status:
        encoded_board_status[status] = torch.IntTensor(
            [encode(board_status[status], vocabulary=vocabulary[status])]
        ).to(DEVICE)
        if encoded_board_status[status].dim() == 1:
            encoded_board_status[status] = encoded_board_status[status].unsqueeze(0)
    moves = (
        torch.LongTensor([vocabulary["output_sequence"]["<move>"]]).unsqueeze(0).to(DEVICE)
    )
    lengths = torch.LongTensor([1]).unsqueeze(0).to(DEVICE)
    with torch.autocast(device_type=DEVICE.type, dtype=torch.float16, enabled=USE_AMP):
        predicted_moves = compiled_model(
            turns=encoded_board_status["turn"],
            white_kingside_castling_rights=encoded_board_status[
                "white_kingside_castling_rights"
            ],
            white_queenside_castling_rights=encoded_board_status[
                "white_queenside_castling_rights"
            ],
            black_kingside_castling_rights=encoded_board_status[
                "black_kingside_castling_rights"
            ],
            black_queenside_castling_rights=encoded_board_status[
                "black_queenside_castling_rights"
            ],
            can_claim_draw=encoded_board_status["can_claim_draw"],
            board_positions=encoded_board_status["board_position"],
            moves=moves,
            lengths=lengths,
        )  # (N, max_move_sequence_length, move_vocab_size)
    predicted_moves = predicted_moves[:, 0, :].squeeze()
    legal_moves = [str(m) for m in list(board.legal_moves)]
    _, model_move_indices = predicted_moves.topk(k=predicted_moves.shape[0])
    for model_move_index in model_move_indices.tolist():
        model_move = reverse_move_vocabulary[model_move_index]
        if model_move == "<loss>":
            print("I lost! :(")
            return board
        if model_move == "<draw>":
            print("Let's draw?")
            return board
        if model_move in legal_moves:
            board.push_uci(model_move)
            display(board)
            return board

def make_human_move(board):
    legal_moves = [m.uci() for m in board.legal_moves]
    while True:
        human_move = input("What move would you like to play? (Enter in UCI notation.)")
        if human_move in legal_moves:
            board.push_uci(human_move)
            display(board)
            return board
        print("\n")
        print("'%s' isn't a move you can make!" % human_move)
    