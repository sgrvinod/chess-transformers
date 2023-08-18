import os
import json
import torch.optim
import torch.utils.data
from utils import *
from config import *
from tqdm import tqdm
from datasets import ChessDataset
from model import ChessTransformer
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader

scaler = GradScaler(enabled=USE_AMP)

def evaluate():

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

    # Dataloader
    test_loader = DataLoader(
        dataset=ChessDataset(
            data_folder=DATA_FOLDER,
            h5_file=H5_FILE,
            splits_file=SPLITS_FILE,
            split="test",
        ),
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=False,
        prefetch_factor=PREFETCH_FACTOR,
        shuffle=False,
    )

    # Compile model
    compiled_model = torch.compile(model, mode=COMPILE_MODE, dynamic=DYNAMIC_COMPILE)
    compiled_model.eval()  # eval mode disables dropout

    # Prohibit gradient computation explicitly
    with torch.no_grad():
        top1_accuracies = AverageMeter()  # top-1 accuracy of first move
        top3_accuracies = AverageMeter()  # top-3 accuracy of first move
        top5_accuracies = AverageMeter()  # top-5 accuracy of first move
        # Batches
        for i, (
            turns,
            white_kingside_castling_rights,
            white_queenside_castling_rights,
            black_kingside_castling_rights,
            black_queenside_castling_rights,
            can_claim_draw,
            board_positions,
            moves,
            lengths,
        ) in tqdm(enumerate(test_loader), desc="Evaluating", total=len(test_loader)):

            # Move to default device
            turns = turns.to(DEVICE)  # (N, 1)
            white_kingside_castling_rights = white_kingside_castling_rights.to(
                DEVICE
            )  # (N, 1)
            white_queenside_castling_rights = white_queenside_castling_rights.to(
                DEVICE
            )  # (N, 1)
            black_kingside_castling_rights = black_kingside_castling_rights.to(
                DEVICE
            )  # (N, 1)
            black_queenside_castling_rights = black_queenside_castling_rights.to(
                DEVICE
            )  # (N, 1)
            can_claim_draw = can_claim_draw.to(DEVICE)  # (N, 1)
            board_positions = board_positions.to(DEVICE)  # (N, 64)
            moves = moves.to(DEVICE)  # (N, max_move_sequence_length + 1)
            lengths = lengths.squeeze(1).to(DEVICE)  # (N)

            with torch.autocast(
                device_type=DEVICE.type, dtype=torch.float16, enabled=USE_AMP
            ):
                # Forward prop. Note: If "max_move_sequence_length" is 8
                # then the move sequence will be like "<move> a b c
                # <loss> <pad> <pad> <pad> <pad>". We do not pass the
                # last token to the Decoder as input (i.e. we left
                # shift)
                predicted_moves = compiled_model(
                    turns=turns,
                    white_kingside_castling_rights=white_kingside_castling_rights,
                    white_queenside_castling_rights=white_queenside_castling_rights,
                    black_kingside_castling_rights=black_kingside_castling_rights,
                    black_queenside_castling_rights=black_queenside_castling_rights,
                    can_claim_draw=can_claim_draw,
                    board_positions=board_positions,
                    moves=moves[:, :-1],
                    lengths=lengths,
                )  # (N, max_move_sequence_length, move_vocab_size)

            # Keep track of accuracy
            top1_accuracy, top3_accuracy, top5_accuracy = accuracy(
                predicted_moves=predicted_moves[:, 0, :],
                actual_moves=moves[:, 1],
                k=[1, 3, 5],
            )
            top1_accuracies.update(top1_accuracy, moves.shape[0])
            top3_accuracies.update(top3_accuracy, moves.shape[0])
            top5_accuracies.update(top5_accuracy, moves.shape[0])
    print("\n(FOR FIRST MOVE ONLY)")
    print("Top-1 accuracy: %.3f" % top1_accuracies.avg)
    print("Top-3 accuracy: %.3f" % top3_accuracies.avg)
    print("Top-5 accuracy: %.3f\n" % top5_accuracies.avg)


if __name__ == "__main__":
    evaluate()
