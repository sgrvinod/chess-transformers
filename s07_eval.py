import os
import json
import argparse
import torch.optim
import torch.utils.data
from utils import *
from tqdm import tqdm
from importlib import import_module
from torch.utils.data import DataLoader

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)  # CPU isn't really practical here


def evaluate(CONFIG):
    """
    Evaluation the final checkpoint against the test data.

    Args:

        CONFIG (dict): The configuration. See ./configs.
    """

    # Initialize model and load checkpoint
    model = CONFIG.MODEL(CONFIG)
    model = model.to(DEVICE)
    checkpoint = torch.load(
        os.path.join(CONFIG.CHECKPOINT_FOLDER, CONFIG.FINAL_CHECKPOINT)
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    print("\nLoaded checkpoint.\n")

    # Dataloader
    test_loader = DataLoader(
        dataset=CONFIG.DATASET(
            data_folder=CONFIG.DATA_FOLDER,
            h5_file=CONFIG.H5_FILE,
            splits_file=CONFIG.SPLITS_FILE,
            split="test",
        ),
        batch_size=CONFIG.BATCH_SIZE,
        num_workers=CONFIG.NUM_WORKERS,
        pin_memory=CONFIG.PIN_MEMORY,
        prefetch_factor=CONFIG.PREFETCH_FACTOR,
        shuffle=False,
    )

    # Compile model
    compiled_model = torch.compile(
        model,
        mode=CONFIG.COMPILATION_MODE,
        dynamic=CONFIG.DYNAMIC_COMPILATION,
        disable=CONFIG.DISABLE_COMPILATION,
    )
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
                device_type=DEVICE.type, dtype=torch.float16, enabled=CONFIG.USE_AMP
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
            top1_accuracy, top3_accuracy, top5_accuracy = topk_accuracy(
                logits=predicted_moves[:, 0, :],
                targets=moves[:, 1],
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
    # Get configuration
    parser = argparse.ArgumentParser()
    parser.add_argument("config_name", type=str, help="Name of configuration file.")
    args = parser.parse_args()
    CONFIG = import_module("configs.{}".format(args.config_name))

    # Evaluate model
    evaluate(CONFIG)
