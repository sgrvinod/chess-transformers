import os
import json
import time
import torch.optim
import torch.utils.data
import torch.backends.cudnn as cudnn
from utils import *
from config import *
from tqdm import tqdm
from datasets import ChessDataset
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader
from model import ChessTransformer, LabelSmoothedCE

cudnn.benchmark = False

def main():
    """
    Training and validation.
    """
    global CHECKPOINT, STEP, START_EPOCH, epoch, epochs

    # Initialize data-loaders
    train_loader = DataLoader(
        dataset=ChessDataset(
            data_folder=DATA_FOLDER,
            h5_file=H5_FILE,
            splits_file=SPLITS_FILE,
            split="train",
        ),
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        prefetch_factor=PREFETCH_FACTOR,
        shuffle=True,
    )
    val_loader = DataLoader(
        dataset=ChessDataset(
            data_folder=DATA_FOLDER,
            h5_file=H5_FILE,
            splits_file=SPLITS_FILE,
            split="val",
        ),
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        prefetch_factor=PREFETCH_FACTOR,
        shuffle=False,
    )

    # Initialize model or load checkpoint
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
    optimizer = torch.optim.Adam(
        params=[p for p in model.parameters() if p.requires_grad],
        lr=LR,
        betas=BETAS,
        eps=EPSILON,
    )

    # Move to default device
    model = model.to(DEVICE)

    # Load checkpoint if available
    if TRAINING_CHECKPOINT is not None:
        TRAINING_CHECKPOINT = torch.load(TRAINING_CHECKPOINT)
        start_epoch = TRAINING_CHECKPOINT["epoch"] + 1
        model.load_state_dict(TRAINING_CHECKPOINT["model_state_dict"])
        optimizer.load_state_dict(TRAINING_CHECKPOINT["optimizer_state_dict"])
        print("\nLoaded checkpoint from epoch %d.\n" % start_epoch)

    # Loss function
    criterion = LabelSmoothedCE(eps=LABEL_SMOOTHING)
    criterion = criterion.to(DEVICE)

    # Compile model
    compiled_model = torch.compile(model, mode="default", dynamic=True)

    # AMP scaler
    scaler = GradScaler(enabled=USE_AMP)

    # Find total epochs to train
    epochs = (N_STEPS // (len(train_loader) // BATCHES_PER_STEP)) + 1

    # Epochs
    for epoch in range(start_epoch, epochs):
        # Step
        step = epoch * len(train_loader) // BATCHES_PER_STEP

        # One epoch's training
        train(
            train_loader=train_loader,
            model=compiled_model,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            epoch=epoch,
            step=step,
        )

        # One epoch's validation
        validate(
            val_loader=val_loader,
            model=compiled_model,
            criterion=criterion,
            epoch=epoch,
        )

        # Save checkpoint
        save_checkpoint(epoch, model, optimizer)


def train(train_loader, model, criterion, optimizer, scaler, epoch, step):
    """
    One epoch's training.

    Args:

        train_loader (torch.utils.data.DataLoader): loader for training
        data

        model (torch.nn.Module): model

        criterion (torch.nn.Module): label-smoothed cross-entropy loss

        optimizer (torch.optim.adam.Adam): optimizer

        scaler (torch.cuda.amp.GradScaler): AMP scaler

        epoch (int): epoch number

        step (int): step number
    """
    model.train()  # training mode enables dropout

    # Track some metrics
    data_time = AverageMeter()  # data loading time
    step_time = AverageMeter()  # forward prop. + back prop. time
    losses = AverageMeter()  # loss
    top1_accuracies = AverageMeter()  # top-1 accuracy of first move
    top3_accuracies = AverageMeter()  # top-3 accuracy of first move
    top5_accuracies = AverageMeter()  # top-5 accuracy of first move

    # Starting time
    start_data_time = time.time()
    start_step_time = time.time()

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
    ) in enumerate(train_loader):

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
        lengths = lengths.squeeze(1).to(DEVICE)  # (N, 1)

        # Time taken to load data
        data_time.update(time.time() - start_data_time)

        with torch.autocast(
            device_type=DEVICE.type, dtype=torch.float16, enabled=USE_AMP
        ):
            # Forward prop. Note: If "max_move_sequence_length" is 8
            # then the move sequence will be like "<move> a b c <loss>
            # <pad> <pad> <pad> <pad>" We do not pass the last token to
            # the Decoder as input (i.e. we left shift)
            predicted_moves = model(
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

            # Loss Note: If max_move_sequence_length is 8 then the move
            # sequence will be like "<move> a b c <loss> <pad> <pad>
            # <pad> <pad>" We do not pass the first token as an
            # "actual_move" as it is not one (i.e. we right shift)
            loss = criterion(
                moves=predicted_moves, actual_moves=moves[:, 1:], lengths=lengths
            )  # scalar
            loss = loss / BATCHES_PER_STEP

        # Backward prop.
        scaler.scale(loss).backward()

        # Keep track of losses
        losses.update(loss.item() * BATCHES_PER_STEP, lengths.sum().item())

        # Keep track of accuracy
        top1_accuracy, top3_accuracy, top5_accuracy = topk_accuracy(
            logits=predicted_moves[:, 0, :],
            targets=moves[:, 1],
            k=[1, 3, 5],
        )
        top1_accuracies.update(top1_accuracy, moves.shape[0])
        top3_accuracies.update(top3_accuracy, moves.shape[0])
        top5_accuracies.update(top5_accuracy, moves.shape[0])

        # Update model (i.e. perform a training step) only after
        # gradients are accumulated from batches_per_step batches
        if (i + 1) % BATCHES_PER_STEP == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            # This step is now complete
            step += 1

            # Update learning rate after each step
            change_lr(
                optimizer,
                new_lr=get_lr(step=step, d_model=D_MODEL, warmup_steps=WARMUP_STEPS),
            )

            # Time taken for this training step
            step_time.update(time.time() - start_step_time)

            # Print status
            if step % PRINT_FREQUENCY == 0:
                print(
                    "Epoch {0}/{1}---"
                    "Batch {2}/{3}---"
                    "Step {4}/{5}---"
                    "Data Time {data_time.val:.3f} ({data_time.avg:.3f})---"
                    "Step Time {step_time.val:.3f} ({step_time.avg:.3f})---"
                    "Loss {losses.val:.4f} ({losses.avg:.4f})---"
                    "Top-5 {top5s.val:.4f} ({top5s.avg:.4f})".format(
                        epoch + 1,
                        epochs,
                        i + 1,
                        len(train_loader),
                        step,
                        N_STEPS,
                        step_time=step_time,
                        data_time=data_time,
                        losses=losses,
                        top5s=top5_accuracies,
                    )
                )

            # Log to tensorboard
            WRITER.add_scalar(
                tag="train/loss", scalar_value=losses.val, global_step=step
            )
            WRITER.add_scalar(
                tag="train/lr",
                scalar_value=optimizer.param_groups[0]["lr"],
                global_step=step,
            )
            WRITER.add_scalar(
                tag="train/data_time", scalar_value=data_time.val, global_step=step
            )
            WRITER.add_scalar(
                tag="train/step_time", scalar_value=step_time.val, global_step=step
            )
            WRITER.add_scalar(
                tag="train/top1_accuracy",
                scalar_value=top1_accuracies.val,
                global_step=step,
            )
            WRITER.add_scalar(
                tag="train/top3_accuracy",
                scalar_value=top3_accuracies.val,
                global_step=step,
            )
            WRITER.add_scalar(
                tag="train/top5_accuracy",
                scalar_value=top5_accuracies.val,
                global_step=step,
            )

            # Reset step time
            start_step_time = time.time()

            # If this is the last one or two epochs, save checkpoints at
            # regular intervals for averaging
            if (
                epoch in [epochs - 1, epochs - 2] and step % 1500 == 0
            ):  # 'epoch' is 0-indexed
                save_checkpoint(
                    epoch, model, optimizer, prefix="step" + str(step) + "_"
                )

        # Reset data time
        start_data_time = time.time()


def validate(val_loader, model, criterion, epoch):
    """
    One epoch's validation.

    Args:

        val_loader (torch.utils.data.DataLoader): loader for validation
        data

        model (torch.nn.Module): model

        criterion (torch.nn.Module): label-smoothed cross-entropy loss

        epoch (int): epoch number
    """
    print("\n")
    model.eval()  # eval mode disables dropout

    # Prohibit gradient computation explicitly
    with torch.no_grad():
        losses = AverageMeter()
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
        ) in tqdm(enumerate(val_loader), desc="Validating", total=len(val_loader)):

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
                predicted_moves = model(
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

                # Loss Note: If max_move_sequence_length is 8 then the
                # move sequence will be like "<move> a b c <loss> <pad>
                # <pad> <pad> <pad>". We do not pass the first token as
                # an "actual_move" as it is not one (i.e. we right
                # shift)
                loss = criterion(
                    moves=predicted_moves, actual_moves=moves[:, 1:], lengths=lengths
                )  # scalar

            # Keep track of losses
            losses.update(loss.item(), lengths.sum().item())

            # Keep track of accuracy
            top1_accuracy, top3_accuracy, top5_accuracy = topk_accuracy(
                logits=predicted_moves[:, 0, :],
                targets=moves[:, 1],
                k=[1, 3, 5],
            )
            top1_accuracies.update(top1_accuracy, moves.shape[0])
            top3_accuracies.update(top3_accuracy, moves.shape[0])
            top5_accuracies.update(top5_accuracy, moves.shape[0])

        # Log to tensorboard
        WRITER.add_scalar(
            tag="val/loss", scalar_value=losses.avg, global_step=epoch + 1
        )
        WRITER.add_scalar(
            tag="val/top1_accuracy",
            scalar_value=top1_accuracies.avg,
            global_step=epoch + 1,
        )
        WRITER.add_scalar(
            tag="val/top3_accuracy",
            scalar_value=top3_accuracies.avg,
            global_step=epoch + 1,
        )
        WRITER.add_scalar(
            tag="val/top5_accuracy",
            scalar_value=top5_accuracies.avg,
            global_step=epoch + 1,
        )

        print("\nValidation loss: %.3f" % losses.avg)
        print("Validation top-1 accuracy: %.3f" % top1_accuracies.avg)
        print("Validation top-3 accuracy: %.3f" % top3_accuracies.avg)
        print("Validation top-5 accuracy: %.3f\n" % top5_accuracies.avg)


if __name__ == "__main__":
    main()
