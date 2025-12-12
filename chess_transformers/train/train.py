"""Training script for chess transformer models.

This module provides a complete PyTorch training pipeline for training
chess transformer models on LMDB datasets. It supports mixed precision
training, gradient accumulation, learning rate scheduling with warmup,
and checkpoint saving/resumption.

Key Features:
    - train_model: Main entry point for training with full configuration.
    - train_one_epoch: Single training epoch with progress logging.
    - validate: Validation loop returning loss and accuracy metrics.
    - Checkpoint saving and resumption for fault-tolerant training.
    - Mixed precision training via torch.amp for faster GPU training.
    - Cosine annealing learning rate schedule with linear warmup.

Notes:
    The training script expects data in LMDB format created by
    `chess_transformers.data.process`. Use the config system to specify
    model architecture, data loading, and training hyperparameters.

Example:
    >>> from chess_transformers.train.train import train_model
    >>> from chess_transformers.utilities.configs import import_config
    >>> config = import_config("vole")
    >>> train_model(config)
"""

import os

import torch
import torch.nn as nn
from pathlib import Path
from tqdm import tqdm
from typing import Optional, Tuple, Union

from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from chess_transformers.models.configs.base import ModelConfig
from chess_transformers.models.criteria import LegalMoveSmoothing
from chess_transformers.utilities.loggers import setup_logger

# Logger
logger = setup_logger(__file__)


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: LambdaLR,
    scaler: GradScaler,
    epoch: int,
    global_step: int,
    best_val_loss: float,
) -> None:
    """Save a training checkpoint.

    Saves model weights, optimizer state, scheduler state, and training
    progress to a checkpoint file. Compatible with both compiled and
    non-compiled models (PyTorch automatically unwraps compiled models).

    Args:
        path: Path to save the checkpoint file.
        model: The model to save.
        optimizer: The optimizer state to save.
        scheduler: The learning rate scheduler state to save.
        scaler: The gradient scaler state (for AMP).
        epoch: Current epoch number.
        global_step: Current global training step.
        best_val_loss: Best validation loss seen so far.
    """
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "best_val_loss": best_val_loss,
    }

    torch.save(checkpoint, path)
    logger.info(f"Checkpoint saved to {path}")


def load_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: LambdaLR,
    scaler: GradScaler,
    device: torch.device,
) -> Tuple[int, int, float]:
    """Load a training checkpoint.

    Args:
        path: Path to the checkpoint file.
        model: The model to load weights into.
        optimizer: The optimizer to load state into.
        scheduler: The scheduler to load state into.
        scaler: The gradient scaler to load state into.
        device: Device to map tensors to.

    Returns:
        Tuple of (epoch, global_step, best_val_loss).
    """
    checkpoint = torch.load(path, map_location=device, weights_only=False)  # noqa: S614

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    if "scaler_state_dict" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler_state_dict"])

    epoch = checkpoint["epoch"]
    global_step = checkpoint["global_step"]
    best_val_loss = checkpoint["best_val_loss"]

    logger.info(f"Loaded checkpoint from {path} (epoch {epoch}, step {global_step})")
    return epoch, global_step, best_val_loss


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: LambdaLR,
    scaler: GradScaler,
    device: torch.device,
    epoch: int,
    global_step: int,
    gradient_accumulation_steps: int = 1,
    max_grad_norm: float = 1.0,
    use_amp: bool = True,
    use_legal_masks: bool = False,
    log_every_n_steps: int = 100,
) -> Tuple[float, float, float, int]:
    """Train the model for one epoch.

    Args:
        model: The model to train.
        dataloader: DataLoader for training data.
        loss_fn: Loss function to use.
        optimizer: Optimizer for parameter updates.
        scheduler: Learning rate scheduler.
        scaler: Gradient scaler for AMP.
        device: Device to run training on.
        epoch: Current epoch number (for logging).
        global_step: Current global step count.
        gradient_accumulation_steps: Number of steps to accumulate gradients.
        max_grad_norm: Maximum gradient norm for clipping.
        use_amp: Whether to use automatic mixed precision.
        use_legal_masks: Whether to pass legal masks to the loss function.
        log_every_n_steps: Log training metrics every N steps.

    Returns:
        Tuple of (average_loss, from_accuracy, to_accuracy, updated_global_step).
    """
    model.train()
    optimizer.zero_grad()  # Ensure zero gradients at start of epoch
    total_loss = 0.0
    total_from_correct = 0
    total_to_correct = 0
    total_samples = 0
    accumulated_loss = 0.0

    progress_bar = tqdm(
        dataloader,
        desc=f"Epoch {epoch} [Train]",
        leave=True,
    )

    for step, batch in enumerate(progress_bar):
        # Move batch to device
        batch = {k: v.to(device) for k, v in batch.items()}

        # Forward pass with AMP
        with autocast(device_type=device.type, enabled=use_amp):
            from_logits, to_logits = model(batch)

            # Prepare targets (squeeze extra dimension)
            from_targets = batch["from_squares"].squeeze(-1)
            to_targets = batch["to_squares"].squeeze(-1)

            # Compute loss
            if use_legal_masks:
                loss = loss_fn(
                    from_logits,
                    to_logits,
                    from_targets,
                    to_targets,
                    batch["legal_from_mask"],
                    batch["legal_to_mask"],
                )
            else:
                loss = loss_fn(from_logits, to_logits, from_targets, to_targets)

            # Scale loss for gradient accumulation
            loss = loss / gradient_accumulation_steps

        # Backward pass (scaler handles enabled/disabled internally)
        scaler.scale(loss).backward()

        accumulated_loss += loss.item()

        # Update weights after accumulation
        if (step + 1) % gradient_accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scale_before = scaler.get_scale()
            scaler.update()

            # Only step scheduler if optimizer actually stepped (no inf/NaN grads)
            if scaler.get_scale() >= scale_before:
                scheduler.step()
            optimizer.zero_grad()
            global_step += 1

            # Log periodically
            if global_step % log_every_n_steps == 0:
                current_lr = scheduler.get_last_lr()[0]
                logger.info(
                    f"Epoch {epoch} | Step {global_step} | "
                    f"Loss: {accumulated_loss:.4f} | LR: {current_lr:.2e}"
                )
            accumulated_loss = 0.0

        # Track metrics (use unscaled loss)
        # Note: We track total summed loss and divide by total samples later for accuracy
        batch_size = from_targets.size(0)
        total_loss += loss.item() * gradient_accumulation_steps * batch_size
        total_samples += batch_size

        # Compute accuracy
        from_preds = from_logits.squeeze(1).argmax(dim=-1)
        to_preds = to_logits.squeeze(1).argmax(dim=-1)
        total_from_correct += (from_preds == from_targets).sum().item()
        total_to_correct += (to_preds == to_targets).sum().item()

        # Update progress bar
        progress_bar.set_postfix(
            loss=f"{total_loss / total_samples:.4f}",
            from_acc=f"{100 * total_from_correct / total_samples:.1f}%",
            to_acc=f"{100 * total_to_correct / total_samples:.1f}%",
        )

    avg_loss = total_loss / total_samples
    from_accuracy = total_from_correct / total_samples
    to_accuracy = total_to_correct / total_samples

    return avg_loss, from_accuracy, to_accuracy, global_step


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    epoch: int,
    use_amp: bool = True,
    use_legal_masks: bool = False,
) -> Tuple[float, float, float]:
    """Validate the model on the validation set.

    Args:
        model: The model to validate.
        dataloader: DataLoader for validation data.
        loss_fn: Loss function to use.
        device: Device to run validation on.
        epoch: Current epoch number (for logging).
        use_amp: Whether to use automatic mixed precision.
        use_legal_masks: Whether to pass legal masks to the loss function.

    Returns:
        Tuple of (average_loss, from_accuracy, to_accuracy).
    """
    model.eval()
    total_loss = 0.0
    total_from_correct = 0
    total_to_correct = 0
    total_samples = 0

    progress_bar = tqdm(
        dataloader,
        desc=f"Epoch {epoch} [Val]",
        leave=True,
    )

    with torch.no_grad():
        for batch in progress_bar:
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}

            # Forward pass with AMP
            with autocast(device_type=device.type, enabled=use_amp):
                from_logits, to_logits = model(batch)

                # Prepare targets
                from_targets = batch["from_squares"].squeeze(-1)
                to_targets = batch["to_squares"].squeeze(-1)

                # Compute loss
                if use_legal_masks:
                    loss = loss_fn(
                        from_logits,
                        to_logits,
                        from_targets,
                        to_targets,
                        batch["legal_from_mask"],
                        batch["legal_to_mask"],
                    )
                else:
                    loss = loss_fn(from_logits, to_logits, from_targets, to_targets)

            batch_size = from_targets.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

            # Compute accuracy
            from_preds = from_logits.squeeze(1).argmax(dim=-1)
            to_preds = to_logits.squeeze(1).argmax(dim=-1)
            total_from_correct += (from_preds == from_targets).sum().item()
            total_to_correct += (to_preds == to_targets).sum().item()

            # Update progress bar
            progress_bar.set_postfix(
                loss=f"{total_loss / total_samples:.4f}",
                from_acc=f"{100 * total_from_correct / total_samples:.1f}%",
                to_acc=f"{100 * total_to_correct / total_samples:.1f}%",
            )

    avg_loss = total_loss / total_samples
    from_accuracy = total_from_correct / total_samples
    to_accuracy = total_to_correct / total_samples

    return avg_loss, from_accuracy, to_accuracy


def train_model(
    config: ModelConfig,
    checkpoint_folder: Optional[Union[str, Path]] = None,
) -> None:
    """Train a chess transformer model.

    Main entry point for training. Creates model, datasets, dataloaders,
    optimizer, scheduler, and runs the training loop with validation.

    Args:
        config: Model configuration containing architecture and training
            parameters. Must have train_config and dataloader_config set.
        checkpoint_folder: Path to save checkpoints. If None, uses
            train_config.checkpoint_dir or CT_CHECKPOINTS_FOLDER env var.

    Raises:
        ValueError: If required config attributes are missing.
        RuntimeError: If CUDA is not available.
    """
    # Validate config
    if config.train_config is None:
        logger.critical("train_config is required in config")
        raise ValueError("config.train_config is required for training")
    if config.dataloader_config is None:
        logger.critical("dataloader_config is required in config")
        raise ValueError("config.dataloader_config is required for training")
    if config.dataloader_config.lmdb_filepath is None:
        logger.critical("lmdb_filepath is required in dataloader_config")
        raise ValueError("dataloader_config.lmdb_filepath is required for training")

    train_config = config.train_config
    dataloader_config = config.dataloader_config

    # Set up device - CUDA is required for training
    if not torch.cuda.is_available():
        logger.critical("CUDA is not available. Training requires a GPU.")
        raise RuntimeError("CUDA is not available. Training requires a GPU.")
    device = torch.device("cuda")
    logger.info(f"Using device: {device}")

    # Enable TF32 for optimal performance on Ampere+ GPUs (A100, H100, H200, RTX 30xx+)
    torch.set_float32_matmul_precision("high")
    logger.info("Set float32 matmul precision to 'high' (TF32 enabled)")

    # Expand environment variables in lmdb_filepath
    lmdb_filepath = dataloader_config.lmdb_filepath
    lmdb_path = Path(os.path.expandvars(lmdb_filepath))
    if not lmdb_path.exists():
        logger.critical(f"LMDB file not found: {lmdb_path}")
        raise FileNotFoundError(f"LMDB file not found: {lmdb_path}")
    logger.info(f"Loading data from {lmdb_path}")

    # Set up checkpoint folder
    # Priority: CLI arg > config.train_config.checkpoint_dir > CT_CHECKPOINTS_FOLDER env var
    if checkpoint_folder is None:
        if train_config.checkpoint_dir is not None:
            checkpoint_folder = Path(train_config.checkpoint_dir)
        else:
            checkpoint_root = os.environ.get("CT_CHECKPOINTS_FOLDER")
            if not checkpoint_root:
                logger.critical(
                    "No checkpoint folder specified. Provide via --checkpoint-folder, "
                    "train_config.checkpoint_dir, or CT_CHECKPOINTS_FOLDER env var."
                )
                raise OSError(
                    "No checkpoint folder specified. Set train_config.checkpoint_dir "
                    "or CT_CHECKPOINTS_FOLDER environment variable."
                )
            # Use name and version for subfolder when using env var
            checkpoint_folder = (
                Path(checkpoint_root) / f"{config.name}_{config.version}"
            )
    else:
        checkpoint_folder = Path(checkpoint_folder)
    checkpoint_folder.mkdir(parents=True, exist_ok=True)
    logger.info(f"Checkpoints will be saved to {checkpoint_folder}")

    # Create datasets using the dataset class from config
    dataset_cls = dataloader_config.dataset
    train_dataset = dataset_cls(lmdb_path=lmdb_path, split="train")
    val_dataset = dataset_cls(lmdb_path=lmdb_path, split="val")
    logger.info(f"Using dataset class: {dataset_cls.__name__}")
    logger.info(f"Train dataset: {len(train_dataset):,} samples")
    logger.info(f"Val dataset: {len(val_dataset):,} samples")

    # Check for legal masks (needed for LegalMoveSmoothing)
    use_legal_masks = train_dataset.has_masks
    is_legal_smoothing = issubclass(train_config.loss_fn, LegalMoveSmoothing)
    if is_legal_smoothing and not use_legal_masks:
        logger.critical(
            "LegalMoveSmoothing loss requires legal masks in dataset. "
            "Either use a different loss function or recreate dataset with legal_mask_mode."
        )
        raise ValueError("LegalMoveSmoothing requires dataset with legal masks")

    # Close datasets in main process before creating DataLoaders
    train_dataset.close()
    val_dataset.close()

    # Create dataloaders
    train_dataloader_kwargs = dataloader_config.to_dataloader_kwargs()
    train_dataloader_kwargs["shuffle"] = True
    train_loader = DataLoader(train_dataset, **train_dataloader_kwargs)
    val_dataloader_kwargs = dataloader_config.to_dataloader_kwargs()
    val_dataloader_kwargs["shuffle"] = False
    val_dataloader_kwargs["drop_last"] = False
    val_loader = DataLoader(val_dataset, **val_dataloader_kwargs)

    # Create model
    model = config.model_type(config).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(
        f"Model {config.name} v{config.version} created with {n_params:,} parameters"
    )

    # Apply torch.compile() with config settings (disable param skips compilation)
    model = torch.compile(
        model,
        mode=config.compilation_mode,
        dynamic=config.dynamic_compilation,
        fullgraph=config.fullgraph_compilation,
        disable=config.disable_compilation,
    )
    if config.disable_compilation:
        logger.info("Model compilation disabled")
    else:
        logger.info(
            f"Compiling model with mode='{config.compilation_mode}', "
            f"dynamic={config.dynamic_compilation}, "
            f"fullgraph={config.fullgraph_compilation}"
        )

    # Create optimizer using the class and args from config
    optimizer = train_config.optimizer(
        model.parameters(),
        **train_config.optimizer_args,
    )
    logger.info(
        f"Using optimizer: {train_config.optimizer.__name__} "
        f"with args: {train_config.optimizer_args}"
    )

    # Compute total steps and logging frequency
    steps_per_epoch = len(train_loader) // train_config.gradient_accumulation_steps
    total_steps = steps_per_epoch * train_config.epochs
    log_every_n_steps = max(1, int(steps_per_epoch * train_config.epochs_per_log))
    logger.info(
        f"Training for {train_config.epochs} epochs, "
        f"{steps_per_epoch} steps/epoch, {total_steps} total steps"
    )
    logger.info(
        f"Logging every {log_every_n_steps} steps "
        f"(epochs_per_log={train_config.epochs_per_log})"
    )

    # Create LR scheduler - always use LambdaLR with the lr_schedule factory from config
    lr_lambda = train_config.lr_schedule(
        total_steps=total_steps,
        **train_config.lr_schedule_args,
    )
    scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
    logger.info(
        f"Using LR schedule: {train_config.lr_schedule.__name__} "
        f"(total_steps={total_steps}, {train_config.lr_schedule_args})"
    )

    # Create loss function by instantiating the class with args from config
    loss_fn = train_config.loss_fn(**train_config.loss_fn_args)
    logger.info(
        f"Using loss function: {train_config.loss_fn.__name__} "
        f"with args: {train_config.loss_fn_args}"
    )

    # Create gradient scaler for AMP (enabled param controls whether scaling is applied)
    scaler = GradScaler(enabled=train_config.use_amp)
    if train_config.use_amp:
        logger.info("Using automatic mixed precision (AMP)")

    # Initialize training state
    start_epoch = 0
    global_step = 0
    best_val_loss = float("inf")

    # Auto-resume from latest.pt if it exists
    latest_checkpoint = checkpoint_folder / "latest.pt"
    if latest_checkpoint.exists():
        logger.info(f"Found existing checkpoint at {latest_checkpoint}, resuming...")
        start_epoch, global_step, best_val_loss = load_checkpoint(
            latest_checkpoint, model, optimizer, scheduler, scaler, device
        )
        start_epoch += 1  # Start from next epoch
        logger.info(f"Resuming from epoch {start_epoch}, step {global_step}")

        # Check if training is already complete
        if start_epoch >= train_config.epochs:
            logger.warning(
                f"Checkpoint is at epoch {start_epoch - 1} (0-indexed), "
                f"but config only has {train_config.epochs} epochs. "
                "No training will be done. Increase epochs in config to continue."
            )
            train_dataset.close()
            val_dataset.close()
            return
    else:
        logger.info("No existing checkpoint found, starting fresh")

    # Training loop
    try:
        for epoch in range(start_epoch, train_config.epochs):
            logger.info(f"Starting epoch {epoch + 1}/{train_config.epochs}")

            # Train
            train_loss, train_from_acc, train_to_acc, global_step = train_one_epoch(
                model=model,
                dataloader=train_loader,
                loss_fn=loss_fn,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                device=device,
                epoch=epoch + 1,
                global_step=global_step,
                gradient_accumulation_steps=train_config.gradient_accumulation_steps,
                max_grad_norm=train_config.max_grad_norm,
                use_amp=train_config.use_amp,
                use_legal_masks=use_legal_masks and is_legal_smoothing,
                log_every_n_steps=log_every_n_steps,
            )
            logger.info(
                f"Epoch {epoch + 1} Train | Loss: {train_loss:.4f} | "
                f"From Acc: {100 * train_from_acc:.2f}% | "
                f"To Acc: {100 * train_to_acc:.2f}%"
            )

            # Validate every epoch
            val_loss, val_from_acc, val_to_acc = validate(
                model=model,
                dataloader=val_loader,
                loss_fn=loss_fn,
                device=device,
                epoch=epoch + 1,
                use_amp=train_config.use_amp,
                use_legal_masks=use_legal_masks and is_legal_smoothing,
            )
            logger.info(
                f"Epoch {epoch + 1} Val | Loss: {val_loss:.4f} | "
                f"From Acc: {100 * val_from_acc:.2f}% | "
                f"To Acc: {100 * val_to_acc:.2f}%"
            )

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_path = checkpoint_folder / "best.pt"
                save_checkpoint(
                    best_path,
                    model,
                    optimizer,
                    scheduler,
                    scaler,
                    epoch,
                    global_step,
                    best_val_loss,
                )
                logger.info(f"New best model saved (val_loss={val_loss:.4f})")

            # Save latest checkpoint (overwritten every epoch for resumption)
            latest_path = checkpoint_folder / "latest.pt"
            save_checkpoint(
                latest_path,
                model,
                optimizer,
                scheduler,
                scaler,
                epoch,
                global_step,
                best_val_loss,
            )

        logger.info("Training complete!")

    finally:
        # Clean up datasets even if an exception occurs
        train_dataset.close()
        val_dataset.close()


if __name__ == "__main__":
    # Command-line interface for training.
    import argparse

    from chess_transformers.utilities.configs import import_config

    parser = argparse.ArgumentParser(description="Train a chess transformer model")
    parser.add_argument("config_name", type=str, help="Name of configuration file")
    parser.add_argument(
        "--checkpoint-folder",
        type=str,
        default=None,
        help="Override checkpoint folder (default: from config or env var)",
    )
    args = parser.parse_args()

    # Import config and train
    config = import_config(args.config_name)
    train_model(config=config, checkpoint_folder=args.checkpoint_folder)
