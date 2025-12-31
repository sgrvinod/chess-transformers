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
    - Rich terminal display with progress bars and sparklines.

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
from aim import Run as AimRun
from datetime import datetime
from aim import Repo as AimRepo
from aim import Text as AimText
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import LambdaLR
from typing import Any, Dict, Optional, Tuple, Union

from chess_transformers.utilities.loggers import setup_logger
from chess_transformers.models.configs.base import ModelConfig
from chess_transformers.models.criteria import LegalMoveSmoothing
from chess_transformers.train.logs import (
    MetricsTracker,
    Timer,
    TrainingDisplay,
    compute_topk_accuracy,
)

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
    best_val_top1_acc: float,
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
        best_val_top1_acc: Best validation top-1 accuracy seen so far.
    """
    # Create checkpoint folder if it doesn't exist
    path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "best_val_top1_acc": best_val_top1_acc,
    }

    torch.save(checkpoint, path)


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
        Tuple of (epoch, global_step, best_val_top1_acc).
    """
    checkpoint = torch.load(path, map_location=device, weights_only=False)  # noqa: S614

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    if "scaler_state_dict" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler_state_dict"])

    epoch = checkpoint["epoch"]
    global_step = checkpoint["global_step"]
    # Support both old (best_val_loss) and new (best_val_top1_acc) checkpoint formats
    best_val_top1_acc = checkpoint.get("best_val_top1_acc", 0.0)

    logger.info(f"Loaded checkpoint from {path} (epoch {epoch}, step {global_step})")
    return epoch, global_step, best_val_top1_acc


def get_hparam_diff(
    repo_path: str,
    experiment: str,
    current_hparams: Dict[str, Any],
    max_changes: int = 20,
) -> str:
    """Generate a description by comparing current hparams to the previous run.

    Queries the Aim repository for the most recent run in the same experiment
    and computes the difference between its hyperparameters and the current
    ones. This provides an automatic audit trail of what changed.

    Args:
        repo_path: Path to the Aim repository.
        experiment: Name of the experiment to query runs from.
        current_hparams: Dictionary of current hyperparameters to compare.
        max_changes: Maximum number of changes to include in the description.

    Returns:
        A description string summarizing the changes:
        - "First run in this experiment" if no previous runs exist.
        - "No changes from previous run" if hparams are identical.
        - "Changed: key1: old → new, key2: old → new, ..." otherwise.
    """
    # Check if the Aim repository exists
    aim_dir = Path(repo_path) / ".aim"
    if not aim_dir.exists():
        return "First run in this experiment"

    try:
        repo = AimRepo(repo_path)
        query = f"run.experiment == '{experiment}'"
        run_collections = list(repo.query_runs(query).iter_runs())

        if not run_collections:
            return "First run in this experiment"

        # Sort runs by creation time (descending)
        run_collections.sort(key=lambda rc: rc.run.created_at, reverse=True)

        # Iterate through runs to find the first one with non-empty hparams
        prev_hparams = {}
        for collection in run_collections:
            run_hparams = dict(collection.run.get("hparams", {}))
            if run_hparams:
                prev_hparams = run_hparams
                break
        
        # If no valid previous run found, treat as first run
        if not prev_hparams:
            return "First run in this experiment"

        # Find differences
        changes = []
        all_keys = set(current_hparams.keys()) | set(prev_hparams.keys())
        for key in sorted(all_keys):
            old_val = prev_hparams.get(key)
            new_val = current_hparams.get(key)
            if old_val != new_val:
                # Format values for readability
                old_str = _format_hparam_value(old_val)
                new_str = _format_hparam_value(new_val)
                changes.append(f"{key}: {old_str} → {new_str}")

        if not changes:
            return "No changes from previous run"

        # Truncate if too many changes
        if len(changes) > max_changes:
            shown = changes[:max_changes]
            remaining = len(changes) - max_changes
            return "Changed: " + ", ".join(shown) + f" (+{remaining} more)"

        return "Changed: " + ", ".join(changes)

    except Exception as e:
        logger.warning(f"Could not compute hparam diff: {e}")
        return "Could not determine changes from previous run"


def _format_hparam_value(value: Any) -> str:
    """Format a hyperparameter value for display in diff descriptions.

    Args:
        value: The hyperparameter value to format.

    Returns:
        A string representation suitable for display.
    """
    if value is None:
        return "None"
    if isinstance(value, float):
        # Format floats nicely (scientific notation for small values)
        if abs(value) < 0.001 and value != 0:
            return f"{value:.2e}"
        return f"{value:.4g}"
    if isinstance(value, dict):
        # Abbreviated dict representation
        if len(value) > 3:
            keys = list(value.keys())[:3]
            return "{" + ", ".join(f"{k}: ..." for k in keys) + ", ...}"
        return str(value)
    if isinstance(value, list) and len(value) > 5:
        return f"[{len(value)} items]"
    return str(value)


def _get_data_size_bucket(n_rows: Optional[int]) -> str:
    """Get the data size bucket tag based on number of rows.

    Args:
        n_rows: Number of samples in the dataset, or None if unknown.

    Returns:
        A bucket tag string: "data_tiny", "data_small", "data_medium",
        "data_large", "data_xlarge", or "data_unknown".
    """
    if n_rows is None:
        return "data_unknown"
    if n_rows < 100_000:
        return "data_tiny"
    if n_rows < 1_000_000:
        return "data_small"
    if n_rows < 10_000_000:
        return "data_medium"
    if n_rows < 100_000_000:
        return "data_large"
    return "data_xlarge"


def _get_model_size_bucket(n_params: int) -> str:
    """Get the model size bucket tag based on number of parameters.

    Args:
        n_params: Number of trainable parameters in the model.

    Returns:
        A bucket tag string: "model_tiny", "model_small", "model_medium",
        "model_large", or "model_xlarge".
    """
    if n_params < 1_000_000:
        return "model_tiny"
    if n_params < 10_000_000:
        return "model_small"
    if n_params < 100_000_000:
        return "model_medium"
    if n_params < 1_000_000_000:
        return "model_large"
    return "model_xlarge"


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
    display: TrainingDisplay,
    aim_run: AimRun,
    effective_batch_size: int,
    gradient_accumulation_steps: int = 1,
    max_grad_norm: float = 1.0,
    use_amp: bool = True,
    use_legal_masks: bool = False,
) -> Tuple[Dict[str, float], int]:
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
        display: TrainingDisplay for progress visualization.
        aim_run: Aim Run instance for experiment tracking.
        effective_batch_size: Number of samples per optimizer step
            (batch_size * gradient_accumulation_steps). Used to compute
            samples_seen for Aim logging x-axis.
        gradient_accumulation_steps: Number of steps to accumulate gradients.
        max_grad_norm: Maximum gradient norm for clipping.
        use_amp: Whether to use automatic mixed precision.
        use_legal_masks: Whether to pass legal masks to the loss function.

    Returns:
        Tuple of (metrics_dict, updated_global_step) where metrics_dict contains
        loss, top1_acc, top3_acc, top5_acc, batch_load_time_avg, step_time_avg.
    """
    model.train()
    optimizer.zero_grad()

    # Initialize metrics tracker
    metrics = MetricsTracker(k_values=[1, 3, 5])

    # Step-level accumulators for Aim logging (aggregated across gradient accumulation)
    step_loss_sum = 0.0
    step_sample_count = 0
    step_topk_correct = {1: 0, 3: 0, 5: 0}
    step_batch_load_time_sum = 0.0
    step_compute_time_sum = 0.0
    last_complete_step_time = 0.0  # Time for the last completed optimizer step

    # Start epoch in display
    display.start_epoch(epoch, steps=len(dataloader), phase="Train")

    # Start timing for first batch load
    batch_timer = Timer().start()

    for batch_idx, batch in enumerate(dataloader):
        # Record batch loading time
        batch_load_time = batch_timer.stop()
        metrics.update_batch_load_time(batch_load_time)
        step_batch_load_time_sum += batch_load_time

        # Start compute timer
        compute_timer = Timer().start()

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

        # Compute raw per-batch metrics
        batch_size = from_targets.size(0)
        raw_loss = loss.item() * gradient_accumulation_steps  # Unscaled loss
        raw_topk = compute_topk_accuracy(
            from_logits, to_logits, from_targets, to_targets, k_values=[1, 3, 5]
        )

        # Accumulate for step-level Aim logging
        step_loss_sum += raw_loss * batch_size
        step_sample_count += batch_size
        for k in [1, 3, 5]:
            step_topk_correct[k] += raw_topk[k]

        # Update sliding window metrics (for display smoothing)
        metrics.update_loss(raw_loss, batch_size)
        metrics.update_accuracy_from_counts(raw_topk, batch_size)

        # Record compute time for this batch
        compute_time = compute_timer.stop()
        step_compute_time_sum += compute_time

        # Update weights after accumulation
        if (batch_idx + 1) % gradient_accumulation_steps == 0:
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_grad_norm
            )
            scaler.step(optimizer)
            scale_before = scaler.get_scale()
            scaler.update()

            # Only step scheduler if optimizer actually stepped (no inf/NaN grads)
            if scaler.get_scale() >= scale_before:
                scheduler.step()
            optimizer.zero_grad()
            global_step += 1

            # Compute step-level metrics (averaged across all batches in step)
            step_loss = step_loss_sum / step_sample_count
            step_top1_acc = step_topk_correct[1] / step_sample_count
            step_top3_acc = step_topk_correct[3] / step_sample_count
            step_top5_acc = step_topk_correct[5] / step_sample_count

            # Store step time for display
            last_complete_step_time = step_compute_time_sum
            metrics.update_step_time(last_complete_step_time)

            # Log every step (display uses smoothed values)
            current_lr = scheduler.get_last_lr()[0]
            log_msg = (
                f"Step {global_step:>6} | "
                f"Loss: {metrics.loss.average():>7.4f} | "
                f"Top-1: {100 * metrics.get_topk_accuracy(1):>5.2f}% | "
                f"LR: {current_lr:.2e}"
            )
            display.log(log_msg)

            # Track step-level metrics to Aim (aggregated across accumulation batches)
            # Use samples_seen as the x-axis for fair comparison across different batch sizes
            samples_seen = global_step * effective_batch_size
            aim_run.track(
                {
                    "loss": step_loss,
                    "top1_acc": step_top1_acc,
                    "top3_acc": step_top3_acc,
                    "top5_acc": step_top5_acc,
                    "lr": current_lr,
                    "grad_norm": grad_norm.item(),
                    "batch_load_time_ms": step_batch_load_time_sum * 1000,
                    "step_time_ms": step_compute_time_sum * 1000,
                    "epoch": epoch,
                    "step": global_step,
                },
                context={"subset": "train"},
                step=samples_seen,
                epoch=epoch,
            )

            # Reset step-level accumulators
            step_loss_sum = 0.0
            step_sample_count = 0
            step_topk_correct = {1: 0, 3: 0, 5: 0}
            step_batch_load_time_sum = 0.0
            step_compute_time_sum = 0.0

        # Update display (uses smoothed values from sliding window)
        current_lr = scheduler.get_last_lr()[0]
        display.update_step(
            loss=metrics.loss.average(),
            top1_acc=metrics.get_topk_accuracy(1),
            top3_acc=metrics.get_topk_accuracy(3),
            top5_acc=metrics.get_topk_accuracy(5),
            lr=current_lr,
            data_time=metrics.batch_load_time.average(),
            step_time=metrics.step_time.average(),  # Smoothed per-step time
        )

        # Start timing for next batch load
        batch_timer.start()

    # End epoch in display
    display.end_epoch()

    return metrics.get_metrics(), global_step


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    epoch: int,
    global_step: int,
    display: TrainingDisplay,
    aim_run: AimRun,
    effective_batch_size: int,
    use_amp: bool = True,
    use_legal_masks: bool = False,
) -> Dict[str, float]:
    """Validate the model on the validation set.

    Args:
        model: The model to validate.
        dataloader: DataLoader for validation data.
        loss_fn: Loss function to use.
        device: Device to run validation on.
        epoch: Current epoch number (for logging).
        global_step: Current global training step (for Aim logging).
        display: TrainingDisplay for progress visualization.
        aim_run: Aim Run instance for experiment tracking.
        effective_batch_size: Number of samples per training optimizer step.
            Used to compute samples_seen for Aim logging x-axis.
        use_amp: Whether to use automatic mixed precision.
        use_legal_masks: Whether to pass legal masks to the loss function.

    Returns:
        Dictionary with loss, top1_acc, top3_acc, top5_acc,
        batch_load_time_avg, step_time_avg.
    """
    model.eval()

    # Initialize metrics tracker with running average for full-epoch metrics
    metrics = MetricsTracker(k_values=[1, 3, 5], use_running_average=True)

    # Start epoch in display
    display.start_epoch(epoch, steps=len(dataloader), phase="Val")

    # Start timing for first batch load
    batch_timer = Timer().start()

    with torch.no_grad():
        for batch in dataloader:
            # Record batch loading time
            batch_load_time = batch_timer.stop()
            metrics.update_batch_load_time(batch_load_time)

            # Start step timer
            step_timer = Timer().start()

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

            # Track metrics
            batch_size = from_targets.size(0)
            metrics.update_loss(loss.item(), batch_size)

            # Compute top-k accuracy
            metrics.update_accuracy(from_logits, to_logits, from_targets, to_targets)

            # Record step time
            step_time = step_timer.stop()
            metrics.update_step_time(step_time)

            # Update display (only loss/accuracy, preserve training LR/timing in "Other" panel)
            display.update_step(
                loss=metrics.loss.average(),
                top1_acc=metrics.get_topk_accuracy(1),
                top3_acc=metrics.get_topk_accuracy(3),
                top5_acc=metrics.get_topk_accuracy(5),
                # lr, data_time, step_time omitted to preserve training values
            )

            # Start timing for next batch load
            batch_timer.start()

    # End epoch in display
    display.end_epoch()

    # Track validation metrics to Aim (once per epoch)
    # Use samples_seen as the x-axis for fair comparison across different batch sizes
    final_metrics = metrics.get_metrics()
    samples_seen = global_step * effective_batch_size
    aim_run.track(
        {
            "loss": final_metrics["loss"],
            "top1_acc": final_metrics["top1_acc"],
            "top3_acc": final_metrics["top3_acc"],
            "top5_acc": final_metrics["top5_acc"],
            "epoch": epoch,
            "step": global_step,
        },
        context={"subset": "val"},
        step=samples_seen,
        epoch=epoch,
    )

    return final_metrics


def train_model(
    config: ModelConfig,
    checkpoint_folder: Optional[Union[str, Path]] = None,
    resume_run: Optional[str] = None,
) -> None:
    """Train a chess transformer model.

    Main entry point for training. Creates model, datasets, dataloaders,
    optimizer, scheduler, and runs the training loop with validation.

    Each training run gets its own checkpoint subfolder named after the run
    (e.g., `Vole_v1.0_20251214_160305`). This prevents checkpoint overwrites
    and makes it easy to identify which checkpoints belong to which run.

    Args:
        config: Model configuration containing architecture and training
            parameters. Must have train_config and dataloader_config set.
        checkpoint_folder: Base path for checkpoints. If None, uses
            train_config.checkpoint_dir or CT_CHECKPOINTS_FOLDER env var.
            The actual checkpoints are saved to a subfolder named after the run.
        resume_run: Name of an existing run to resume (e.g.,
            "Vole_v1.0_20251214_160305"). If provided, loads checkpoints from
            that run's folder and continues logging to the same Aim run.
            If None, starts a fresh run with a new timestamp-based name.

    Raises:
        ValueError: If required config attributes are missing, or if resume_run
            is specified but the run doesn't exist.
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

    # Set up base checkpoint directory
    # Priority: CLI arg > config.train_config.checkpoint_dir > CT_CHECKPOINTS_FOLDER env var
    # The actual checkpoint folder will be base_checkpoint_dir / run_name
    if checkpoint_folder is None:
        if train_config.checkpoint_dir is not None:
            base_checkpoint_dir = Path(train_config.checkpoint_dir)
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
            base_checkpoint_dir = Path(checkpoint_root)
    else:
        base_checkpoint_dir = Path(checkpoint_folder)
    base_checkpoint_dir.mkdir(parents=True, exist_ok=True)

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
    train_dataloader_kwargs["drop_last"] = True
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

    # Apply torch.compile() with config settings
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

    # Compute total steps and effective batch size
    steps_per_epoch = len(train_loader) // train_config.gradient_accumulation_steps
    total_steps = steps_per_epoch * train_config.epochs
    effective_batch_size = (
        dataloader_config.batch_size * train_config.gradient_accumulation_steps
    )
    logger.info(
        f"Training for {train_config.epochs} epochs, "
        f"{steps_per_epoch} steps/epoch, {total_steps} total steps"
    )
    logger.info(f"Effective batch size: {effective_batch_size:,}")

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

    # Initialize Aim repository path and experiment name
    aim_repo_path = str(train_config.log_dir) if train_config.log_dir else None
    experiment_name = f"{config.name}_v{config.version}"

    # Build complete hyperparameters dictionary
    hparams = {
        # ModelConfig - architecture
        "model_type": config.model_type.__name__,
        "model_name": config.name,
        "model_version": config.version,
        "vocab_sizes": config.vocab_sizes,
        "d_model": config.d_model,
        "n_heads": config.n_heads,
        "d_ff": config.d_ff,
        "n_layers": config.n_layers,
        "dropout": config.dropout,
        "share_rel_pos_embeddings": config.share_rel_pos_embeddings,
        "disable_compilation": config.disable_compilation,
        "compilation_mode": config.compilation_mode,
        "dynamic_compilation": config.dynamic_compilation,
        "fullgraph_compilation": config.fullgraph_compilation,
        # TrainConfig - training
        "epochs": train_config.epochs,
        "gradient_accumulation_steps": train_config.gradient_accumulation_steps,
        "optimizer": train_config.optimizer.__name__,
        "optimizer_args": train_config.optimizer_args,
        "lr_schedule": train_config.lr_schedule.__name__,
        "lr_schedule_args": train_config.lr_schedule_args,
        "max_grad_norm": train_config.max_grad_norm,
        "use_amp": train_config.use_amp,
        "loss_fn": train_config.loss_fn.__name__,
        "loss_fn_args": train_config.loss_fn_args,
        "log_dir": str(train_config.log_dir) if train_config.log_dir else None,
        "checkpoint_dir": (
            str(train_config.checkpoint_dir) if train_config.checkpoint_dir else None
        ),
        # DataLoaderConfig - data loading
        "dataset": dataloader_config.dataset.__name__,
        "batch_size": dataloader_config.batch_size,
        "num_workers": dataloader_config.num_workers,
        "shuffle": dataloader_config.shuffle,
        "pin_memory": dataloader_config.pin_memory,
        "drop_last": dataloader_config.drop_last,
        "prefetch_factor": dataloader_config.prefetch_factor,
        "persistent_workers": dataloader_config.persistent_workers,
        "lmdb_filepath": str(lmdb_path),
        # Dataset metadata
        "source_name": dataloader_config.source_name,
        "n_rows": dataloader_config.n_rows,
        "val_split_fraction": dataloader_config.val_split_fraction,
        "legal_mask_mode": dataloader_config.legal_mask_mode,
        # Computed values
        "n_params": n_params,
        "effective_batch_size": effective_batch_size,
    }

    # Initialize training state
    start_epoch = 0
    global_step = 0
    best_val_top1_acc = 0.0

    # Determine run name and set up checkpoint folder
    if resume_run is not None:
        # Resume an existing run
        run_name = resume_run
        checkpoint_folder = base_checkpoint_dir / run_name

        # Verify the checkpoint folder and checkpoint exist
        latest_checkpoint = checkpoint_folder / "latest.pt"
        if not latest_checkpoint.exists():
            logger.critical(
                f"Cannot resume run '{run_name}': checkpoint not found at "
                f"{latest_checkpoint}"
            )
            raise ValueError(
                f"Cannot resume run '{run_name}': no checkpoint found. "
                f"Expected checkpoint at {latest_checkpoint}"
            )

        # Find the existing Aim run by name (requires configured log directory)
        if not aim_repo_path:
            logger.critical(
                "Cannot resume run: no log directory configured. "
                "Set train_config.log_dir to enable run resumption."
            )
            raise ValueError(
                "Cannot resume run: no Aim repository configured. "
                "Set train_config.log_dir to enable run resumption."
            )

        try:
            repo = AimRepo(aim_repo_path)
            query = f"run.name == '{run_name}'"
            matching_runs = list(repo.query_runs(query).iter_runs())
            if matching_runs:
                # Get the run hash and reconnect
                run_hash = matching_runs[0].hash
                aim_run = AimRun(
                    run_hash=run_hash,
                    repo=aim_repo_path,
                    experiment=experiment_name,
                )
                logger.info(f"Reconnected to Aim run: {run_name} ({run_hash})")
            else:
                logger.critical(f"Cannot find Aim run with name '{run_name}'")
                raise ValueError(
                    f"Cannot resume run '{run_name}': Aim run not found in "
                    f"repository at {aim_repo_path}."
                )
        except ValueError:
            # Re-raise ValueError without wrapping
            raise
        except Exception as e:
            logger.critical(f"Error querying Aim repository: {e}")
            raise ValueError(
                f"Cannot resume run '{run_name}': failed to query Aim repository. "
                f"Error: {e}"
            )

        # Load the checkpoint
        logger.info(f"Resuming from checkpoint: {latest_checkpoint}")
        start_epoch, global_step, best_val_top1_acc = load_checkpoint(
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
            aim_run.close()
            train_dataset.close()
            val_dataset.close()
            return
    else:
        # Start a fresh run with a new timestamp-based name
        run_name = (
            f"{config.name}_v{config.version}_"
            f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        checkpoint_folder = base_checkpoint_dir / run_name
        # Note: checkpoint folder is created on first checkpoint save

        # Auto-generate description by comparing to previous run
        description = get_hparam_diff(aim_repo_path, experiment_name, hparams)
        logger.info(f"Run description: {description}")

        # Initialize Aim repository if it doesn't exist
        aim_dir = Path(aim_repo_path) / ".aim"
        if not aim_dir.exists():
            AimRepo.from_path(aim_repo_path, init=True)
            logger.info(f"Initialized new Aim repository at {aim_repo_path}")

        # Create the Aim run with all metadata
        aim_run = AimRun(repo=aim_repo_path, experiment=experiment_name)
        aim_run.name = run_name
        aim_run.description = description
        aim_run["hparams"] = hparams

        # Add tags for filtering runs
        aim_run.add_tag(config.model_type.__name__)  # Model architecture
        aim_run.add_tag(dataloader_config.dataset.__name__)  # PyTorch dataset class
        if dataloader_config.source_name:
            aim_run.add_tag(dataloader_config.source_name)  # Data source name
        aim_run.add_tag(train_config.loss_fn.__name__)  # Loss function
        aim_run.add_tag(train_config.lr_schedule.__name__)  # LR scheduler
        aim_run.add_tag(train_config.optimizer.__name__)  # Optimizer
        if dataloader_config.legal_mask_mode:
            aim_run.add_tag(f"mask_{dataloader_config.legal_mask_mode}")
        aim_run.add_tag("amp" if train_config.use_amp else "fp32")
        aim_run.add_tag("compiled" if not config.disable_compilation else "uncompiled")
        if not config.disable_compilation:
            aim_run.add_tag(f"compilation_mode_{config.compilation_mode}")
        aim_run.add_tag(_get_data_size_bucket(dataloader_config.n_rows))
        aim_run.add_tag(_get_model_size_bucket(n_params))

        # Log hardware information
        hardware_info = {
            "gpu_name": torch.cuda.get_device_name(0),
            "gpu_count": torch.cuda.device_count(),
            "gpu_memory_gb": torch.cuda.get_device_properties(0).total_memory / 1e9,
            "cuda_version": torch.version.cuda,
            "pytorch_version": str(torch.__version__),
        }
        aim_run["hardware"] = hardware_info
        aim_run.add_tag(torch.cuda.get_device_name(0).replace(" ", "_"))

        logger.info(f"Initialized new Aim run: {aim_run.name} ({aim_run.hash})")
        logger.info("Starting fresh training run")

    logger.info(f"Checkpoints will be saved to {checkpoint_folder}")

    # Training loop
    try:
        # Create the rich training display
        remaining_epochs = train_config.epochs - start_epoch
        with TrainingDisplay(
            total_epochs=remaining_epochs,
            model_name=f"{config.name} v{config.version}",
        ) as display:
            for epoch in range(start_epoch, train_config.epochs):
                # Log epoch start to display
                display.log(f"Starting epoch {epoch + 1}/{train_config.epochs}")

                # Train
                train_metrics, global_step = train_one_epoch(
                    model=model,
                    dataloader=train_loader,
                    loss_fn=loss_fn,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    device=device,
                    epoch=epoch + 1,
                    global_step=global_step,
                    display=display,
                    aim_run=aim_run,
                    effective_batch_size=effective_batch_size,
                    gradient_accumulation_steps=train_config.gradient_accumulation_steps,
                    max_grad_norm=train_config.max_grad_norm,
                    use_amp=train_config.use_amp,
                    use_legal_masks=use_legal_masks and is_legal_smoothing,
                )
                train_summary = (
                    f"Epoch {epoch + 1} Train | Loss: {train_metrics['loss']:.4f} | "
                    f"Top-1: {100 * train_metrics['top1_acc']:.2f}% | "
                    f"Top-3: {100 * train_metrics['top3_acc']:.2f}% | "
                    f"Top-5: {100 * train_metrics['top5_acc']:.2f}%"
                )
                display.log(train_summary)

                # Validate every epoch
                val_metrics = validate(
                    model=model,
                    dataloader=val_loader,
                    loss_fn=loss_fn,
                    device=device,
                    epoch=epoch + 1,
                    global_step=global_step,
                    display=display,
                    aim_run=aim_run,
                    effective_batch_size=effective_batch_size,
                    use_amp=train_config.use_amp,
                    use_legal_masks=use_legal_masks and is_legal_smoothing,
                )
                val_loss = val_metrics["loss"]
                val_summary = (
                    f"Epoch {epoch + 1} Val | Loss: {val_loss:.4f} | "
                    f"Top-1: {100 * val_metrics['top1_acc']:.2f}% | "
                    f"Top-3: {100 * val_metrics['top3_acc']:.2f}% | "
                    f"Top-5: {100 * val_metrics['top5_acc']:.2f}%"
                )
                display.log(val_summary)

                # Record validation metrics in display
                display.record_validation(
                    epoch=epoch + 1,
                    loss=val_loss,
                    top1_acc=val_metrics["top1_acc"],
                    top3_acc=val_metrics["top3_acc"],
                    top5_acc=val_metrics["top5_acc"],
                )

                # Save best model (based on highest top-1 accuracy)
                val_top1_acc = val_metrics["top1_acc"]
                if val_top1_acc > best_val_top1_acc:
                    best_val_top1_acc = val_top1_acc
                    best_path = checkpoint_folder / "best.pt"
                    save_checkpoint(
                        best_path,
                        model,
                        optimizer,
                        scheduler,
                        scaler,
                        epoch,
                        global_step,
                        best_val_top1_acc,
                    )
                    display.log(
                        f"New best model saved (val_top1_acc={100 * val_top1_acc:.2f}%)"
                    )
                    aim_run.track(
                        AimText(
                            f"best.pt (epoch={epoch + 1}, "
                            f"val_top1_acc={100 * val_top1_acc:.2f}%)"
                        ),
                        name="checkpoints",
                        step=global_step,
                        epoch=epoch + 1,
                    )

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
                    best_val_top1_acc,
                )
                aim_run.track(
                    AimText(f"latest.pt (epoch={epoch + 1}, step={global_step})"),
                    name="checkpoints",
                    step=global_step,
                    epoch=epoch + 1,
                )

            display.log("Training complete!")

        logger.info("Training complete!")

    finally:
        # Clean up resources even if an exception occurs
        aim_run.close()
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
        help="Base checkpoint directory (default: from config or env var). "
        "Each run creates a subfolder named after the run.",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        metavar="RUN_NAME",
        help="Resume a previous run by name (e.g., 'Vole_v1.0_20251214_160305'). "
        "Loads checkpoints and continues logging to the same Aim run.",
    )
    args = parser.parse_args()

    # Import config and train
    config = import_config(args.config_name)
    train_model(
        config=config,
        checkpoint_folder=args.checkpoint_folder,
        resume_run=args.resume,
    )
