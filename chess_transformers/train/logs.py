"""Logging utilities for training chess transformer models.

This module provides helper classes and functions for tracking training
metrics, computing accuracies, timing operations, and displaying rich
terminal output during training.

Key Features:
    - RunningAverage: Tracks running averages of scalar metrics.
    - Timer: Context manager for timing code blocks.
    - compute_topk_accuracy: Computes top-k accuracy for move predictions.
    - MetricsTracker: Comprehensive tracker for training metrics.
    - TrainingDisplay: Rich terminal display with progress bars and sparklines.

Notes:
    These utilities are designed to work with the training loop in
    `chess_transformers.train.train` and provide detailed logging
    of accuracy and timing metrics.
"""

import time
import torch

from rich.live import Live
from rich.text import Text
from rich.align import Align
from rich.panel import Panel
from collections import deque
from rich.table import Column
from rich.spinner import Spinner
from typing import Dict, List, Tuple
from rich.console import Console, Group
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    ProgressColumn,
    SpinnerColumn,
    Task,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)


# Sparkline characters for mini graphs
SPARKLINE_CHARS = "▁▂▃▄▅▆▇█"  # 8 levels (0-7)

# Colors
TRAIN_BORDER_COLOR = "grey82"
VAL_BORDER_COLOR = "grey82"
DIM_COLOR = "grey42"  # Explicit grey for cross-terminal compatibility


def interpolate_color(
    val: float, start_rgb: Tuple[int, int, int], end_rgb: Tuple[int, int, int]
) -> str:
    """Interpolate between two RGB colors.

    Performs linear interpolation between two RGB color values based on
    a normalized interpolation factor.

    Args:
        val: Interpolation factor between 0.0 and 1.0. At 0.0, returns
            start_rgb; at 1.0, returns end_rgb.
        start_rgb: Starting RGB color as a tuple of (red, green, blue)
            integers in range 0-255.
        end_rgb: Ending RGB color as a tuple of (red, green, blue)
            integers in range 0-255.

    Returns:
        CSS-style RGB color string in format "rgb(r,g,b)".
    """
    r = int(start_rgb[0] + (end_rgb[0] - start_rgb[0]) * val)
    g = int(start_rgb[1] + (end_rgb[1] - start_rgb[1]) * val)
    b = int(start_rgb[2] + (end_rgb[2] - start_rgb[2]) * val)
    return f"rgb({r},{g},{b})"


def get_gradient_color(normalized_val: float, metric_type: str = "loss") -> str:
    """Get color for a normalized value (0-1) based on metric type.

    Loss: Green (low) -> Yellow -> Red (high)
    Accuracy: Red (low) -> Yellow -> Green (high)
    """
    # RGB constants (darker versions)
    color_red = (135, 0, 0)
    color_green = (0, 135, 0)
    color_yellow = (135, 135, 0)

    # Clamp
    v = max(0.0, min(1.0, normalized_val))

    if metric_type == "loss":
        # For loss: low values are good (green), high values are bad (red)
        if v < 0.5:
            # Interpolate from green to yellow for lower half
            return interpolate_color(v * 2, color_green, color_yellow)
        else:
            # Interpolate from yellow to red for upper half
            return interpolate_color((v - 0.5) * 2, color_yellow, color_red)
    else:
        # For accuracy: high values are good (green), low values are bad (red)
        if v < 0.5:
            # Interpolate from red to yellow for lower half
            return interpolate_color(v * 2, color_red, color_yellow)
        else:
            # Interpolate from yellow to green for upper half
            return interpolate_color((v - 0.5) * 2, color_yellow, color_green)


def gradient_sparkline(
    values: List[float], width: int = 20, metric_type: str = "loss"
) -> Text:
    """Generate a sparkline Text object with gradient colors.

    The sparkline always occupies the full specified width, padding with
    spaces on the left if fewer values are available.

    Args:
        values: List of historical values to display.
        width: Fixed width for the sparkline in characters.
        metric_type: Type of metric ("loss" or "acc") for gradient coloring.

    Returns:
        Rich Text object containing the sparkline, always of the specified width.
    """
    if not values:
        return Text(" " * width)

    # Keep only the most recent values if there are more than width
    # (older data on left is dropped, newer data on right is kept)
    if len(values) > width:
        values = values[-width:]

    min_val = min(values)
    max_val = max(values)

    # Avoid division by zero
    diff = max_val - min_val
    if diff == 0:
        diff = 1.0

    text = Text()

    # Center sparkline by splitting padding between left and right
    padding_needed = width - len(values)
    left_padding = padding_needed // 2
    right_padding = padding_needed - left_padding

    if left_padding > 0:
        text.append(" " * left_padding)

    for v in values:
        normalized = (v - min_val) / diff

        # Determine character
        char_idx = int(normalized * 7)
        char_idx = min(7, max(0, char_idx))
        char = SPARKLINE_CHARS[char_idx]

        # Determine color
        color = get_gradient_color(normalized, metric_type)

        text.append(char, style=color)

    if right_padding > 0:
        text.append(" " * right_padding)

    return text


class RightAlignedTaskColumn(ProgressColumn):
    """Renders spinner and task description, right aligned in a single column."""

    def __init__(self, spinner_name: str = "dots") -> None:
        """Initialize the column.

        Args:
            spinner_name: Name of the spinner to use.
        """
        self.spinner = Spinner(spinner_name)
        # Use a table column with right justification
        super().__init__(table_column=Column(justify="right"))

    def render(self, task: Task) -> Text:
        """Render the spinner and description.

        Args:
            task: The progress task.

        Returns:
            Rich Text object with spinner and description.
        """
        text = Text()

        # Render spinner frame
        frame = self.spinner.render(time.time())
        text.append(frame)
        text.append(" ")

        # Render description
        text.append(task.description)

        return text


class RunningAverage:
    """Tracks a running average of a scalar metric.

    Maintains cumulative sum and count to efficiently compute the
    running average without storing all values.

    Attributes:
        total: Cumulative sum of all values.
        count: Number of values added.
    """

    def __init__(self) -> None:
        """Initialize the running average tracker."""
        self.total = 0.0
        self.count = 0

    def update(self, value: float, n: int = 1) -> None:
        """Add a value (or batch of values) to the running average.

        Args:
            value: The value to add. If n > 1, this should be the sum
                of n individual values.
            n: Number of samples this value represents.
        """
        self.total += value
        self.count += n

    def average(self) -> float:
        """Compute the current running average.

        Returns:
            The running average, or 0.0 if no values have been added.
        """
        if self.count == 0:
            return 0.0
        return self.total / self.count

    def reset(self) -> None:
        """Reset the running average tracker."""
        self.total = 0.0
        self.count = 0


class SlidingWindowAverage:
    """Tracks a sliding window average of a scalar metric.

    Maintains a fixed-size window of recent values to compute
    an average that reflects recent performance rather than
    the entire epoch.

    Attributes:
        window_size: Maximum number of entries to keep.
        values: Deque of (value, count) tuples for each entry.
    """

    def __init__(self, window_size: int = 100) -> None:
        """Initialize the sliding window average tracker.

        Args:
            window_size: Number of entries to keep in the window.
        """
        self.window_size = window_size
        self.values = deque(maxlen=window_size)

    def update(self, value: float, n: int = 1) -> None:
        """Add a value (or batch of values) to the sliding window.

        Args:
            value: The value to add. If n > 1, this should be the sum
                of n individual values.
            n: Number of samples this value represents.
        """
        self.values.append((value, n))

    def average(self) -> float:
        """Compute the current sliding window average.

        Returns:
            The sliding window average, or 0.0 if empty.
        """
        if not self.values:
            return 0.0
        total = sum(v for v, n in self.values)
        count = sum(n for v, n in self.values)
        if count == 0:
            return 0.0
        return total / count

    def reset(self) -> None:
        """Reset the sliding window tracker."""
        self.values.clear()


class Timer:
    """Context manager for timing code blocks.

    Can be used as a context manager or manually with start()/stop().

    Attributes:
        elapsed: Time elapsed in seconds (after stop() or exiting context).

    Example:
        >>> with Timer() as t:
        ...     # code to time
        >>> print(f"Elapsed: {t.elapsed:.3f}s")
    """

    def __init__(self) -> None:
        """Initialize the timer."""
        self._start_time = None
        self.elapsed = 0.0

    def start(self) -> "Timer":
        """Start the timer.

        Returns:
            Self for method chaining.
        """
        self._start_time = time.perf_counter()
        return self

    def stop(self) -> float:
        """Stop the timer and return elapsed time.

        Returns:
            Elapsed time in seconds.
        """
        if self._start_time is not None:
            self.elapsed = time.perf_counter() - self._start_time
            self._start_time = None
        return self.elapsed

    def __enter__(self) -> "Timer":
        """Enter context manager and start timer."""
        return self.start()

    def __exit__(self, *args) -> None:
        """Exit context manager and stop timer."""
        self.stop()


def compute_topk_accuracy(
    from_logits: torch.Tensor,
    to_logits: torch.Tensor,
    from_targets: torch.Tensor,
    to_targets: torch.Tensor,
    k_values: List[int] = None,
) -> Dict[int, int]:
    """Compute top-k accuracy for move predictions.

    A move is considered correct at top-k if:
    - The correct from-square is in the top-k from predictions, AND
    - The correct to-square is in the top-k to predictions.

    This is a strict criterion where both components must be in their
    respective top-k predictions for the move to count as correct.

    Args:
        from_logits: From-square logits, shape (batch, 1, 64) or (batch, 64).
        to_logits: To-square logits, shape (batch, 1, 64) or (batch, 64).
        from_targets: From-square targets, shape (batch,).
        to_targets: To-square targets, shape (batch,).
        k_values: List of k values to compute accuracy for.
            Defaults to [1, 3, 5].

    Returns:
        Dictionary mapping k to number of correct predictions at that k.
    """
    if k_values is None:
        k_values = [1, 3, 5]

    # Squeeze out the extra dimension if present
    if from_logits.dim() == 3:
        from_logits = from_logits.squeeze(1)
    if to_logits.dim() == 3:
        to_logits = to_logits.squeeze(1)

    results = {}
    for k in k_values:
        # Get top-k predictions for each head
        _, from_topk = from_logits.topk(k, dim=-1)  # (batch, k)
        _, to_topk = to_logits.topk(k, dim=-1)  # (batch, k)

        # Check if target is in top-k for each head
        from_correct = (from_topk == from_targets.unsqueeze(-1)).any(dim=-1)
        to_correct = (to_topk == to_targets.unsqueeze(-1)).any(dim=-1)

        # Both must be correct for the move to be correct
        both_correct = from_correct & to_correct
        results[k] = both_correct.sum().item()

    return results


class MetricsTracker:
    """Comprehensive tracker for training and validation metrics.

    Tracks loss, top-k accuracies, and timing metrics. Supports two modes:
    - Sliding window average (default): For training, provides responsive
      feedback based on the last N steps.
    - Running average: For validation, computes metrics over the entire
      dataset for accurate epoch-level reporting.

    Attributes:
        window_size: Number of steps to include in the sliding window.
        use_running_average: If True, use cumulative running averages.
        loss: Average tracker for loss values.
        topk_accuracy: Dictionary of average trackers for accuracy.
        batch_load_time: Average tracker for batch loading time.
        step_time: Average tracker for step time.
    """

    def __init__(
        self,
        k_values: List[int] = None,
        window_size: int = 100,
        use_running_average: bool = False,
    ) -> None:
        """Initialize the metrics tracker.

        Args:
            k_values: List of k values to track for top-k accuracy.
                Defaults to [1, 3, 5].
            window_size: Number of steps to include in the sliding window.
                Defaults to 100. Ignored if use_running_average is True.
            use_running_average: If True, use cumulative running averages
                (for validation). If False, use sliding window averages
                (for training). Defaults to False.
        """
        self.k_values = k_values if k_values is not None else [1, 3, 5]
        self.window_size = window_size
        self.use_running_average = use_running_average

        # Select tracker class based on mode
        if use_running_average:
            tracker_class = RunningAverage
            tracker_args = ()
        else:
            tracker_class = SlidingWindowAverage
            tracker_args = (window_size,)

        # Metric trackers
        self.loss = tracker_class(*tracker_args)
        # For accuracy, we store (correct, total) pairs per step
        self.topk_accuracy = {k: tracker_class(*tracker_args) for k in self.k_values}

        # Timing trackers
        self.batch_load_time = tracker_class(*tracker_args)
        self.step_time = tracker_class(*tracker_args)

    def update_loss(self, loss: float, batch_size: int) -> None:
        """Update loss metric.

        Args:
            loss: Loss value for the batch.
            batch_size: Number of samples in the batch.
        """
        self.loss.update(loss * batch_size, batch_size)

    def update_accuracy(
        self,
        from_logits: torch.Tensor,
        to_logits: torch.Tensor,
        from_targets: torch.Tensor,
        to_targets: torch.Tensor,
    ) -> None:
        """Update accuracy metrics by computing from logits.

        This method computes top-k accuracy internally. If you have
        pre-computed accuracy counts, use update_accuracy_from_counts()
        instead to avoid redundant computation.

        Args:
            from_logits: From-square logits.
            to_logits: To-square logits.
            from_targets: From-square targets.
            to_targets: To-square targets.
        """
        batch_size = from_targets.size(0)

        topk_results = compute_topk_accuracy(
            from_logits, to_logits, from_targets, to_targets, self.k_values
        )
        for k, correct in topk_results.items():
            # Store (correct_count, total_count) so we can compute accuracy
            self.topk_accuracy[k].update(correct, batch_size)

    def update_accuracy_from_counts(
        self,
        topk_correct: Dict[int, int],
        batch_size: int,
    ) -> None:
        """Update accuracy metrics from pre-computed correct counts.

        Use this method when you've already computed top-k accuracy counts
        (e.g., for logging raw values to an experiment tracker) to avoid
        redundant computation.

        Args:
            topk_correct: Dictionary mapping k values to correct counts.
            batch_size: Number of samples in the batch.
        """
        for k, correct in topk_correct.items():
            if k in self.topk_accuracy:
                self.topk_accuracy[k].update(correct, batch_size)

    def update_batch_load_time(self, seconds: float) -> None:
        """Update batch loading time metric.

        Args:
            seconds: Time taken to load the batch.
        """
        self.batch_load_time.update(seconds)

    def update_step_time(self, seconds: float) -> None:
        """Update step time metric.

        Args:
            seconds: Time taken for one training step.
        """
        self.step_time.update(seconds)

    def get_topk_accuracy(self, k: int) -> float:
        """Get top-k accuracy as a fraction (sliding window).

        Args:
            k: The k value for top-k accuracy.

        Returns:
            Top-k accuracy as a fraction (0.0 to 1.0).
        """
        return self.topk_accuracy[k].average()

    def get_metrics(self) -> Dict[str, float]:
        """Get all current metrics.

        Returns:
            Dictionary with all metric values.
        """
        metrics = {
            "loss": self.loss.average(),
            "batch_load_time_avg": self.batch_load_time.average(),
            "step_time_avg": self.step_time.average(),
        }
        for k in self.k_values:
            metrics[f"top{k}_acc"] = self.get_topk_accuracy(k)
        return metrics

    def get_progress_bar_postfix(self) -> Dict[str, str]:
        """Get formatted metrics for progress bar display.

        Returns:
            Dictionary with formatted string values for tqdm postfix.
        """
        postfix = {
            "loss": f"{self.loss.average():.4f}",
        }
        for k in self.k_values:
            postfix[f"top{k}"] = f"{100 * self.get_topk_accuracy(k):.1f}%"
        postfix["batch_t"] = f"{self.batch_load_time.average() * 1000:.1f}ms"
        postfix["step_t"] = f"{self.step_time.average() * 1000:.1f}ms"
        return postfix

    def get_log_string(self, epoch: int, step: int, lr: float) -> str:
        """Get formatted log string for periodic logging.

        Args:
            epoch: Current epoch number.
            step: Current global step.
            lr: Current learning rate.

        Returns:
            Formatted log string.
        """
        acc_parts = [
            f"Top-{k}: {100 * self.get_topk_accuracy(k):.2f}%" for k in self.k_values
        ]
        return (
            f"Epoch {epoch} | Step {step} | "
            f"Loss: {self.loss.average():.4f} | LR: {lr:.2e} | "
            f"{' | '.join(acc_parts)} | "
            f"Batch Load: {self.batch_load_time.average() * 1000:.2f}ms | "
            f"Step: {self.step_time.average() * 1000:.2f}ms"
        )

    def get_epoch_summary(
        self,
        epoch: int,
        phase: str,
    ) -> Tuple[str, Dict[str, float]]:
        """Get epoch summary string and metrics dictionary.

        Args:
            epoch: Epoch number.
            phase: Phase name (e.g., "Train" or "Val").

        Returns:
            Tuple of (formatted summary string, metrics dictionary).
        """
        metrics = self.get_metrics()
        acc_parts = [
            f"Top-{k} Acc: {100 * metrics[f'top{k}_acc']:.2f}%" for k in self.k_values
        ]
        summary = (
            f"Epoch {epoch} {phase} | Loss: {metrics['loss']:.4f} | "
            f"{' | '.join(acc_parts)}"
        )
        return summary, metrics

    def reset(self) -> None:
        """Reset all metrics for a new epoch."""
        self.loss.reset()
        for k in self.k_values:
            self.topk_accuracy[k].reset()
        self.batch_load_time.reset()
        self.step_time.reset()


class TrainingDisplay:
    """Rich terminal display for training progress.

    Displays:
    - Overall training progress bar (across all epochs)
    - Current epoch/phase progress bar
    - Metrics panel with current values and sparkline graphs
    - Validation history panel showing metrics for all completed epochs
    - Rolling log window

    Attributes:
        console: Rich console for output.
        live: Live display context manager.
        progress: Progress bar manager.

    Example:
        >>> display = TrainingDisplay(total_epochs=10, model_name="Vole")
        >>> with display:
        ...     for epoch in range(10):
        ...         display.start_epoch(epoch + 1, steps=1000, phase="Train")
        ...         for step in range(1000):
        ...             display.update_step(loss=2.5, top1_acc=0.15)
        ...         display.end_epoch()
    """

    def __init__(
        self,
        total_epochs: int,
        model_name: str = "Model",
        log_window_size: int = 3,
        sparkline_history: int = 100,
    ) -> None:
        """Initialize the training display.

        Args:
            total_epochs: Total number of epochs for training.
            model_name: Name of the model being trained.
            log_window_size: Number of log lines to show.
            sparkline_history: Number of data points to keep for sparklines.
        """
        self.total_epochs = total_epochs
        self.model_name = model_name
        self.log_window_size = log_window_size
        self.sparkline_history = sparkline_history

        self.console = Console()
        self.task_column = RightAlignedTaskColumn()
        self.live = None

        # Progress tracking (internal, for rendering inside panels)
        # Bar width calculated to fit within panel: sparkline_width - other columns
        self._progress_bar_width = 20
        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("{task.description}"),
            BarColumn(bar_width=self._progress_bar_width),
            MofNCompleteColumn(),
            TextColumn("•"),
            TimeElapsedColumn(),
            TextColumn("/"),
            TimeRemainingColumn(),
            expand=False,
        )
        self.overall_task = None
        self.epoch_task = None

        # Current state
        self.current_epoch = 0
        self.current_phase = "Train"
        self.current_step = 0
        self.current_lr = 0.0

        # Metrics
        self.loss = 0.0
        self.top1_acc = 0.0
        self.top3_acc = 0.0
        self.top5_acc = 0.0
        self.data_time = 0.0
        self.step_time = 0.0

        # History for sparklines
        self.loss_history = deque(maxlen=sparkline_history)
        self.top1_history = deque(maxlen=sparkline_history)
        self.top3_history = deque(maxlen=sparkline_history)
        self.top5_history = deque(maxlen=sparkline_history)

        # Validation history (stored per epoch)
        self.val_history = []

        # Best validation metrics (based on highest top-1 accuracy)
        self.best_val_epoch = None
        self.best_val_metrics = None

        # Log window
        self.log_lines = deque(maxlen=log_window_size)

    @property
    def sparkline_width(self) -> int:
        """Compute sparkline width dynamically based on terminal size.

        Returns:
            Width in characters for sparklines, accounting for panel borders.
        """
        # Account for panel borders (2 chars each side) and some padding
        return max(20, self.console.width - 6)

    def __enter__(self) -> "TrainingDisplay":
        """Start the live display.

        Uses alternate screen buffer to prevent resize artifacts.
        """
        self.overall_task = self.progress.add_task(
            "Overall",
            total=self.total_epochs,
        )
        self.live = Live(
            self._build_display(),
            console=self.console,
            refresh_per_second=4,
            screen=True,  # Use alternate screen buffer
        )
        self.live.__enter__()
        return self

    def __exit__(self, *args) -> None:
        """Stop the live display and print final state to main terminal."""
        if self.live:
            # Capture final display before exiting alternate screen
            final_display = self._build_display()
            self.live.__exit__(*args)
            # Print final state to main terminal so it remains visible
            self.console.print(final_display)

    def _get_progress_display(self) -> Align:
        """Get the combined progress display for all active tasks.

        Creates a single Progress table containing all active tasks
        so that columns align perfectly vertically.
        """
        # Create a new Progress display for rendering
        progress_display = Progress(
            self.task_column,
            BarColumn(bar_width=self._progress_bar_width),
            MofNCompleteColumn(),
            TextColumn("•"),
            TimeElapsedColumn(),
            TextColumn("/"),
            TimeRemainingColumn(),
            expand=False,
        )

        # Collect active tasks
        active_tasks = []
        if self.overall_task is not None and self.overall_task in self.progress._tasks:
            active_tasks.append(self.progress._tasks[self.overall_task])

        if self.epoch_task is not None and self.epoch_task in self.progress._tasks:
            active_tasks.append(self.progress._tasks[self.epoch_task])

        if not active_tasks:
            return Align.center(Text("No active tasks", style=DIM_COLOR))

        # Add tasks to the display progress
        for task in active_tasks:
            new_task_id = progress_display.add_task(
                task.description,
                total=task.total,
                completed=task.completed,
            )
            # Sync timing state manually
            new_task = progress_display.tasks[new_task_id]
            new_task.start_time = task.start_time
            new_task.stop_time = task.stop_time

        return Align.center(progress_display)

    def _build_display(self) -> Group:
        """Build the complete display layout.

        Returns:
            Group containing all display elements.
        """
        # Build merged progress panel
        progress_content = self._get_progress_display()

        progress_panel = Panel(
            progress_content,
            title=f"[bold]{self.model_name} Progress[/]",
            border_style=TRAIN_BORDER_COLOR,
            expand=True,
        )

        # Build combined metrics panels
        train_panel = self._build_combined_train_panel()
        val_panel = self._build_combined_val_panel()

        # Log window (expands with terminal, logs centered)

        if self.log_lines:
            # Stack log lines vertically, each centered
            log_content = Group(
                *[Align.center(Text(line, style="white")) for line in self.log_lines]
            )
            log_panel = Panel(
                log_content,
                title="[bold]Recent Logs[/]",
                border_style=TRAIN_BORDER_COLOR,
                expand=True,
            )
        else:
            log_panel = Panel(
                Align.center(Text("No logs yet...", style=DIM_COLOR)),
                title="[bold]Recent Logs[/]",
                border_style=TRAIN_BORDER_COLOR,
                expand=True,
            )

        # Build the group with all panels
        return Group(progress_panel, train_panel, val_panel, log_panel)

    def _build_metric_row(
        self,
        label: str,
        value: str,
        history: List[float],
        metric_type: str = "loss",
        sparkline_width: int = None,
    ) -> Text:
        """Build a single metric row with label, value, and sparkline.

        Args:
            label: Display label for the metric (e.g., "Loss", "Top-1 Acc").
            value: Formatted current value string.
            history: List of historical values for sparkline.
            metric_type: Type of metric ("loss" or "acc") for gradient coloring.
            sparkline_width: Width for the sparkline. If None, uses a default.

        Returns:
            Rich Text object containing the formatted row.
        """
        # Fixed column widths for alignment
        label_width = 12
        value_width = 10

        if sparkline_width is None:
            # Reserve space for label, value, and some padding
            sparkline_width = max(
                20, self.sparkline_width - label_width - value_width - 8
            )

        # Build label (left-aligned within column)
        label_text = label.ljust(label_width)

        # Build value (right-aligned within column)
        value_text = value.rjust(value_width)

        # Build sparkline
        spark = gradient_sparkline(
            history, width=sparkline_width, metric_type=metric_type
        )

        # Determine color for the value text (matching the latest sparkline point)
        value_style = "white"
        if history:
            # Replicate the normalization logic from gradient_sparkline
            # to ensure the color matches exactly.

            # Use same window as sparkline
            values_in_window = history
            if len(history) > sparkline_width:
                values_in_window = history[-sparkline_width:]

            if values_in_window:
                min_val = min(values_in_window)
                max_val = max(values_in_window)
                diff = max_val - min_val
                if diff == 0:
                    diff = 1.0

                # Get latest value
                latest = values_in_window[-1]
                normalized = (latest - min_val) / diff

                # Get color
                color = get_gradient_color(normalized, metric_type)
                value_style = color

        # Combine into a single Text object
        row = Text()
        row.append(label_text, style="white")
        row.append(value_text, style=value_style)
        row.append("    ")  # Spacer
        row.append_text(spark)

        return row

    def _build_header_row(
        self,
        value_label: str = "Value",
        history_label: str = "History",
        sparkline_width: int = None,
    ) -> Text:
        """Build a header row for the metrics table.

        Creates a header row with "Metric", value, and history columns
        that align with the metric data rows.

        Args:
            value_label: Label for the value column.
            history_label: Label for the history column.
            sparkline_width: Width for the History column. If None, uses a default.

        Returns:
            Rich Text object containing the header row.
        """
        # Fixed column widths matching _build_metric_row
        label_width = 12
        value_width = 10

        if sparkline_width is None:
            sparkline_width = max(
                20, self.sparkline_width - label_width - value_width - 8
            )

        # Build header text
        header = Text()
        header.append("Metric".ljust(label_width), style="bold white")
        header.append(value_label.rjust(value_width), style="bold white")
        header.append("    ")  # Spacer (same as metric rows)
        header.append(history_label.center(sparkline_width), style="bold white")

        return header

    def _build_metric_panel(
        self,
        name: str,
        value: str,
        history: List[float],
        border_color: str,
        metric_type: str = "loss",
    ) -> Panel:
        """Build a panel for a single metric with single-row sparkline.

        Args:
            name: Display name for the metric.
            value: Formatted current value string.
            history: List of historical values for sparkline.
            border_color: Color for the panel border.
            metric_type: Type of metric ("loss" or "acc") for gradient coloring.

        Returns:
            Rich Panel containing the metric visualization.
        """
        # Generate gradient sparkline (width adjusts to terminal size)
        spark_text = gradient_sparkline(
            history, width=self.sparkline_width, metric_type=metric_type
        )

        return Panel(
            Align.center(spark_text),
            title=f"[bold]{name}: [{border_color}]{value}[/{border_color}][/]",
            border_style=border_color,
            expand=True,
        )

    def _build_combined_train_panel(self) -> Panel:
        """Build a combined panel for all training metrics.

        Creates a single panel containing Loss, Top-1/3/5 Accuracy metrics
        with sparklines, plus timing/LR information. Metrics are displayed
        with blank line spacing, and timing info is separated by a divider.

        Returns:
            Rich Panel containing all training metrics.
        """
        elements = []

        # Top padding
        elements.append(Text(""))

        # Determine sparkline width
        label_width = 12
        value_width = 10
        sparkline_width = max(20, self.sparkline_width - label_width - value_width - 8)

        # Header row
        # Determine the number of visible data points
        n_visible = min(len(self.loss_history), sparkline_width)
        history_label = f"History (Last {n_visible} Steps)"
        elements.append(
            Align.center(
                self._build_header_row(
                    value_label="Value",
                    history_label=history_label,
                    sparkline_width=sparkline_width,
                )
            )
        )
        elements.append(Text(""))  # Blank line after header

        # Build metric rows
        loss_row = self._build_metric_row(
            label="Loss",
            value=f"{self.loss:.4f}",
            history=list(self.loss_history),
            metric_type="loss",
            sparkline_width=sparkline_width,
        )
        top1_row = self._build_metric_row(
            label="Top-1 Acc",
            value=f"{100 * self.top1_acc:.2f}%",
            history=list(self.top1_history),
            metric_type="acc",
            sparkline_width=sparkline_width,
        )
        top3_row = self._build_metric_row(
            label="Top-3 Acc",
            value=f"{100 * self.top3_acc:.2f}%",
            history=list(self.top3_history),
            metric_type="acc",
            sparkline_width=sparkline_width,
        )
        top5_row = self._build_metric_row(
            label="Top-5 Acc",
            value=f"{100 * self.top5_acc:.2f}%",
            history=list(self.top5_history),
            metric_type="acc",
            sparkline_width=sparkline_width,
        )

        # Add metric rows with blank lines between them (centered)
        elements.append(Align.center(loss_row))
        elements.append(Text(""))  # Blank line
        elements.append(Align.center(top1_row))
        elements.append(Text(""))  # Blank line
        elements.append(Align.center(top3_row))
        elements.append(Text(""))  # Blank line
        elements.append(Align.center(top5_row))
        elements.append(Text(""))  # Blank line
        elements.append(Text(""))  # Extra blank line

        # Timing/LR info (centered)
        timing_text = Text()
        timing_text.append("LR: ", style="white")
        timing_text.append(f"{self.current_lr:.2e}", style="magenta")
        timing_text.append("  •  ", style="white")
        timing_text.append("Data: ", style="white")
        timing_text.append(f"{self.data_time * 1000:.1f}ms", style="cyan")
        timing_text.append("  •  ", style="white")
        timing_text.append("Step: ", style="white")
        timing_text.append(f"{self.step_time * 1000:.1f}ms", style="cyan")
        elements.append(Align.center(timing_text))
        elements.append(Text(""))  # Blank line (bottom padding)

        return Panel(
            Group(*elements),
            title="[bold]Training Metrics[/]",
            border_style=TRAIN_BORDER_COLOR,
            expand=True,
        )

    def _build_combined_val_panel(self) -> Panel:
        """Build a combined panel for all validation metrics.

        Creates a single panel containing Loss, Top-1/3/5 Accuracy metrics
        with sparklines showing validation history across epochs.

        Returns:
            Rich Panel containing all validation metrics.
        """
        if not self.val_history:
            return Panel(
                Align.center(Text("No validation results yet...", style=DIM_COLOR)),
                title="[bold]Validation Metrics[/]",
                border_style=VAL_BORDER_COLOR,
                expand=True,
            )

        # Extract histories for sparklines
        val_losses = [e["loss"] for e in self.val_history]
        val_top1 = [e["top1_acc"] for e in self.val_history]
        val_top3 = [e["top3_acc"] for e in self.val_history]
        val_top5 = [e["top5_acc"] for e in self.val_history]

        # Use best metrics (from epoch with highest top-1 acc) for display values
        best = self.best_val_metrics if self.best_val_metrics else self.val_history[-1]
        best_epoch = self.best_val_epoch if self.best_val_epoch else best["epoch"]

        elements = []

        # Top padding
        elements.append(Text(""))

        # Determine sparkline width
        label_width = 12
        value_width = 10
        sparkline_width = max(20, self.sparkline_width - label_width - value_width - 8)

        # Header row with epoch number in value column
        n_visible = min(len(self.val_history), sparkline_width)
        value_label = f"Best (Ep. {best_epoch})"
        history_label = f"History (Last {n_visible} Epochs)"
        elements.append(
            Align.center(
                self._build_header_row(
                    value_label=value_label,
                    history_label=history_label,
                    sparkline_width=sparkline_width,
                )
            )
        )
        elements.append(Text(""))  # Blank line after header

        # Build metric rows using best values
        loss_row = self._build_metric_row(
            label="Loss",
            value=f"{best['loss']:.4f}",
            history=val_losses,
            metric_type="loss",
            sparkline_width=sparkline_width,
        )
        top1_row = self._build_metric_row(
            label="Top-1 Acc",
            value=f"{100 * best['top1_acc']:.2f}%",
            history=val_top1,
            metric_type="acc",
            sparkline_width=sparkline_width,
        )
        top3_row = self._build_metric_row(
            label="Top-3 Acc",
            value=f"{100 * best['top3_acc']:.2f}%",
            history=val_top3,
            metric_type="acc",
            sparkline_width=sparkline_width,
        )
        top5_row = self._build_metric_row(
            label="Top-5 Acc",
            value=f"{100 * best['top5_acc']:.2f}%",
            history=val_top5,
            metric_type="acc",
            sparkline_width=sparkline_width,
        )

        # Add metric rows with blank lines between them (centered)
        elements.append(Align.center(loss_row))
        elements.append(Text(""))  # Blank line
        elements.append(Align.center(top1_row))
        elements.append(Text(""))  # Blank line
        elements.append(Align.center(top3_row))
        elements.append(Text(""))  # Blank line
        elements.append(Align.center(top5_row))

        return Panel(
            Group(*elements),
            title="[bold]Validation Metrics[/]",
            border_style=VAL_BORDER_COLOR,
            expand=True,
        )

    def start_epoch(self, epoch: int, steps: int, phase: str = "Train") -> None:
        """Start a new epoch.

        Args:
            epoch: Epoch number (1-indexed).
            steps: Total number of steps in this epoch.
            phase: Phase name ("Train" or "Val").
        """
        self.current_epoch = epoch
        self.current_phase = phase
        self.current_step = 0

        # Reset sparkline history at start of each epoch
        if phase == "Train":
            self.loss_history.clear()
            self.top1_history.clear()
            self.top3_history.clear()
            self.top5_history.clear()

        # Create epoch progress bar
        self.epoch_task = self.progress.add_task(
            "This Epoch",
            total=steps,
        )

    def update_step(
        self,
        loss: float = 0.0,
        top1_acc: float = 0.0,
        top3_acc: float = 0.0,
        top5_acc: float = 0.0,
        lr: float = None,
        data_time: float = None,
        step_time: float = None,
        advance: int = 1,
    ) -> None:
        """Update metrics for the current step.

        Args:
            loss: Current loss value.
            top1_acc: Top-1 accuracy (0.0 to 1.0).
            top3_acc: Top-3 accuracy (0.0 to 1.0).
            top5_acc: Top-5 accuracy (0.0 to 1.0).
            lr: Current learning rate. If None, preserves existing value.
            data_time: Data loading time in seconds. If None, preserves existing.
            step_time: Step time in seconds. If None, preserves existing value.
            advance: Number of steps to advance the progress bar.
        """
        # Only update metrics if in Training phase
        if self.current_phase == "Train":
            self.loss = loss
            self.top1_acc = top1_acc
            self.top3_acc = top3_acc
            self.top5_acc = top5_acc

            # Only update LR and timing if provided (preserves training values during validation)
            if lr is not None:
                self.current_lr = lr
            if data_time is not None:
                self.data_time = data_time
            if step_time is not None:
                self.step_time = step_time

            # Update history for sparklines
            self.loss_history.append(loss)
            self.top1_history.append(top1_acc)
            self.top3_history.append(top3_acc)
            self.top5_history.append(top5_acc)

        self.current_step += advance

        # Update progress bar
        if self.epoch_task is not None:
            self.progress.update(self.epoch_task, advance=advance)

        # Refresh display
        if self.live:
            self.live.update(self._build_display())

    def log(self, message: str) -> None:
        """Add a message to the log window.

        Args:
            message: Log message to display.
        """
        self.log_lines.append(message)
        if self.live:
            self.live.update(self._build_display())

    def record_validation(
        self,
        epoch: int,
        loss: float,
        top1_acc: float,
        top3_acc: float,
        top5_acc: float,
    ) -> None:
        """Record validation metrics for an epoch.

        Args:
            epoch: Epoch number (1-indexed).
            loss: Validation loss.
            top1_acc: Top-1 accuracy (0.0 to 1.0).
            top3_acc: Top-3 accuracy (0.0 to 1.0).
            top5_acc: Top-5 accuracy (0.0 to 1.0).
        """
        metrics = {
            "epoch": epoch,
            "loss": loss,
            "top1_acc": top1_acc,
            "top3_acc": top3_acc,
            "top5_acc": top5_acc,
        }
        self.val_history.append(metrics)

        # Track best epoch (based on highest top-1 accuracy)
        if (
            self.best_val_metrics is None
            or top1_acc > self.best_val_metrics["top1_acc"]
        ):
            self.best_val_epoch = epoch
            self.best_val_metrics = metrics

        if self.live:
            self.live.update(self._build_display())

    def end_epoch(self) -> None:
        """End the current epoch and update overall progress."""
        # Remove epoch task
        if self.epoch_task is not None:
            self.progress.remove_task(self.epoch_task)
            self.epoch_task = None

        # Update overall progress (only after training phase, not validation)
        if self.current_phase == "Val" and self.overall_task is not None:
            self.progress.update(self.overall_task, advance=1)

        if self.live:
            self.live.update(self._build_display())
