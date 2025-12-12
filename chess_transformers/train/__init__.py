"""Training components for chess-transformers.

This package provides training pipelines, datasets, and learning rate
schedules for training chess transformer models.

Key Features:
    - train: Main training script with checkpointing and mixed precision.
    - datasets: PyTorch Dataset classes for loading LMDB chess data.
    - schedules: Learning rate schedule factory functions (warmup + decay).

Notes:
    Training requires data in LMDB format created by the data processing
    pipeline. See `chess_transformers.data.process` for data preparation.
"""
