"""PyTorch Dataset classes for chess transformer training.

This module provides dataset implementations that load chess positions and
moves from LMDB databases created by the data processing pipeline. The
datasets are designed for training chess transformers, specifically the
ChessTransformerEncoderFT model that predicts From/To squares.

Key Features:
    - ChessDatasetFT: Dataset for From/To prediction with LMDB backend.
    - Fork-safe lazy initialization for multi-worker DataLoader support.
    - Support for optional legal move masks (for label smoothing losses).
    - Efficient zero-copy reads via memoryview and struct unpacking.
    - NumPy-based output for optimal DataLoader throughput.

Notes:
    The LMDB database must have been created with `chess_transformers.data.process`
    and contain metadata including 'length', 'val_split_index', and optionally
    'legal_mask_mode'. See `chess_transformers.utilities.lmdb.ChessLMDB` for
    the database interface.
"""

import struct
import contextlib

import numpy as np
from pathlib import Path
from torch.utils.data import Dataset
from typing import Any, Dict, Literal, Optional, Union

from chess_transformers.utilities.lmdb import ChessLMDB
from chess_transformers.utilities.loggers import setup_logger

# Logger
logger = setup_logger(__file__)

# Struct formats for unpacking LMDB records
# Base: 64 bytes (board) + 7 bytes (turn, 4x castling, from, to) = 71 bytes
# With masks: + 16 bytes (two 64-bit masks) = 87 bytes
_STRUCT_BASE = struct.Struct("64B B B B B B B B")
_STRUCT_WITH_MASKS = struct.Struct("64B B B B B B B B Q Q")


class ChessDatasetFT(Dataset):
    """PyTorch Dataset for From/To prediction with LMDB backend.

    Loads chess positions and target moves from an LMDB database. Each
    record contains a board position (64 bytes), metadata (turn, castling
    rights), and target squares (from, to). Optionally includes legal move
    masks for label smoothing.

    The dataset uses lazy initialization for fork-safety: the LMDB connection
    is opened on first access, not in __init__. This allows safe use with
    multi-worker DataLoaders where workers are forked from the main process.

    Attributes:
        lmdb_path (Path): Path to the LMDB database file.
        split (str | None): Data split, one of "train", "val", or None.
        has_masks (bool | None): Whether the database contains legal masks.
            Set after first access to the database.

    Example:
        >>> dataset = ChessDatasetFT(
        ...     lmdb_path="/path/to/data.lmdb",
        ...     split="train",
        ... )
        >>> len(dataset)
        1000000
        >>> batch = dataset[0]
        >>> batch["board_positions"].shape
        torch.Size([64])
    """

    def __init__(
        self,
        lmdb_path: Union[str, Path],
        split: Optional[Literal["train", "val"]] = None,
    ) -> None:
        """Initialize the dataset.

        Args:
            lmdb_path: Path to the LMDB database file created by the data
                processing pipeline. Must contain metadata with 'length'
                and optionally 'val_split_index' and 'legal_mask_mode'.
            split: Data split to use. Options:
                - "train": Use indices [0, val_split_index).
                - "val": Use indices [val_split_index, length).
                - None: Use all indices [0, length).

        Note:
            The LMDB connection is not opened here to ensure fork-safety.
            It will be opened lazily on first __getitem__ or __len__ call.
        """
        self.lmdb_path = Path(lmdb_path).expanduser().resolve()
        self.split = split

        # Lazy initialization for fork-safety
        self._db: Optional[ChessLMDB] = None
        self._metadata: Optional[Dict[str, Any]] = None
        self._has_masks: Optional[bool] = None
        self._first_index: Optional[int] = None
        self._length: Optional[int] = None
        self._struct: Optional[struct.Struct] = None

    def _open_db(self) -> None:
        """Open the LMDB database and load metadata.

        This method is called lazily on first access. It opens the LMDB
        connection, reads metadata, and sets up the struct format and
        index ranges based on the split.

        Raises:
            FileNotFoundError: If the LMDB file does not exist.
            KeyError: If required metadata is missing from the database.
            ValueError: If split is not one of "train", "val", or None.
        """
        if self._db is not None:
            return

        if not self.lmdb_path.exists():
            logger.critical(f"LMDB file not found: {self.lmdb_path}")
            raise FileNotFoundError(f"LMDB file not found: {self.lmdb_path}")

        self._db = ChessLMDB(self.lmdb_path, readonly=True)
        self._metadata = self._db.read_metadata()

        # Determine if we have legal masks
        legal_mask_mode = self._metadata.get("legal_mask_mode")
        self._has_masks = legal_mask_mode is not None
        self._struct = _STRUCT_WITH_MASKS if self._has_masks else _STRUCT_BASE

        # Compute index ranges based on split
        total_length = self._metadata["length"]
        val_split_index = self._metadata.get("val_split_index")

        if self.split == "train":
            self._first_index = 0
            if val_split_index is not None:
                self._length = val_split_index
            else:
                self._length = total_length
        elif self.split == "val":
            if val_split_index is None:
                logger.critical("Validation split requested but no split index found")
                raise ValueError(
                    "Validation split requested but 'val_split_index' not "
                    "found in metadata"
                )
            self._first_index = val_split_index
            self._length = total_length - val_split_index
        elif self.split is None:
            self._first_index = 0
            self._length = total_length
        else:
            logger.critical(f"Invalid split: {self.split}")
            raise ValueError(
                f"Split must be 'train', 'val', or None, got: {self.split}"
            )

        logger.info(
            f"Opened dataset at {self.lmdb_path}, split={self.split}, "
            f"length={self._length:,}, has_masks={self._has_masks}"
        )

    @property
    def has_masks(self) -> bool:
        """Check if the dataset contains legal move masks.

        Returns:
            True if the database was created with legal_mask_mode, False otherwise.

        Note:
            This property triggers lazy initialization if not already done.
        """
        if self._has_masks is None:
            self._open_db()
        return self._has_masks

    def __len__(self) -> int:
        """Return the number of samples in this split.

        Returns:
            Number of samples available in the current split.
        """
        if self._length is None:
            self._open_db()
        return self._length

    def __getitem__(self, index: int) -> Dict[str, np.ndarray]:
        """Get a single training sample.

        Args:
            index: Sample index within this split (0 to len-1).

        Returns:
            Dictionary containing:
                - turns: Turn indicator, shape (1,)
                - white_kingside_castling_rights: shape (1,)
                - white_queenside_castling_rights: shape (1,)
                - black_kingside_castling_rights: shape (1,)
                - black_queenside_castling_rights: shape (1,)
                - board_positions: Piece encodings for each square, shape (64,)
                - from_squares: Target from-square index, shape (1,)
                - to_squares: Target to-square index, shape (1,)
                - legal_from_mask: (optional) 64-bit mask of legal from-squares
                - legal_to_mask: (optional) 64-bit mask of legal to-squares

        Raises:
            IndexError: If index is out of range for this split.
        """
        if self._db is None:
            self._open_db()

        if index < 0 or index >= self._length:
            raise IndexError(
                f"Index {index} out of range for split '{self.split}' "
                f"with length {self._length}"
            )

        # Read the raw record from LMDB
        global_index = self._first_index + index
        raw = self._db[global_index]

        # Unpack the binary data
        if self._has_masks:
            unpacked = self._struct.unpack(raw)
            # 64 board + turn + 4 castling + from + to + 2 masks
            board = unpacked[:64]
            turn = unpacked[64]
            wk = unpacked[65]
            wq = unpacked[66]
            bk = unpacked[67]
            bq = unpacked[68]
            from_sq = unpacked[69]
            to_sq = unpacked[70]
            legal_from_mask = unpacked[71]
            legal_to_mask = unpacked[72]
        else:
            unpacked = self._struct.unpack(raw)
            board = unpacked[:64]
            turn = unpacked[64]
            wk = unpacked[65]
            wq = unpacked[66]
            bk = unpacked[67]
            bq = unpacked[68]
            from_sq = unpacked[69]
            to_sq = unpacked[70]

        # Build the output dictionary using numpy arrays for optimal DataLoader
        # throughput. PyTorch's default collate_fn efficiently converts these
        # to tensors at batch time (~26% faster than creating tensors here).
        result = {
            "turns": np.array([turn], dtype=np.int32),
            "white_kingside_castling_rights": np.array([wk], dtype=np.int32),
            "white_queenside_castling_rights": np.array([wq], dtype=np.int32),
            "black_kingside_castling_rights": np.array([bk], dtype=np.int32),
            "black_queenside_castling_rights": np.array([bq], dtype=np.int32),
            "board_positions": np.array(board, dtype=np.int32),
            "from_squares": np.array([from_sq], dtype=np.int64),
            "to_squares": np.array([to_sq], dtype=np.int64),
        }

        # Add legal masks if available
        if self._has_masks:
            result["legal_from_mask"] = np.array(legal_from_mask, dtype=np.int64)
            result["legal_to_mask"] = np.array(legal_to_mask, dtype=np.int64)

        return result

    def close(self) -> None:
        """Close the LMDB database connection.

        Safe to call multiple times or if the database was never opened.
        """
        if self._db is not None:
            self._db.close()
            self._db = None

    def __del__(self) -> None:
        """Destructor that cleans up the database connection."""
        with contextlib.suppress(Exception):
            self.close()

    def __repr__(self) -> str:
        """Return a string representation of the dataset.

        Returns:
            String describing the dataset path, split, and length.
        """
        length_str = f", len={self._length}" if self._length is not None else ""
        mask_str = (
            f", has_masks={self._has_masks}" if self._has_masks is not None else ""
        )
        return f"ChessDatasetFT({self.lmdb_path!s}, split={self.split!r}{length_str}{mask_str})"


if __name__ == "__main__":
    # Test the dataset with a config.
    import argparse

    from chess_transformers.utilities.configs import import_config

    parser = argparse.ArgumentParser()
    parser.add_argument("config_name", type=str, help="Name of configuration file.")
    parser.add_argument(
        "--data-folder",
        type=str,
        default=None,
        help="Path to data folder (overrides CT_DATA_FOLDER env var).",
    )
    args = parser.parse_args()

    # Import config and determine data path
    import os

    config = import_config(args.config_name)

    # Get LMDB path from config, expanding environment variables
    if config.dataloader_config is None:
        raise ValueError("config.dataloader_config is required")
    if config.dataloader_config.lmdb_filepath is None:
        raise ValueError("config.dataloader_config.lmdb_filepath is required")

    lmdb_filepath = config.dataloader_config.lmdb_filepath
    if args.data_folder:
        # Override the data folder portion of the path
        lmdb_path = Path(args.data_folder) / Path(lmdb_filepath).name
    else:
        lmdb_path = Path(os.path.expandvars(lmdb_filepath))

    # Test train split
    dataset = ChessDatasetFT(lmdb_path=lmdb_path, split="train")
    logger.info(f"Train dataset: {dataset}")
    logger.info(f"Train length: {len(dataset):,}")
    dataset.close()

    # Test val split
    dataset_val = ChessDatasetFT(lmdb_path=lmdb_path, split="val")
    logger.info(f"Val dataset: {dataset_val}")
    logger.info(f"Val length: {len(dataset_val):,}")
    dataset_val.close()
