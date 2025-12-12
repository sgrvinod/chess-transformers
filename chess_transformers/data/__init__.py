"""Data processing utilities for chess-transformers.

This package provides tools for processing chess game data into LMDB format
suitable for training transformer models.

Key Features:
    - process: Main data processing pipeline for converting .moves/.fens to LMDB.
    - levels: Mappings for chess board representation (pieces, squares, moves).
    - configs: Dataset configuration classes and instances.

Notes:
    Set the CT_DATA_FOLDER environment variable before running the data
    processing scripts to specify the root folder containing dataset subfolders.
"""
