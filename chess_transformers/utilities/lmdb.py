"""High-level LMDB helper for the *chess-transformers* project.

This module surfaces a single public class ``ChessLMDB`` that turns a raw
``lmdb.Environment`` into an ergonomic, opinionated dataset backend.

Key Features:
    - Sensible project-wide defaults when calling ``lmdb.open`` (512 readers,
      single-file env, 10-GiB default map size for writers).
    - JSON metadata embedded inside the LMDB under a reserved key – keeps the
      dataset self-contained and eliminates stray side-car files.
    - Cached read-only transaction that delivers zero-copy ``memoryview``s and
      plays nicely with multi-process ``torch.utils.data.DataLoader`` workers.
    - Pythonic convenience: context-manager behaviour, ``__len__``,
      ``__getitem__``, and ``__setitem__``.

Fork-Safety for PyTorch DataLoaders:
    When using ``num_workers > 0`` in a PyTorch DataLoader, this class should
    be instantiated inside the worker process (e.g. lazily on first access).
    Do NOT open an LMDB environment in the main process before forking –
    LMDB file handles are not fork-safe.

    Recommended pattern::

        class ChessDataset(torch.utils.data.Dataset):
            def __init__(self, lmdb_path):
                self.lmdb_path = lmdb_path
                self._db = None  # opened lazily

            def _open_db(self):
                if self._db is None:
                    self._db = ChessLMDB(self.lmdb_path, readonly=True)

            def __getitem__(self, idx):
                self._open_db()
                return self._db[idx]
"""

import json
import lmdb

from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, Optional, Union

from chess_transformers.utilities.loggers import setup_logger

# Logger
logger = setup_logger(__file__)

# Reserved key used to store dataset metadata inside the LMDB
_META_KEY: bytes = b"__meta__"


class ChessLMDB:
    """Object-oriented helper around a single-file LMDB dataset.

    The wrapper hides the raw ``lmdb.Environment`` behind a small API tailored
    for the chess-transformers training pipeline. It provides:

    - Parameterised opening of the database with project-wide defaults.
    - Metadata persisted inside the DB under a reserved key.
    - A cached read transaction so each DataLoader worker only opens once.
    - Convenient batch writing with periodic commits.

    Args:
        path (str | Path): Filesystem path to the LMDB file. The file is
            created if it doesn't exist and ``readonly`` is ``False``.
        readonly (bool): If ``True`` (default), open the environment read-only.
            This is the common case for training and evaluation. In this mode,
            locks are disabled, readahead is enabled, and ``map_size`` is
            ignored by LMDB.
        lock (bool): If ``True``, acquire the LMDB write lock. Should remain
            ``False`` inside multi-process dataloader workers. Default is
            ``False``.
        map_size (int | None): Maximum database size in bytes. Only used when
            ``readonly`` is ``False``. Defaults to 10 GiB if not specified.
        **kwargs: Additional keyword arguments forwarded verbatim to
            ``lmdb.open()``. Useful for advanced options like ``max_dbs``.

    Attributes:
        path (Path): The resolved filesystem path to the LMDB file.
        readonly (bool): Whether the environment was opened read-only.
        env (lmdb.Environment): The underlying LMDB environment object.

    Example:
        Opening for reading::

            db = ChessLMDB("/path/to/data.lmdb")
            print(len(db))  # dataset size
            record = db[0]  # read first record

        Opening for writing::

            db = ChessLMDB("/path/to/data.lmdb", readonly=False)
            with db.write() as write:
                for i, data in enumerate(records):
                    write(i, data)
            db.write_metadata({"length": len(records)})
            db.close()
    """

    __slots__ = (
        "env",
        "path",
        "readonly",
        "_open_kwargs",
        "_read_txn",
        "_meta_cache",
        "_length",
        "_closed",
    )

    @staticmethod
    def _encode_key(index: int) -> bytes:
        """Encode integer index to 9-digit zero-padded ASCII key.

        Args:
            index (int): Zero-based row index.

        Returns:
            bytes: A 9-byte ASCII key like b'000000042'.
        """
        return f"{index:09d}".encode("ascii")

    def __init__(
        self,
        path: Union[str, Path],
        readonly: bool = True,
        lock: bool = False,
        map_size: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize and open the LMDB environment.

        Args:
            path (str | Path): Filesystem path to the LMDB file. The file is
                created if it doesn't exist and ``readonly`` is ``False``.
            readonly (bool): If ``True`` (default), open the environment
                read-only. In this mode, locks are disabled and ``map_size``
                is ignored.
            lock (bool): If ``True``, acquire the LMDB write lock. Default is
                ``False``.
            map_size (int | None): Maximum database size in bytes. Only used
                when ``readonly`` is ``False``. Defaults to 10 GiB.
            **kwargs: Additional keyword arguments forwarded to ``lmdb.open()``.
        """
        self.path = Path(path).expanduser().resolve()
        self.readonly = readonly
        self._open_kwargs = dict(lock=lock, map_size=map_size, **kwargs)

        # Initialize caches and state
        self._read_txn = None
        self._meta_cache = None
        self._length = None
        self._closed = False

        # Open environment
        open_args = self._build_open_args(readonly, lock, map_size, **kwargs)
        self.env = lmdb.open(str(self.path), **open_args)
        logger.info(f"Opened LMDB at {self.path} (readonly={readonly}, lock={lock})")

        # Load metadata if readonly (and it exists)
        if readonly:
            try:
                self._meta_cache = self.read_metadata()
            except KeyError:
                logger.error(f"No metadata found during init for {self.path}")

    # -------------------------------------------------------------------------
    # Lifecycle
    # -------------------------------------------------------------------------

    def _build_open_args(
        self,
        readonly: bool,
        lock: bool,
        map_size: Optional[int],
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Build lmdb.open() arguments with project defaults.

        Args:
            readonly (bool): Whether to open the environment read-only.
            lock (bool): Whether to acquire the LMDB write lock.
            map_size (int | None): Maximum database size in bytes.
            **kwargs: Additional keyword arguments for lmdb.open().

        Returns:
            dict: Keyword arguments ready to pass to lmdb.open().
        """
        effective_map_size = None if readonly else (map_size or 10 * 1024**3)
        args: Dict[str, Any] = dict(
            subdir=False,  # Single-file mode
            readonly=readonly,
            lock=lock,
            readahead=not readonly,
            max_readers=512,
            **kwargs,
        )
        if effective_map_size is not None:
            args["map_size"] = effective_map_size
        return args

    @property
    def closed(self) -> bool:
        """Check whether the environment is closed.

        Returns:
            bool: ``True`` if the environment has been closed, ``False``
                otherwise.
        """
        return self._closed

    def close(self) -> None:
        """Close the environment and abort any outstanding read transaction.

        Safe to call multiple times. After closing, most operations will raise
        ``lmdb.Error``.
        """
        if self._closed:
            return
        if self._read_txn is not None:
            try:
                self._read_txn.abort()
            except lmdb.Error:
                logger.error(f"Failed to abort read transaction for {self.path}")
            self._read_txn = None
        try:
            self.env.close()
        except lmdb.Error:
            logger.error(f"Failed to close environment for {self.path}")
        self._closed = True

    def reopen(self, readonly: bool = True, **kwargs: Any) -> "ChessLMDB":
        """Close and reopen the environment, possibly with a different mode.

        Useful for switching from write mode to read mode after populating
        the database.

        Args:
            readonly (bool): If ``True`` (default), reopen the environment
                read-only. If ``False``, reopen for writing.
            **kwargs: Additional keyword arguments passed to ``lmdb.open()``.
                These are merged with the original open kwargs, with these
                taking precedence.

        Returns:
            ChessLMDB: Returns ``self`` for method chaining.

        Example:
            >>> db = ChessLMDB(path, readonly=False)
            >>> # ... write data ...
            >>> db.reopen(readonly=True)  # switch to read mode
        """
        self.close()

        merged_kwargs = {**self._open_kwargs, **kwargs}
        lock = merged_kwargs.pop("lock", False)
        map_size = merged_kwargs.pop("map_size", None)

        self.readonly = readonly
        self._read_txn = None
        self._meta_cache = None
        self._length = None
        self._closed = False

        open_args = self._build_open_args(readonly, lock, map_size, **merged_kwargs)
        self.env = lmdb.open(str(self.path), **open_args)
        logger.info(f"Reopened LMDB at {self.path} (readonly={readonly}, lock={lock})")

        if readonly:
            try:
                self._meta_cache = self.read_metadata()
                self._length = self._meta_cache.get("length")
            except KeyError:
                logger.error(f"No metadata found during reopen for {self.path}")

        return self

    def __enter__(self) -> "ChessLMDB":
        """Enter the context manager.

        Returns:
            ChessLMDB: Returns ``self``.
        """
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        """Exit the context manager, closing the environment.

        Args:
            exc_type: Exception type, if an exception was raised.
            exc: Exception instance, if an exception was raised.
            tb: Traceback, if an exception was raised.
        """
        self.close()

    def __del__(self) -> None:
        """Destructor that attempts to close resources quietly."""
        from contextlib import suppress

        with suppress(Exception):
            self.close()

    def __repr__(self) -> str:
        """Return a string representation of this instance.

        Returns:
            str: A string like ``ChessLMDB(/path/to/db, readonly, len=12345)``.
        """
        status = (
            "closed" if self.closed else ("readonly" if self.readonly else "writable")
        )
        length_str = f", len={self._length}" if self._length is not None else ""
        return f"ChessLMDB({self.path!s}, {status}{length_str})"

    # -------------------------------------------------------------------------
    # Metadata
    # -------------------------------------------------------------------------

    def write_metadata(self, metadata: Dict[str, Any]) -> None:
        """Write dataset metadata into the LMDB under a reserved key.

        The metadata is stored as JSON. This method should be called after
        all records have been written to store the final dataset length
        and any other relevant information.

        Args:
            metadata (dict): A JSON-serialisable dictionary. Should contain
                at minimum a ``"length"`` key with the total number of
                records. Additional keys like ``"val_split_index"`` or
                ``"schema"`` are encouraged.

        Raises:
            lmdb.ReadonlyError: If the environment is opened read-only and
                cannot be reopened.

        Note:
            If the environment was closed (e.g. after a write session), this
            method will automatically reopen it in write mode, write the
            metadata, and close it again.

        Example:
            >>> db.write_metadata({
            ...     "length": 1000000,
            ...     "val_split_index": 900000,
            ...     "schema": "board(64B), turn(1B), from(1B), to(1B)"
            ... })
        """
        reopened = False
        if self.closed or self.readonly:
            self.reopen(readonly=False)
            reopened = True

        payload = json.dumps(metadata).encode("utf-8")
        with self.env.begin(write=True) as txn:
            txn.put(_META_KEY, payload, overwrite=True)
        self._meta_cache = metadata
        self._length = metadata.get("length")
        logger.info(f"Stored metadata ({len(payload)} bytes)")

        if reopened:
            self.close()

    def read_metadata(self) -> Dict[str, Any]:
        """Read and cache the metadata dictionary from the LMDB.

        The result is cached after the first read, so subsequent calls are
        instant.

        Returns:
            dict: The stored metadata dictionary.

        Raises:
            KeyError: If no metadata has been stored yet (i.e. the database
                is newly created or ``write_metadata`` was never called).

        Example:
            >>> meta = db.read_metadata()
            >>> print(meta["length"])
            1000000
        """
        if self._meta_cache is not None:
            return self._meta_cache

        with self.env.begin(buffers=True) as txn:
            raw = txn.get(_META_KEY)
        if raw is None:
            logger.critical(f"No metadata stored in LMDB at {self.path}")
            raise KeyError("No metadata stored in this LMDB")

        self._meta_cache = json.loads(bytes(raw))
        self._length = self._meta_cache.get("length")
        return self._meta_cache

    # -------------------------------------------------------------------------
    # Writing
    # -------------------------------------------------------------------------

    def __setitem__(self, index: int, data: bytes) -> None:
        """Write a single record at the given index.

        Each call opens a new transaction and commits immediately. For bulk
        writes, use the :meth:`write` context manager instead.

        Args:
            index (int): Zero-based row index. Will be encoded as a
                zero-padded 9-digit ASCII key.
            data (bytes): The packed binary data to store.

        Raises:
            lmdb.ReadonlyError: If the environment is opened read-only.

        Example:
            >>> db[0] = packed_record
            >>> db[1] = another_record
        """
        if self.readonly:
            logger.critical(f"Attempted write to read-only database at {self.path}")
            raise lmdb.ReadonlyError("Cannot write to read-only database")
        with self.env.begin(write=True) as txn:
            txn.put(self._encode_key(index), data)

    @contextmanager
    def write(
        self, commit_every: int = 10000
    ) -> Iterator[Callable[[int, bytes], None]]:
        """Context manager for efficient batch writing.

        Opens a write transaction and yields a ``write(index, data)`` function.
        The transaction is automatically committed when the context exits, or
        periodically after ``commit_every`` writes to avoid unbounded memory
        growth.

        Args:
            commit_every (int): Number of writes after which to commit the
                current transaction and start a new one. This prevents the
                transaction from growing too large in memory. Default is
                ``10000``. Set to ``0`` to disable periodic commits (not
                recommended for large datasets).

        Yields:
            Callable[[int, bytes], None]: A function ``write(index, data)``
                that writes a record to the database. ``index`` is the
                zero-based row index, ``data`` is the packed binary record.

        Raises:
            lmdb.ReadonlyError: If the environment is opened read-only.

        Note:
            If an exception occurs inside the context, the current transaction
            is aborted (rolled back) and the exception is re-raised.

        Example:
            >>> with db.write(commit_every=5000) as write:
            ...     for i, record in enumerate(records):
            ...         packed = struct.pack("64B B B", *record)
            ...         write(i, packed)
            >>> db.write_metadata({"length": len(records)})
        """
        if self.readonly:
            logger.critical(
                f"Attempted batch write to read-only database at {self.path}"
            )
            raise lmdb.ReadonlyError("Cannot write to read-only database")

        state = {"txn": self.env.begin(write=True), "count": 0, "total": 0}
        encode_key = self._encode_key  # Capture for closure

        def _write(index: int, data: bytes) -> None:
            """Write a single record to the current transaction.

            This closure captures ``state`` and ``commit_every`` from the
            enclosing scope. It handles periodic commits automatically.

            Args:
                index (int): Zero-based row index.
                data (bytes): The packed binary record to store.
            """
            state["txn"].put(encode_key(index), data)
            state["count"] += 1
            state["total"] += 1

            if commit_every > 0 and state["count"] >= commit_every:
                state["txn"].commit()
                state["txn"] = self.env.begin(write=True)
                state["count"] = 0

        try:
            yield _write
            state["txn"].commit()
            logger.info(f"Final commit: {state['total']} total records written")
        except Exception:
            logger.critical(
                f"Exception during batch write at {self.path}, aborting transaction"
            )
            state["txn"].abort()
            raise

    # -------------------------------------------------------------------------
    # Reading
    # -------------------------------------------------------------------------

    def __len__(self) -> int:
        """Return the number of records in the dataset.

        The length is read from the stored metadata. If metadata hasn't been
        loaded yet, this triggers a :meth:`read_metadata` call.

        Returns:
            int: The total number of records as stored in metadata.

        Raises:
            KeyError: If no metadata has been stored (via :meth:`write_metadata`).

        Example:
            >>> print(len(db))
            1000000
        """
        if self._length is None:
            self._length = self.read_metadata()["length"]
        return self._length

    def __getitem__(self, index: int) -> memoryview:
        """Read a record by its index.

        Returns a zero-copy ``memoryview`` onto the stored bytes. The view
        remains valid as long as the read transaction is open (i.e. until
        the database is closed).

        Args:
            index (int): Zero-based row index.

        Returns:
            memoryview: A read-only view onto the packed record bytes. Can be
                passed directly to ``struct.unpack()`` or ``torch.frombuffer()``.

        Raises:
            IndexError: If ``index`` is not a valid key in the database.

        Example:
            >>> data = db[42]
            >>> board, turn, from_sq, to_sq = struct.unpack("64B B B B", data)
        """
        if self._read_txn is None:
            self._read_txn = self.env.begin(buffers=True)
        res = self._read_txn.get(self._encode_key(index))
        if res is None:
            logger.critical(f"Index {index} not found in LMDB at {self.path}")
            raise IndexError(index)
        return res
