"""Utility helpers for setting up consistent colored logging.

This module provides logging configuration utilities for the chess-transformers
project, with colorized console output and contextual information.

Key Features:
    - ContextFilter: Enriches log records with short_name and funcName context.
    - setup_logger: Convenience wrapper returning a configured colored logger.

Notes:
    All modules in the project should use ``setup_logger(__file__)`` to get
    a consistently configured logger instance with colored output.
"""

import os
import logging
import colorlog


class ContextFilter(logging.Filter):
    """Inject additional contextual information into :class:`logging.LogRecord`.

    The filter computes a human-friendly ``short_name`` for each record based
    on the logger name or originating file and replaces the default
    ``funcName`` of ``<module>`` with ``root`` so that top-level log calls are
    labelled consistently.
    """

    def filter(self, record):  # noqa: A003
        """Populate the given :class:`logging.LogRecord` with extra attributes.

        Args:
            record: The record about to be emitted.

        Returns:
            Always ``True`` so that the record is not filtered out and will
            continue through the logging pipeline.
        """
        if (
            record.name.endswith(".py")
            or os.sep in record.name
            or (os.altsep and os.altsep in record.name)
        ):
            filename = os.path.basename(record.name)
            record.short_name = os.path.splitext(filename)[0]
        else:
            parts = record.name.split(".")
            if len(parts) >= 2:
                record.short_name = f"{parts[-2]}.{parts[-1]}"
            else:
                record.short_name = record.name

        if record.funcName == "<module>":
            record.funcName = "root"

        return True


def setup_logger(name: str = __file__, log_level: int = logging.INFO) -> logging.Logger:
    """Return a colour-enabled :class:`logging.Logger`.

    Args:
        name: The logger name. Defaults to the current module path so that
            the caller gets an independent logger hierarchy unless a custom
            name is given.
        log_level: Logging level threshold, e.g. ``logging.INFO``. The
            default is ``logging.INFO``.

    Returns:
        A configured logger instance with a :class:`logging.StreamHandler`,
        coloured formatter, and the :class:`ContextFilter` attached.
    """
    logger = logging.getLogger(name)
    logger.setLevel(log_level)

    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.addFilter(ContextFilter())
        formatter = colorlog.ColoredFormatter(
            "%(cyan)s%(asctime)s%(reset)s - "
            "%(purple)s%(short_name)s.%(funcName)s%(reset)s - "
            "%(log_color)s%(levelname)s%(reset)s - "
            "%(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            reset=True,
            log_colors={
                "DEBUG": "cyan",
                "INFO": "green",
                "WARNING": "yellow",
                "ERROR": "red",
                "CRITICAL": "white,bg_red",
            },
            secondary_log_colors={},
            style="%",
        )

        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger
