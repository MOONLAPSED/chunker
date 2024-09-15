import asyncio
import inspect
import json
import logging
import os
import pathlib
import struct
import sys
import threading
import time
import uuid
import argparse
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum, auto
from functools import wraps
from queue import Queue, Empty
from typing import Any, Dict, List, Optional, Union, Callable, TypeVar, Tuple, Generic, Set, Coroutine, Type, ClassVar, Protocol
import ctypes
# platforms: Ubuntu-22.04LTS, Windows-11 modern as of writing
if os.name == 'posix':
    from ctypes import cdll
    sys.platform = 'linux'
else:
    from ctypes import windll
    sys.platform = 'windows'

class CustomFormatter(logging.Formatter):
    grey = "\x1b[38;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    green = "\x1b[32;20m"
    reset = "\x1b[0m"

    format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)"

    FORMATS = {
        logging.DEBUG: grey + format + reset,
        logging.INFO: green + format + reset,
        logging.WARNING: yellow + format + reset,
        logging.ERROR: red + format + reset,
        logging.CRITICAL: bold_red + format + reset
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno, self.format)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)

def setup_logger(name: str, level: int, datefmt: str, handlers: list):
    """
    Setup logger with custom formatter.
    :param name: logger name
    :param level: logging level
    :param datefmt: date format
    :param handlers: list of logging handlers
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if logger.hasHandlers():
        logger.handlers.clear()

    for handler in handlers:
        if not isinstance(handler, logging.Handler):
            raise ValueError(f"Invalid handler provided: {handler}")
        handler.setLevel(level)
        handler.setFormatter(CustomFormatter())
        logger.addHandler(handler)

    return logger

def parse_args():
    parser = argparse.ArgumentParser(description="Logger Configuration")
    parser.add_argument('--log-level', type=str, default='DEBUG', choices=logging._nameToLevel.keys(), help='Set logging level')
    parser.add_argument('--log-file', type=str, help='Set log file path')
    parser.add_argument('--log-format', type=str, default='%(asctime)s - %(name)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)', help='Set log format')
    parser.add_argument('--log-datefmt', type=str, default='%Y-%m-%d %H:%M:%S', help='Set date format')
    parser.add_argument('--log-name', type=str, default=__name__, help='Set logger name')
    return parser.parse_args()

def main():
    args = parse_args()
    log_level = logging._nameToLevel.get(args.log_level.upper(), logging.DEBUG)

    handlers = [logging.FileHandler(args.log_file)] if args.log_file else [logging.StreamHandler()]

    logger = setup_logger(name=args.log_name, level=log_level, datefmt=args.log_datefmt, handlers=handlers)
    logger.info("Logger setup complete.")

if __name__ == "__main__":
    main()

Logger = setup_logger("ApplicationBus", logging.DEBUG, "%Y-%m-%d %H:%M:%S", [logging.StreamHandler()])