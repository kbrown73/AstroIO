"""Common frame-oriented IO for astronomy and image-sequence formats."""

from .base import FrameInfo, FrameReader, FrameWriter
from .exceptions import (
    AstroIOError,
    CorruptFileError,
    InvalidFrameError,
    RandomAccessUnsupportedError,
    UnsupportedFormatError,
    UnsupportedPixelFormatError,
)
from .exr import ExrReader, ExrWriter
from .factory import open_reader, open_writer, register_reader, register_writer
from .image_sequence import ImageSequenceReader

__all__ = [
    "AstroIOError",
    "CorruptFileError",
    "ExrReader",
    "ExrWriter",
    "FrameInfo",
    "FrameReader",
    "FrameWriter",
    "ImageSequenceReader",
    "InvalidFrameError",
    "RandomAccessUnsupportedError",
    "UnsupportedFormatError",
    "UnsupportedPixelFormatError",
    "open_reader",
    "open_writer",
    "register_reader",
    "register_writer",
]
