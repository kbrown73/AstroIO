from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from .base import FrameReader, FrameWriter
from .exceptions import UnsupportedFormatError

ReaderFactory = Callable[..., FrameReader]
WriterFactory = Callable[..., FrameWriter]

_reader_factories: dict[str, ReaderFactory] = {}
_writer_factories: dict[str, WriterFactory] = {}
_extension_formats: dict[str, str] = {}


def register_reader(
    format_name: str,
    factory: ReaderFactory,
    *,
    extensions: list[str] | tuple[str, ...] = (),
) -> None:
    key = _format_key(format_name)
    _reader_factories[key] = factory
    for extension in extensions:
        _extension_formats[_extension_key(extension)] = key


def register_writer(format_name: str, factory: WriterFactory) -> None:
    _writer_factories[_format_key(format_name)] = factory


def open_reader(path, *, format: str | None = None, **kwargs: Any) -> FrameReader:
    key = _resolve_format(path, format)
    try:
        factory = _reader_factories[key]
    except KeyError as exc:
        raise UnsupportedFormatError(f"No AstroIO reader registered for format: {key}") from exc
    return factory(path, **kwargs)


def open_writer(path, *, format: str | None = None, **kwargs: Any) -> FrameWriter:
    key = _resolve_format(path, format)
    try:
        factory = _writer_factories[key]
    except KeyError as exc:
        raise UnsupportedFormatError(f"No AstroIO writer registered for format: {key}") from exc
    return factory(path, **kwargs)


def _resolve_format(path, explicit_format: str | None) -> str:
    if explicit_format is not None:
        return _format_key(explicit_format)

    suffix = Path(path).suffix
    if not suffix:
        raise UnsupportedFormatError(f"Could not infer format from path: {path}")

    extension = _extension_key(suffix)
    try:
        return _extension_formats[extension]
    except KeyError as exc:
        raise UnsupportedFormatError(f"No AstroIO format registered for extension: {suffix}") from exc


def _format_key(format_name: str) -> str:
    return format_name.lower().lstrip(".")


def _extension_key(extension: str) -> str:
    return extension.lower().lstrip(".")
