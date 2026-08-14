from __future__ import annotations

import glob
import re
import warnings
from pathlib import Path
from typing import Any, Iterable

from .base import FrameInfo, FrameReader
from .exceptions import InvalidFrameError, UnsupportedFormatError
from .exr import ExrReader
from .factory import register_reader

_DIGIT_RUN = re.compile(r"(\d+)")


class ImageSequenceReader(FrameReader):
    """Lazy reader for ordered image sequences."""

    def __init__(
        self,
        source: str | Path | Iterable[str | Path],
        *,
        strict_gaps: bool = False,
        warn_on_gaps: bool = True,
    ):
        self.paths = _resolve_paths(source)
        if not self.paths:
            raise InvalidFrameError(f"No image sequence frames matched: {source}")

        self._frame_readers = [_open_frame_reader(path) for path in self.paths]
        first = self._frame_readers[0]
        self._width = first.width
        self._height = first.height
        self._dtype = first.dtype
        self._channels = first.channels
        self._source_numbers = [_source_number(path) for path in self.paths]
        self._gaps = _detect_gaps(self.paths)

        self._validate_consistent_frames()
        if self._gaps:
            message = _gap_message(self._gaps)
            if strict_gaps:
                raise InvalidFrameError(message)
            if warn_on_gaps:
                warnings.warn(message, stacklevel=2)

    @property
    def width(self) -> int:
        return self._width

    @property
    def height(self) -> int:
        return self._height

    @property
    def frame_count(self) -> int:
        return len(self.paths)

    @property
    def dtype(self):
        return self._dtype

    @property
    def channels(self) -> int:
        return self._channels

    @property
    def supports_random_access(self) -> bool:
        return True

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "format": "image_sequence",
            "frame_paths": [str(path) for path in self.paths],
            "gaps": list(self._gaps),
        }

    def __getitem__(self, index: int):
        self._validate_index(index)
        return self._frame_readers[index][0]

    def frame_info(self, index: int) -> FrameInfo:
        self._validate_index(index)
        return FrameInfo(
            index=index,
            source_path=self.paths[index],
            source_number=self._source_numbers[index],
        )

    def close(self) -> None:
        for reader in self._frame_readers:
            reader.close()

    def _validate_consistent_frames(self) -> None:
        for path, reader in zip(self.paths[1:], self._frame_readers[1:]):
            if reader.width != self.width or reader.height != self.height:
                raise InvalidFrameError(
                    f"Inconsistent image dimensions in sequence: {path} has "
                    f"{reader.width}x{reader.height}, expected {self.width}x{self.height}"
                )
            if reader.dtype != self.dtype:
                raise InvalidFrameError(
                    f"Inconsistent image dtype in sequence: {path} has {reader.dtype}, expected {self.dtype}"
                )
            if reader.channels != self.channels:
                raise InvalidFrameError(
                    f"Inconsistent channel count in sequence: {path} has {reader.channels}, expected {self.channels}"
                )


def _resolve_paths(source: str | Path | Iterable[str | Path]) -> list[Path]:
    if isinstance(source, str | Path):
        text = str(source)
        path = Path(source)
        if path.is_dir():
            paths = list(path.glob("*.exr"))
        elif "#" in text:
            paths = [Path(item) for item in glob.glob(_hash_pattern_to_glob(text))]
        elif glob.has_magic(text):
            paths = [Path(item) for item in glob.glob(text)]
        else:
            paths = [path]
    else:
        paths = [Path(item) for item in source]
    return sorted(paths, key=_numeric_sort_key)


def _hash_pattern_to_glob(pattern: str) -> str:
    return re.sub(r"#+", lambda match: "[0-9]" * len(match.group(0)), pattern)


def _open_frame_reader(path: Path) -> FrameReader:
    if path.suffix.lower() == ".exr":
        return ExrReader(path)
    raise UnsupportedFormatError(f"Unsupported image sequence frame format: {path.suffix}")


def _numeric_sort_key(path: str | Path) -> tuple[str, int, str]:
    name = Path(path).name
    numbers = _DIGIT_RUN.findall(name)
    if not numbers:
        return (name, -1, name)
    return (name[: name.find(numbers[-1])], int(numbers[-1]), name)


def _source_number(path: Path) -> int | None:
    parts = _numbered_name_parts(path)
    if parts is None:
        return None
    return parts[1]


def _detect_gaps(paths: list[Path]) -> tuple[tuple[int, int], ...]:
    groups: dict[tuple[str, str], list[int]] = {}
    for path in paths:
        parts = _numbered_name_parts(path)
        if parts is None:
            continue
        prefix, number, suffix = parts
        groups.setdefault((prefix, suffix), []).append(number)

    gaps: list[tuple[int, int]] = []
    for numbers in groups.values():
        unique_numbers = sorted(set(numbers))
        for previous, current in zip(unique_numbers, unique_numbers[1:]):
            if current > previous + 1:
                gaps.append((previous + 1, current - 1))
    return tuple(gaps)


def _numbered_name_parts(path: Path) -> tuple[str, int, str] | None:
    matches = list(_DIGIT_RUN.finditer(path.name))
    if not matches:
        return None
    match = matches[-1]
    return path.name[: match.start()], int(match.group(0)), path.name[match.end() :]


def _gap_message(gaps: tuple[tuple[int, int], ...]) -> str:
    ranges = []
    for start, end in gaps:
        ranges.append(str(start) if start == end else f"{start}-{end}")
    return f"Image sequence has numbering gaps: {', '.join(ranges)}"


register_reader("exr", ImageSequenceReader, extensions=[".exr"])
register_reader("exr-sequence", ImageSequenceReader)
