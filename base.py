from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .exceptions import RandomAccessUnsupportedError


@dataclass(frozen=True)
class FrameInfo:
    """Side-channel information for a frame returned by a reader."""

    index: int
    source_path: Path | None = None
    source_number: int | None = None
    timestamp: object | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class FrameReader(ABC):
    """Base interface for frame-oriented readers."""

    @property
    @abstractmethod
    def width(self) -> int:
        ...

    @property
    @abstractmethod
    def height(self) -> int:
        ...

    @property
    @abstractmethod
    def frame_count(self) -> int:
        ...

    @property
    @abstractmethod
    def dtype(self):
        ...

    @property
    @abstractmethod
    def channels(self) -> int:
        ...

    @property
    @abstractmethod
    def supports_random_access(self) -> bool:
        ...

    @property
    def fps(self) -> float | None:
        return None

    @property
    def timestamps(self):
        return None

    @property
    def metadata(self) -> dict[str, Any]:
        return {}

    @abstractmethod
    def __getitem__(self, index: int):
        ...

    def frame_info(self, index: int) -> FrameInfo:
        self._validate_index(index)
        return FrameInfo(index=index)

    def __len__(self) -> int:
        return self.frame_count

    def __iter__(self):
        if not self.supports_random_access:
            raise RandomAccessUnsupportedError(
                f"{type(self).__name__} must override __iter__ when random access is unsupported"
            )
        for index in range(self.frame_count):
            yield self[index]

    def close(self) -> None:
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def _validate_index(self, index: int) -> None:
        if index < 0 or index >= self.frame_count:
            raise IndexError(index)


class FrameWriter(ABC):
    """Base interface for frame-oriented writers."""

    @abstractmethod
    def write(self, frame) -> None:
        ...

    def close(self) -> None:
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
