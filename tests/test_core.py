from __future__ import annotations

import numpy as np
import pytest

from astroio import (
    FrameInfo,
    FrameReader,
    RandomAccessUnsupportedError,
    UnsupportedFormatError,
    open_reader,
    register_reader,
)


class DummyReader(FrameReader):
    def __init__(self, path, *, frame_count: int = 3):
        self.path = path
        self._frame_count = frame_count
        self.closed = False

    @property
    def width(self) -> int:
        return 2

    @property
    def height(self) -> int:
        return 2

    @property
    def frame_count(self) -> int:
        return self._frame_count

    @property
    def dtype(self):
        return np.uint16

    @property
    def channels(self) -> int:
        return 1

    @property
    def supports_random_access(self) -> bool:
        return True

    def __getitem__(self, index: int):
        self._validate_index(index)
        return np.full((self.height, self.width), index, dtype=self.dtype)

    def frame_info(self, index: int) -> FrameInfo:
        self._validate_index(index)
        return FrameInfo(index=index, source_number=index + 10)

    def close(self) -> None:
        self.closed = True


class SequentialReader(DummyReader):
    @property
    def supports_random_access(self) -> bool:
        return False

    def __getitem__(self, index: int):
        raise RandomAccessUnsupportedError("sequential only")


def test_frame_reader_default_iteration_uses_exact_random_access():
    reader = DummyReader("dummy")

    frames = list(reader)

    assert [int(frame[0, 0]) for frame in frames] == [0, 1, 2]


def test_frame_info_is_side_channel_metadata():
    reader = DummyReader("dummy")

    info = reader.frame_info(2)

    assert info.index == 2
    assert info.source_number == 12


def test_frame_reader_context_manager_closes_reader():
    reader = DummyReader("dummy")

    with reader as opened:
        assert opened is reader

    assert reader.closed


def test_default_iteration_rejects_non_random_access_reader_without_override():
    reader = SequentialReader("dummy")

    with pytest.raises(RandomAccessUnsupportedError):
        list(reader)


def test_factory_opens_registered_reader_by_extension():
    register_reader("dummy", DummyReader, extensions=[".dum"])

    reader = open_reader("capture.dum", frame_count=1)

    assert isinstance(reader, DummyReader)
    assert len(reader) == 1


def test_factory_raises_for_unknown_extension():
    with pytest.raises(UnsupportedFormatError):
        open_reader("capture.unknown")
