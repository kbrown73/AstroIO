from __future__ import annotations

import numpy as np
import av
import pytest

from astroio import RandomAccessUnsupportedError, VideoReader, open_reader
from astroio.video import _resolve_output_format


def write_test_mp4(path, *, frame_count: int = 3, rate: int = 2) -> None:
    container = av.open(str(path), "w")
    stream = container.add_stream("mpeg4", rate=rate)
    stream.width = 4
    stream.height = 3
    stream.pix_fmt = "yuv420p"
    try:
        for index in range(frame_count):
            image = np.full((3, 4, 3), index * 60, dtype=np.uint8)
            frame = av.VideoFrame.from_ndarray(image, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    finally:
        container.close()


def test_open_reader_decodes_mp4_sequentially(tmp_path):
    path = tmp_path / "clip.mp4"
    write_test_mp4(path)

    reader = open_reader(path)

    assert isinstance(reader, VideoReader)
    assert reader.width == 4
    assert reader.height == 3
    assert reader.frame_count == 3
    assert reader.channels == 3
    assert reader.dtype == np.dtype(np.uint8)
    assert reader.fps == 2.0
    assert not reader.supports_random_access
    assert reader.metadata["codec"] == "mpeg4"
    assert reader.metadata["source_pixel_format"] == "yuv420p"
    assert reader.metadata["output_format"] == "rgb24"

    frames = list(reader)

    assert len(frames) == 3
    assert frames[0].shape == (3, 4, 3)
    assert frames[0].dtype == np.uint8


def test_video_reader_rejects_indexed_access(tmp_path):
    path = tmp_path / "clip.mp4"
    write_test_mp4(path)
    reader = open_reader(path)

    with pytest.raises(RandomAccessUnsupportedError):
        reader[0]


def test_video_reader_frame_info_reports_source_path_and_dense_number(tmp_path):
    path = tmp_path / "clip.mp4"
    write_test_mp4(path)
    reader = open_reader(path)

    info = reader.frame_info(2)

    assert info.index == 2
    assert info.source_path == path
    assert info.source_number == 2


def test_video_reader_supports_explicit_rgb48_output(tmp_path):
    path = tmp_path / "clip.mp4"
    write_test_mp4(path)

    reader = open_reader(path, output_format="rgb48le")
    frame = next(iter(reader))

    assert reader.dtype == np.dtype(np.uint16)
    assert reader.channels == 3
    assert reader.metadata["output_format"] == "rgb48le"
    assert frame.dtype == np.uint16
    assert frame.shape == (3, 4, 3)


def test_auto_output_format_preserves_high_bit_depth_sources():
    assert _resolve_output_format("auto", "yuv420p") == "rgb24"
    assert _resolve_output_format("auto", "yuv420p10le") == "rgb48le"
    assert _resolve_output_format("auto", "gray16le") == "rgb48le"
