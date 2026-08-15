from __future__ import annotations

import numpy as np
import av
import pytest

from astroio import RandomAccessUnsupportedError, UnsupportedPixelFormatError, VideoReader, open_reader
from astroio.video import _debayer_mosaic, _resolve_debayer_pattern, _resolve_output_format, _source_component_bits


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


def test_video_reader_supports_explicit_debayer(tmp_path):
    path = tmp_path / "clip.mp4"
    write_test_mp4(path)

    reader = open_reader(path, debayer="GRBG")
    frame = next(iter(reader))

    assert reader.channels == 3
    assert reader.metadata["requested_debayer"] == "GRBG"
    assert reader.metadata["debayer_pattern"] == "GRBG"
    assert frame.shape == (3, 4, 3)


def test_auto_debayer_uses_bayer_source_metadata():
    assert _resolve_debayer_pattern("auto", "bayer_grbg8") == "GRBG"
    with pytest.raises(UnsupportedPixelFormatError):
        _resolve_debayer_pattern("auto", "pal8")


def test_auto_output_format_preserves_high_bit_depth_sources():
    assert _resolve_output_format("auto", "yuv420p") == "rgb24"
    assert _resolve_output_format("auto", "yuv420p10le") == "rgb48le"
    assert _resolve_output_format("auto", "gray16le") == "rgb48le"
    assert _source_component_bits("bayer_grbg16le") == 16


def test_debayer_mosaic_returns_rgb_channels():
    mosaic = np.zeros((4, 4), dtype=np.uint8)
    mosaic[0::2, 0::2] = 100
    mosaic[0::2, 1::2] = 50
    mosaic[1::2, 0::2] = 50
    mosaic[1::2, 1::2] = 10

    frame = _debayer_mosaic(mosaic, "RGGB")

    assert frame.shape == (4, 4, 3)
    assert frame.dtype == np.uint8
    assert frame[2, 2].tolist() == [100, 50, 10]
