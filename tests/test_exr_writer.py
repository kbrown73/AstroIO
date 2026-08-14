from __future__ import annotations

import numpy as np
import OpenImageIO as oiio
import pytest

from astroio import InvalidFrameError, open_writer
from astroio.exr import ExrWriter, read_exr, write_exr


def exr_pixel_format(path) -> str:
    inp = oiio.ImageInput.open(str(path))
    assert inp is not None
    try:
        return str(inp.spec().format)
    finally:
        inp.close()


def test_write_exr_writes_half_by_default(tmp_path):
    path = tmp_path / "frame.exr"
    image = np.arange(6, dtype=np.float32).reshape(2, 3)

    write_exr(path, image)

    assert exr_pixel_format(path) == "half"
    decoded = read_exr(path)
    assert decoded.shape == (2, 3)
    assert decoded.dtype == np.float16
    np.testing.assert_array_equal(decoded, image.astype(np.float16))


def test_write_exr_can_write_float_pixels(tmp_path):
    path = tmp_path / "frame.exr"
    image = np.ones((2, 3, 3), dtype=np.float32)

    write_exr(path, image, half=False)

    assert exr_pixel_format(path) == "float"
    decoded = read_exr(path)
    assert decoded.shape == (2, 3, 3)
    assert decoded.dtype == np.float32


def test_open_writer_uses_registered_exr_writer(tmp_path):
    path = tmp_path / "frame.exr"
    image = np.ones((2, 3), dtype=np.float32)

    with open_writer(path, width=3, height=2, dtype=np.float32, channels=1) as writer:
        writer.write(image)

    assert exr_pixel_format(path) == "half"


def test_exr_writer_rejects_unexpected_dtype(tmp_path):
    path = tmp_path / "frame.exr"
    writer = ExrWriter(path, width=3, height=2, dtype=np.float32, channels=1)

    with pytest.raises(InvalidFrameError, match="dtype"):
        writer.write(np.ones((2, 3), dtype=np.float16))


def test_exr_writer_rejects_second_frame(tmp_path):
    path = tmp_path / "frame.exr"
    writer = ExrWriter(path, width=3, height=2, dtype=np.float32, channels=1)
    writer.write(np.ones((2, 3), dtype=np.float32))

    with pytest.raises(InvalidFrameError, match="only accepts one frame"):
        writer.write(np.ones((2, 3), dtype=np.float32))
