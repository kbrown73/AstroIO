from __future__ import annotations

import numpy as np
import OpenImageIO as oiio
import pytest

from astroio import ImageSequenceReader, InvalidFrameError, open_reader


def write_exr(path, image: np.ndarray, pixel_type) -> None:
    image = np.asarray(image)
    if image.ndim == 2:
        height, width = image.shape
        channels = 1
        image = image[:, :, np.newaxis]
    else:
        height, width, channels = image.shape
    spec = oiio.ImageSpec(width, height, channels, pixel_type)
    out = oiio.ImageOutput.create(str(path))
    assert out is not None
    try:
        assert out.open(str(path), spec)
        assert out.write_image(image)
    finally:
        out.close()


def test_open_reader_reads_single_exr_as_one_frame_sequence(tmp_path):
    path = tmp_path / "IMG_0001.exr"
    source = np.arange(6, dtype=np.float32).reshape(2, 3)
    write_exr(path, source, oiio.HALF)

    reader = open_reader(path)

    assert reader.width == 3
    assert reader.height == 2
    assert reader.frame_count == 1
    assert reader.channels == 1
    assert reader.dtype == np.dtype(np.float16)
    assert reader.supports_random_access
    assert reader.frame_info(0).source_path == path
    np.testing.assert_array_equal(reader[0], source.astype(np.float16))


def test_image_sequence_sorts_by_source_number_and_warns_on_gaps(tmp_path):
    first = np.full((2, 2, 3), 1.0, dtype=np.float32)
    third = np.full((2, 2, 3), 3.0, dtype=np.float32)
    write_exr(tmp_path / "IMG_0003.exr", third, oiio.FLOAT)
    write_exr(tmp_path / "IMG_0001.exr", first, oiio.FLOAT)

    with pytest.warns(UserWarning, match="numbering gaps: 2"):
        reader = open_reader(tmp_path / "IMG_*.exr")

    assert reader.frame_count == 2
    assert reader.frame_info(0).source_number == 1
    assert reader.frame_info(1).source_number == 3
    np.testing.assert_array_equal(reader[0], first)
    np.testing.assert_array_equal(reader[1], third)


def test_hash_pattern_matches_fixed_width_sequence_numbers(tmp_path):
    write_exr(tmp_path / "IMG_0001.exr", np.ones((2, 2), dtype=np.float32), oiio.FLOAT)
    write_exr(tmp_path / "IMG_0010.exr", np.ones((2, 2), dtype=np.float32), oiio.FLOAT)
    write_exr(tmp_path / "IMG_010.exr", np.ones((2, 2), dtype=np.float32), oiio.FLOAT)

    with pytest.warns(UserWarning, match="numbering gaps"):
        reader = ImageSequenceReader(tmp_path / "IMG_####.exr")

    assert reader.frame_count == 2
    assert [reader.frame_info(index).source_number for index in range(reader.frame_count)] == [1, 10]


def test_image_sequence_strict_gaps_raise(tmp_path):
    write_exr(tmp_path / "IMG_0001.exr", np.ones((2, 2), dtype=np.float32), oiio.FLOAT)
    write_exr(tmp_path / "IMG_0003.exr", np.ones((2, 2), dtype=np.float32), oiio.FLOAT)

    with pytest.raises(InvalidFrameError, match="numbering gaps: 2"):
        ImageSequenceReader(tmp_path / "IMG_*.exr", strict_gaps=True)


def test_image_sequence_rejects_inconsistent_dimensions(tmp_path):
    write_exr(tmp_path / "IMG_0001.exr", np.ones((2, 2), dtype=np.float32), oiio.FLOAT)
    write_exr(tmp_path / "IMG_0002.exr", np.ones((3, 2), dtype=np.float32), oiio.FLOAT)

    with pytest.raises(InvalidFrameError, match="Inconsistent image dimensions"):
        ImageSequenceReader(tmp_path / "IMG_*.exr")
