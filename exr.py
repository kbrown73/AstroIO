from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import OpenImageIO as oiio

from .base import FrameInfo, FrameReader, FrameWriter
from .exceptions import CorruptFileError, InvalidFrameError
from .factory import register_writer


class ExrReader(FrameReader):
    """Single-frame OpenEXR reader backed by OpenImageIO."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._spec = _read_spec(self.path)
        self._dtype = _dtype_for_format(self._spec.format)

    @property
    def width(self) -> int:
        return int(self._spec.width)

    @property
    def height(self) -> int:
        return int(self._spec.height)

    @property
    def frame_count(self) -> int:
        return 1

    @property
    def dtype(self):
        return self._dtype

    @property
    def channels(self) -> int:
        return int(self._spec.nchannels)

    @property
    def supports_random_access(self) -> bool:
        return True

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "format": "exr",
            "pixel_format": str(self._spec.format),
            "channel_names": list(self._spec.channelnames),
        }

    def __getitem__(self, index: int):
        self._validate_index(index)
        return read_exr(self.path)

    def frame_info(self, index: int) -> FrameInfo:
        self._validate_index(index)
        return FrameInfo(index=index, source_path=self.path)


class ExrWriter(FrameWriter):
    """Single-frame OpenEXR writer backed by OpenImageIO."""

    def __init__(
        self,
        path: str | Path,
        *,
        width: int,
        height: int,
        dtype,
        channels: int,
        half: bool = True,
        create_dirs: bool = True,
    ):
        self.path = Path(path)
        self.width = int(width)
        self.height = int(height)
        self.dtype = np.dtype(dtype)
        self.channels = int(channels)
        self.half = half
        self.create_dirs = create_dirs
        self._written = False

    def write(self, frame) -> None:
        if self._written:
            raise InvalidFrameError(f"EXR writer only accepts one frame: {self.path}")
        image = _normalize_write_image(frame)
        self._validate_frame(image)
        _write_exr_image(self.path, image, half=self.half, create_dirs=self.create_dirs)
        self._written = True

    def _validate_frame(self, image: np.ndarray) -> None:
        height, width, channels = image.shape
        if width != self.width or height != self.height:
            raise InvalidFrameError(
                f"EXR frame has dimensions {width}x{height}, expected {self.width}x{self.height}"
            )
        if channels != self.channels:
            raise InvalidFrameError(f"EXR frame has {channels} channels, expected {self.channels}")
        if image.dtype != self.dtype:
            raise InvalidFrameError(f"EXR frame has dtype {image.dtype}, expected {self.dtype}")


def read_exr(path: str | Path) -> np.ndarray:
    path = Path(path)
    inp = oiio.ImageInput.open(str(path))
    if inp is None:
        raise CorruptFileError(f"Could not open EXR: {path}")
    try:
        spec = inp.spec()
        image = inp.read_image(format=spec.format)
    finally:
        inp.close()

    if image is None:
        raise CorruptFileError(f"Could not read EXR pixels: {path}")

    image = np.asarray(image)
    if image.ndim == 3 and image.shape[2] == 1:
        image = image[:, :, 0]
    return image


def write_exr(path: str | Path, image: np.ndarray, *, half: bool = True, create_dirs: bool = True) -> None:
    normalized = _normalize_write_image(image)
    height, width, channels = normalized.shape
    with ExrWriter(
        path,
        width=width,
        height=height,
        dtype=normalized.dtype,
        channels=channels,
        half=half,
        create_dirs=create_dirs,
    ) as writer:
        writer.write(normalized)


def _read_spec(path: Path):
    inp = oiio.ImageInput.open(str(path))
    if inp is None:
        raise CorruptFileError(f"Could not open EXR: {path}")
    try:
        return inp.spec()
    finally:
        inp.close()


def _dtype_for_format(format_desc) -> np.dtype:
    if format_desc == oiio.UINT8:
        return np.dtype(np.uint8)
    if format_desc == oiio.UINT16:
        return np.dtype(np.uint16)
    if format_desc == oiio.UINT:
        return np.dtype(np.uint32)
    if format_desc == oiio.HALF:
        return np.dtype(np.float16)
    if format_desc == oiio.FLOAT:
        return np.dtype(np.float32)
    if format_desc == oiio.DOUBLE:
        return np.dtype(np.float64)
    return np.dtype(np.float32)


def _normalize_write_image(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if image.ndim == 2:
        image = image[:, :, np.newaxis]
    if image.ndim != 3:
        raise InvalidFrameError(f"Expected 2D mono or 3D channel image, got shape {image.shape}")
    return image


def _write_exr_image(path: Path, image: np.ndarray, *, half: bool, create_dirs: bool) -> None:
    if create_dirs:
        path.parent.mkdir(parents=True, exist_ok=True)
    height, width, channels = image.shape
    pixel_type = oiio.HALF if half else oiio.FLOAT
    spec = oiio.ImageSpec(width, height, channels, pixel_type)
    out = oiio.ImageOutput.create(str(path))
    if out is None:
        raise CorruptFileError(f"Could not create EXR output: {path}")
    try:
        if not out.open(str(path), spec):
            raise CorruptFileError(f"Could not open EXR output: {path}: {out.geterror()}")
        if not out.write_image(image):
            raise CorruptFileError(f"Could not write EXR output: {path}: {out.geterror()}")
    finally:
        out.close()


register_writer("exr", ExrWriter)
