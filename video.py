from __future__ import annotations

from fractions import Fraction
from pathlib import Path
from typing import Any

import av
import numpy as np

from .base import FrameInfo, FrameReader
from .exceptions import CorruptFileError, RandomAccessUnsupportedError, UnsupportedPixelFormatError
from .factory import register_reader

_AUTO_FORMAT = "auto"
_OUTPUT_DTYPES = {
    "rgb24": np.dtype(np.uint8),
    "rgb48le": np.dtype(np.uint16),
    "gray": np.dtype(np.uint8),
    "gray16le": np.dtype(np.uint16),
}
_OUTPUT_CHANNELS = {
    "rgb24": 3,
    "rgb48le": 3,
    "gray": 1,
    "gray16le": 1,
}


class VideoReader(FrameReader):
    """Sequential PyAV-backed video reader."""

    def __init__(self, path: str | Path, *, output_format: str = _AUTO_FORMAT):
        self.path = Path(path)
        self._requested_output_format = output_format

        with _open_container(self.path) as container:
            stream = _video_stream(container)
            self._width = int(stream.codec_context.width)
            self._height = int(stream.codec_context.height)
            self._frame_count = int(stream.frames or 0)
            self._fps = _rate_to_float(stream.average_rate or stream.base_rate)
            self._source_pixel_format = _format_name(stream.codec_context.format)
            self._container_format = container.format.name
            self._codec = stream.codec_context.name
            self._duration = _duration_seconds(stream.duration, stream.time_base)
            self._time_base = str(stream.time_base) if stream.time_base is not None else None
            self._timestamps: tuple[float | None, ...] | None = None

        if self._frame_count <= 0 or self._source_pixel_format is None:
            scan = _scan_video(self.path)
            self._frame_count = scan["frame_count"]
            self._source_pixel_format = self._source_pixel_format or scan["source_pixel_format"]
            self._timestamps = scan["timestamps"]

        self._output_format = _resolve_output_format(self._requested_output_format, self._source_pixel_format)

    @property
    def width(self) -> int:
        return self._width

    @property
    def height(self) -> int:
        return self._height

    @property
    def frame_count(self) -> int:
        return self._frame_count

    @property
    def dtype(self):
        return _OUTPUT_DTYPES[self._output_format]

    @property
    def channels(self) -> int:
        return _OUTPUT_CHANNELS[self._output_format]

    @property
    def supports_random_access(self) -> bool:
        return False

    @property
    def fps(self) -> float | None:
        return self._fps

    @property
    def timestamps(self):
        return self._timestamps

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "format": "video",
            "container": self._container_format,
            "codec": self._codec,
            "source_pixel_format": self._source_pixel_format,
            "requested_output_format": self._requested_output_format,
            "output_format": self._output_format,
            "duration_seconds": self._duration,
            "time_base": self._time_base,
        }

    def __getitem__(self, index: int):
        self._validate_index(index)
        raise RandomAccessUnsupportedError("VideoReader does not support exact indexed access")

    def __iter__(self):
        with _open_container(self.path) as container:
            stream = _video_stream(container)
            for frame in container.decode(stream):
                yield _frame_to_ndarray(frame, self._output_format)

    def frame_info(self, index: int) -> FrameInfo:
        self._validate_index(index)
        timestamp = None if self._timestamps is None else self._timestamps[index]
        return FrameInfo(index=index, source_path=self.path, source_number=index, timestamp=timestamp)


def _open_container(path: Path):
    try:
        return av.open(str(path))
    except av.AVError as exc:
        raise CorruptFileError(f"Could not open video: {path}") from exc


def _video_stream(container):
    try:
        return container.streams.video[0]
    except IndexError as exc:
        raise CorruptFileError("Video container has no video stream") from exc


def _frame_to_ndarray(frame, output_format: str) -> np.ndarray:
    image = frame.to_ndarray(format=output_format)
    image = np.asarray(image)
    if image.ndim == 3 and image.shape[2] == 1:
        image = image[:, :, 0]
    return image


def _resolve_output_format(requested: str, source_pixel_format: str | None) -> str:
    if requested != _AUTO_FORMAT:
        if requested not in _OUTPUT_DTYPES:
            raise UnsupportedPixelFormatError(f"Unsupported video output format: {requested}")
        return requested
    source_bits = _source_component_bits(source_pixel_format)
    return "rgb48le" if source_bits > 8 else "rgb24"


def _source_component_bits(source_pixel_format: str | None) -> int:
    if source_pixel_format is None:
        return 8
    try:
        pixel_format = av.VideoFormat(source_pixel_format)
    except ValueError:
        return 8
    bits = [component.bits for component in pixel_format.components if not component.is_alpha]
    return max(bits, default=8)


def _format_name(format_desc) -> str | None:
    if format_desc is None:
        return None
    return format_desc.name


def _rate_to_float(rate: Fraction | None) -> float | None:
    if rate is None:
        return None
    return float(rate)


def _duration_seconds(duration, time_base) -> float | None:
    if duration is None or time_base is None:
        return None
    return float(duration * time_base)


def _scan_video(path: Path) -> dict[str, Any]:
    frame_count = 0
    source_pixel_format = None
    timestamps: list[float | None] = []
    with _open_container(path) as container:
        stream = _video_stream(container)
        for frame in container.decode(stream):
            frame_count += 1
            source_pixel_format = source_pixel_format or frame.format.name
            timestamps.append(frame.time)
    if frame_count <= 0:
        raise CorruptFileError(f"Video contains no decodable frames: {path}")
    return {
        "frame_count": frame_count,
        "source_pixel_format": source_pixel_format,
        "timestamps": tuple(timestamps),
    }


register_reader("video", VideoReader, extensions=[".mp4", ".mov", ".avi", ".mkv"])
