from __future__ import annotations

from fractions import Fraction
import re
from pathlib import Path
from typing import Any

import av
import cv2
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
_BAYER_PATTERNS = {"RGGB", "BGGR", "GBRG", "GRBG"}
_BAYER_FORMAT_RE = re.compile(r"^bayer_(rggb|bggr|gbrg|grbg)(8|16)(?:le|be)?$")


class VideoReader(FrameReader):
    """Sequential PyAV-backed video reader."""

    def __init__(self, path: str | Path, *, output_format: str = _AUTO_FORMAT, debayer: str = "none"):
        self.path = Path(path)
        self._requested_output_format = output_format
        self._requested_debayer = debayer

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

        self._debayer_pattern = _resolve_debayer_pattern(self._requested_debayer, self._source_pixel_format)
        self._output_format = _resolve_output_format(
            self._requested_output_format,
            self._source_pixel_format,
            debayer_pattern=self._debayer_pattern,
        )

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
            "requested_debayer": self._requested_debayer,
            "debayer_pattern": self._debayer_pattern,
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
                yield _frame_to_ndarray(frame, self._output_format, debayer_pattern=self._debayer_pattern)

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


def _frame_to_ndarray(frame, output_format: str, *, debayer_pattern: str | None = None) -> np.ndarray:
    if debayer_pattern is not None:
        raw_format = _raw_bayer_format_for_output(output_format)
        image = frame.to_ndarray(format=raw_format)
        image = np.asarray(image)
        if image.ndim == 3 and image.shape[2] == 1:
            image = image[:, :, 0]
        return _debayer_mosaic(image, debayer_pattern)

    image = frame.to_ndarray(format=output_format)
    image = np.asarray(image)
    if image.ndim == 3 and image.shape[2] == 1:
        image = image[:, :, 0]
    return image


def _resolve_output_format(
    requested: str,
    source_pixel_format: str | None,
    *,
    debayer_pattern: str | None = None,
) -> str:
    if requested != _AUTO_FORMAT:
        if requested not in _OUTPUT_DTYPES:
            raise UnsupportedPixelFormatError(f"Unsupported video output format: {requested}")
        if debayer_pattern is not None and requested not in {"rgb24", "rgb48le"}:
            raise UnsupportedPixelFormatError("Debayering requires an RGB video output format")
        return requested
    source_bits = _source_component_bits(source_pixel_format)
    if debayer_pattern is not None:
        return "rgb48le" if source_bits > 8 else "rgb24"
    return "rgb48le" if source_bits > 8 else "rgb24"


def _source_component_bits(source_pixel_format: str | None) -> int:
    if source_pixel_format is None:
        return 8
    match = _BAYER_FORMAT_RE.match(source_pixel_format)
    if match is not None:
        return int(match.group(2))
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


def _resolve_debayer_pattern(requested: str, source_pixel_format: str | None) -> str | None:
    token = requested.strip().upper()
    if token == _AUTO_FORMAT.upper():
        pattern = _source_bayer_pattern(source_pixel_format)
        if pattern is None:
            source = source_pixel_format or "unknown"
            raise UnsupportedPixelFormatError(
                f"Could not infer debayer pattern from video pixel format: {source}. "
                "Pass debayer='RGGB', 'BGGR', 'GBRG', or 'GRBG', or use debayer='none'."
            )
        return pattern
    if token == "NONE":
        return None
    if token not in _BAYER_PATTERNS:
        raise UnsupportedPixelFormatError(f"Unsupported debayer pattern: {requested}")
    return token


def _source_bayer_pattern(source_pixel_format: str | None) -> str | None:
    if source_pixel_format is None:
        return None
    match = _BAYER_FORMAT_RE.match(source_pixel_format)
    if match is None:
        return None
    return match.group(1).upper()


def _raw_bayer_format_for_output(output_format: str) -> str:
    if output_format == "rgb48le":
        return "gray16le"
    if output_format == "rgb24":
        return "gray"
    raise UnsupportedPixelFormatError("Debayering requires an RGB video output format")


def _debayer_mosaic(image: np.ndarray, pattern: str) -> np.ndarray:
    if image.ndim != 2:
        raise UnsupportedPixelFormatError("Debayering requires a single-channel raw mosaic frame")
    code = getattr(cv2, f"COLOR_Bayer{pattern}2RGB")
    return cv2.cvtColor(image, code)


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
