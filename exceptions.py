class AstroIOError(Exception):
    """Base exception for AstroIO failures."""


class UnsupportedFormatError(AstroIOError):
    """Raised when no backend supports the requested format."""


class UnsupportedPixelFormatError(AstroIOError):
    """Raised when a backend cannot preserve or decode a pixel format."""


class InvalidFrameError(AstroIOError):
    """Raised when a frame does not match a writer or backend contract."""


class CorruptFileError(AstroIOError):
    """Raised when a file is structurally invalid or truncated."""


class RandomAccessUnsupportedError(AstroIOError):
    """Raised when exact indexed access is unavailable for a reader."""
