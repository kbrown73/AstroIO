# AstroIO Module — Draft Specification

## 1. Purpose

`astroio` is a small reusable Python module for reading and writing astronomy/image-sequence formats through a common frame-oriented API.

Primary initial formats (in order of priority):

- EXR image sequences
- MP4 / AVI / other video containers via PyAV
- Other image sequences (TIFF, PNG, etc.)
- FITS
- SER

The processing layer should operate on NumPy arrays and remain independent of the source file format.

---

## 2. Design Goals

1. Present all supported frame-based sources through a common reader interface.
2. Use NumPy arrays as the common in-memory image representation.
3. Preserve source pixel data as faithfully as possible.
4. Never implicitly change:
   - bit depth
   - dtype
   - colour space
   - channel count
   - Bayer/CFA layout
5. Keep normalized/common metadata separate from format-specific metadata.
6. Support random access where practical.
7. Support sequential iteration efficiently.
8. Keep readers and writers as separate abstractions.
9. Make it straightforward to add new backends later.

---

## 3. Proposed Package Layout

```text
astroio/
    __init__.py
    base.py
    factory.py

    exr.py
    fits.py
    image_sequence.py
    ser.py
    video.py

    metadata.py        # optional
    exceptions.py      # optional
```

Possible later additions:

```text
    raw.py
```

---

## 4. Common Image Representation

AstroIO preserves the native pixel representation. Processing code may explicitly promote images to a floating-point working representation, with float32 recommended as the normal default. No implicit normalization is performed.

Frames should be returned as NumPy arrays.

Typical mono frame:

```python
frame.shape == (height, width)
frame.dtype == np.uint8 | np.uint16 | np.float32
```

Mono images should be returned as 2D arrays. Callers that prefer an explicit
singleton channel dimension should add that dimension outside `astroio`.

Typical RGB frame:

```python
frame.shape == (height, width, 3)
```

The reader must not silently normalize or convert image data.

Examples of transformations that belong outside `astroio`:

- normalization to 0..1
- conversion to float
- debayering
- RGB conversion
- gamma correction
- colour management
- resizing
- denoising
- registration/alignment

---

## 5. Reader API

### 5.1 Base Reader

Suggested abstract interface:

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class FrameInfo:
    index: int
    source_path: Path | None = None
    source_number: int | None = None
    timestamp: object | None = None
    metadata: dict = field(default_factory=dict)


class FrameReader(ABC):

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
    def metadata(self) -> dict:
        return {}

    @abstractmethod
    def __getitem__(self, index: int):
        ...

    def frame_info(self, index: int) -> FrameInfo:
        if index < 0 or index >= self.frame_count:
            raise IndexError(index)
        return FrameInfo(index=index)

    def __len__(self):
        return self.frame_count

    def __iter__(self):
        for i in range(self.frame_count):
            yield self[i]

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
```

`frame_count` is always known for a `FrameReader`. Inputs where the frame count
cannot be determined cheaply should either determine it at open time or use a
separate future streaming API.

`__getitem__` must provide exact indexed access. Backends that cannot support
exact random access should set `supports_random_access` to `False` and raise an
appropriate error, such as `RandomAccessUnsupportedError`, from `__getitem__`.
Approximate video seeking should not be hidden behind normal indexing.

Individual backends may override iteration when a sequential decoder is more
efficient than repeated random access. Backends that do not support random
access must override iteration.

---

## 6. Factory Function

Provide a simple entry point:

```python
reader = open_reader("capture.ser")
```

Suggested signature:

```python
def open_reader(path, *, format=None, **kwargs) -> FrameReader:
    ...
```

Format should normally be inferred from:

- extension
- file signature where useful

The explicit `format=` argument can override detection.

Example:

```python
with open_reader("capture.ser") as src:
    print(src.width)
    print(src.height)
    print(src.frame_count)
    print(src.dtype)
    print(src.supports_random_access)

    if src.supports_random_access:
        frame = src[100]

    for frame in src:
        process(frame)
```

---

## 7. Common Reader Properties

Readers should expose the following normalized properties where meaningful:

```text
width
height
frame_count
dtype
channels
supports_random_access
fps
timestamps
metadata
```

### Semantics

#### `width`
Image width in pixels.

#### `height`
Image height in pixels.

#### `frame_count`
Total number of frames. `FrameReader` implementations must know this value.

#### `dtype`
NumPy dtype of returned frame data.

Examples:

```python
np.uint8
np.uint16
np.float32
```

#### `channels`
Suggested convention:

```text
1 = mono / CFA raw
3 = RGB
4 = RGBA
```

Bayer/CFA images should not automatically be counted as RGB simply because they represent colour data.

#### `supports_random_access`
Whether `reader[index]` is supported as exact indexed access.

When this is `False`, callers should use sequential iteration:

```python
for frame in reader:
    process(frame)
```

Backends must not use approximate seeking for `__getitem__`.

#### `fps`
Frames per second where meaningful.

Return `None` when the format does not define a useful fixed frame rate.

#### `timestamps`
Optional per-frame timestamps.

May be:

- `None`
- a sequence
- a lazy accessor

The exact representation can be finalized during implementation.

#### `metadata`
Dictionary containing format-specific metadata that does not fit the normalized API.

#### `frame_info(index)`
Return per-frame side-channel information without wrapping the returned NumPy
array.

Common fields:

```python
FrameInfo(
    index=0,
    source_path=Path("IMG_0001.exr"),
    source_number=1,
    timestamp=None,
    metadata={},
)
```

`index` is the dense 0-based AstroIO frame index. `source_number`, where known,
is the original frame number inferred from the source, such as `1` from
`IMG_0001.exr`. Processing code should use `index` for iteration order and treat
`source_number` as metadata.

---

## 8. Metadata Policy

Do not force all formats into a single rigid metadata schema.

Instead use:

1. a small normalized set of common properties
2. a backend-specific `metadata` dictionary

Example FITS metadata:

```python
{
    "EXPTIME": 0.005,
    "GAIN": 120,
    "OBJECT": "Sun",
}
```

Example SER metadata:

```python
{
    "observer": "...",
    "instrument": "...",
    "telescope": "...",
    "color_id": ...,
    "little_endian": True,
}
```

Example AVI metadata:

```python
{
    "codec": "...",
    "pixel_format": "gray16le",
    "container": "avi",
}
```

Where possible, original source metadata should be preserved without lossy reinterpretation.

---

## 9. Writer API

Reader and writer APIs should remain separate.

Suggested base class:

```python
class FrameWriter(ABC):

    @abstractmethod
    def write(self, frame):
        ...

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
```

Factory:

```python
writer = open_writer(
    "result.ser",
    width=reader.width,
    height=reader.height,
    dtype=reader.dtype,
    channels=reader.channels,
)
```

Typical usage:

```python
with open_reader("input.ser") as reader:
    with open_writer(
        "output.ser",
        width=reader.width,
        height=reader.height,
        dtype=reader.dtype,
        channels=reader.channels,
    ) as writer:

        for frame in reader:
            writer.write(frame)
```

Writers should validate input frames rather than silently converting them.

For example, a writer initialized as `uint16` mono should reject an unexpected `float32 RGB` frame unless conversion has been explicitly requested by the caller.

Full writer option coverage can be added as formats need it. Likely later
options include overwrite policy, metadata, output pixel format, compression,
channel names, and directory creation. Reader support is the first priority.

---

## 10. Backend: SER

SER is a strong candidate for a native implementation.

Reasons:

- relatively simple binary format
- uncompressed frame data
- fixed-size header
- optional timestamp trailer
- easy random access
- suitable for `numpy.memmap`

Suggested class:

```python
class SerReader(FrameReader):
    ...
```

Potential implementation strategy:

1. Parse SER header.
2. Determine:
   - width
   - height
   - frame count
   - colour/CFA type
   - pixel depth
   - byte order
3. Determine byte offset of image data.
4. Expose frames using either:
   - `numpy.memmap`
   - direct seek/read
5. Parse timestamps if present.

Example:

```python
reader = SerReader("capture.ser")
frame = reader[123]
```

Possible writer:

```python
class SerWriter(FrameWriter):
    ...
```

Initial SER implementation scope:

- mono 8-bit
- mono 16-bit

Later:

- Bayer/CFA formats
- RGB formats
- timestamps
- complete metadata preservation

SER file documentation can be found at https://siril.readthedocs.io/en/latest/file-formats/SER.html

---

## 11. Backend: FITS

Use:

```text
astropy.io.fits
```

Suggested class:

```python
class FitsReader(FrameReader):
    ...
```

FITS may represent:

- a single 2D image
- a 3D image cube
- more complex HDU structures

Initial convention could treat:

```text
2D FITS -> one frame
3D FITS -> frame stack
```

Example:

```python
data.shape == (height, width)
```

means:

```python
frame_count == 1
```

while:

```python
data.shape == (frames, height, width)
```

means:

```python
frame_count == frames
```

The original FITS header should be accessible through metadata.

By default, FITS reading should preserve the stored array representation rather
than applying implicit physical-value scaling that changes dtype. Transformations
such as applying `BSCALE` / `BZERO` should require an explicit caller option.

Writing should use Astropy.

---

## 12. Backend: MP4 / AVI / Video

Use:

```text
PyAV
```

rather than OpenCV as the primary backend.

Reasons:

- better access to codec/container metadata
- explicit pixel formats
- better handling of unusual raw/uncompressed formats
- less risk of implicit conversion to conventional 8-bit BGR images

Suggested class:

```python
class VideoReader(FrameReader):
    ...
```

Important requirements:

- inspect source pixel format
- preserve decoded mono/bit-depth data where PyAV permits
- avoid implicit conversion unless explicitly requested
- expose codec, container, and source pixel format in metadata

Example pixel formats:

```text
gray
gray16le
rgb24
rgb48le
```

Random access may be expensive or approximate for compressed formats.

The reader should therefore prefer sequential iteration for inter-frame
compressed video. If exact indexed access is unavailable, video readers should
set:

```python
supports_random_access == False
```

and `reader[100]` should raise rather than performing approximate seeking.

MP4 and similar formats are usually decoded from compressed YUV data rather than
native astronomy image data. AstroIO should be explicit about the decoded pixel
representation it returns and should avoid additional hidden conversion beyond
what the decoder requires.

---

## 13. Backend: Image Sequences

This should be supported early because it is useful in astronomy and VFX workflows.

Suggested class:

```python
class ImageSequenceReader(FrameReader):
    ...
```

Example sequence:

```text
IMG_0001.exr
IMG_0002.exr
IMG_0003.exr
```

Potential constructors:

```python
reader = ImageSequenceReader(
    "IMG_####.exr"
)
```

or:

```python
reader = ImageSequenceReader(
    "IMG_*.exr"
)
```

or:

```python
reader = ImageSequenceReader(
    [
        "IMG_0001.exr",
        "IMG_0002.exr",
        "IMG_0003.exr",
    ]
)
```

Potential supported formats:

- TIFF
- PNG
- EXR
- JPEG
- possibly camera RAW later

The sequence reader should:

1. sort filenames deterministically
2. infer source frame numbers where practical
3. warn about detected numbering gaps by default
4. allow gaps by default
5. verify dimensions and dtype consistency
6. expose frames through the normal reader API

For sequences with non-contiguous camera frame numbers, sequence position and
source frame number should remain distinct concepts.

The dense AstroIO index is always the processing order:

```text
reader[0] -> IMG_0001.exr
reader[1] -> IMG_0002.exr
reader[2] -> IMG_0004.exr
```

The missing `IMG_0003.exr` is informational by default. `frame_info(2)` may
report:

```python
FrameInfo(
    index=2,
    source_path=Path("IMG_0004.exr"),
    source_number=4,
)
```

A future strict mode may turn numbering gaps into errors for workflows that
require contiguous source numbering.

---

## 14. Error Handling

Potential custom exceptions:

```python
class AstroIOError(Exception):
    pass

class UnsupportedFormatError(AstroIOError):
    pass

class UnsupportedPixelFormatError(AstroIOError):
    pass

class InvalidFrameError(AstroIOError):
    pass

class CorruptFileError(AstroIOError):
    pass

class RandomAccessUnsupportedError(AstroIOError):
    pass
```

Avoid silently recovering from structural or pixel-format errors when doing so could alter image interpretation.

---

## 15. Performance Considerations

### SER
Prefer memory mapping for large files.

### FITS
Use Astropy's memory mapping where practical.

### AVI/video
Prefer sequential decoding.

### Image sequences
Load frames lazily rather than loading the entire sequence into RAM.

In general:

```text
opening a reader should not load all image data
```

unless the backend requires it.

---

## 16. Threading / Parallelism

Do not build threading or multiprocessing into the first version.

The reader API should instead be simple enough for calling code to parallelize processing if desired.

Example:

```python
frame = reader[i]
```

should return an independent NumPy frame suitable for downstream processing.

Thread-safety guarantees can be added later if required.

---

## 17. Dependencies

Initial likely dependencies:

```text
numpy
astropy
av
```

Image sequence support may additionally use one or more of:

```text
imageio
tifffile
OpenImageIO
OpenEXR
opencv-python
Pillow
```

Prefer optional backend dependencies where possible.

For example:

```text
astroio core
astroio[fits]
astroio[video]
astroio[all]
```

could eventually be supported through package extras.

---

## 18. Public API

Possible `astroio/__init__.py` exports:

```python
from .factory import open_reader, open_writer
from .base import FrameInfo, FrameReader, FrameWriter
```

Typical user-facing code should require little or no knowledge of individual backend classes:

```python
import astroio

with astroio.open_reader("capture.ser") as src:
    frame = src[0]
```

Backend classes can remain available for advanced use:

```python
from astroio.ser import SerReader
```

---

## 19. Testing Strategy

Create small reference files for each backend.

Tests should cover:

### General reader behaviour

- dimensions
- frame count
- dtype
- channel count
- random-access capability
- iteration
- indexed access
- frame info
- context manager behaviour

### Pixel integrity

For lossless formats:

```python
np.array_equal(source_frame, decoded_frame)
```

should hold.

### Round-trip tests

Where writing is supported:

```text
NumPy frame
    -> writer
    -> file
    -> reader
    -> NumPy frame
```

The result should be identical for lossless formats.

### Metadata

Verify important metadata survives where the format permits it.

### Invalid input

Test:

- corrupt headers
- unsupported bit depth
- truncated frames
- inconsistent image sequences
- image sequence gaps
- out-of-range frame access
- unsupported random access

---

## 20. Suggested Initial Milestones

### Milestone 1 — Core API

Implement:

- `FrameReader`
- `FrameWriter`
- `open_reader()`
- known `frame_count`
- `supports_random_access`
- `FrameInfo`
- common exceptions

### Milestone 2 — EXR Image Sequence Reader

Implement:

- EXR sequence discovery
- deterministic sorting
- native-preserving EXR reads
- random access
- iteration
- frame info
- gap warnings

### Milestone 3 — Eclipse Align Adapter

Implement an explicit compatibility layer in `eclipse_align`:

- read frames through AstroIO
- convert to the existing float32 working representation outside AstroIO
- preserve existing EXR output behavior until AstroIO writer support is needed

### Milestone 4 — PyAV Video Reader

Implement:

- MP4
- AVI
- sequential decoding
- common decoded mono/RGB pixel formats
- codec/container/source pixel format metadata
- explicit `supports_random_access` behavior

### Milestone 5 — FITS Reader

Implement using Astropy:

- raw/preserved default reads
- 2D images
- 3D image cubes
- FITS header access

### Milestone 6 — Other Image Sequences

Implement:

- TIFF
- PNG
- filename sequence detection
- missing-frame warnings

### Milestone 7 — SER Reader

Implement:

- mono 8-bit
- mono 16-bit
- random access
- iteration
- basic metadata

### Milestone 8 — Writers As Needed

Initial writer work should be driven by real use cases.

Likely first writer:

- EXR output support for `eclipse_align`

Later possible writers:

- SER mono 8-bit / 16-bit
- FITS

---

## 21. Key Architectural Rule

The core processing pipeline should never need to know whether a frame originated from:

```text
SER
FITS
AVI
MP4
EXR sequence
TIFF sequence
```

It should only receive:

```python
numpy.ndarray
```

plus any explicitly requested metadata.

Conceptually:

```text
Video --- PyAV ------------+
                           |
SER  ---- native reader ---+
                           +--> NumPy ndarray --> processing
FITS ---- Astropy ---------+
                           |
Images -- sequence reader -+
```

This separation is the main purpose of the module.

---

## 22. Open Questions

These can be resolved during implementation rather than up front:

1. Exact timestamp representation:
   - `datetime`
   - NumPy datetime
   - integer nanoseconds
   - raw backend values

2. How Bayer/CFA layout should be exposed:
   - normalized `cfa_pattern` property
   - metadata only

3. Whether multi-HDU FITS files should appear as:
   - multiple readers
   - selectable HDUs
   - a separate FITS-specific API

4. Exact EXR backend:
   - OpenEXR
   - OpenImageIO
   - imageio
   - another library

5. Whether writers should accept optional explicit conversion policies.

6. Exact warning mechanism for non-fatal image sequence gaps.

7. Exact decoded pixel representation defaults for compressed video sources.
