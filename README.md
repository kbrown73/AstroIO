# AstroIO

AstroIO is a small Python module for reading astronomy and image-sequence
formats through a common frame-oriented API.

The core contract is simple:

- readers return NumPy arrays
- mono frames are returned as `(height, width)`
- RGB/RGBA frames are returned as `(height, width, channels)`
- readers preserve source dtype, bit depth, colour space, channel count, and CFA
  layout where the backend can expose them
- processing code performs explicit conversion to working formats such as
  `float32`

## Initial Scope

Implementation priority:

1. EXR image sequences
2. MP4 / AVI / other video containers through PyAV
3. Other image sequences such as TIFF and PNG
4. FITS
5. SER

Writer support is intentionally secondary until a concrete workflow needs it.
The current concrete writer support is EXR output backed by OpenImageIO.

## Runtime Requirements

Core AstroIO needs:

```text
numpy
```

Planned backend dependencies:

```text
av              # MP4 / AVI / video containers via PyAV
OpenImageIO     # likely first EXR backend, matching eclipse_align
astropy         # FITS
tifffile        # TIFF sequences
Pillow          # PNG/JPEG fallback
```

`av` is enough for the planned video reader. For the first EXR milestone this
repo currently uses the Python `OpenImageIO` bindings, so that package also
needs to be importable in the Python environment used to run AstroIO.

Be careful about Python environments. In this checkout, `pytest` may run under a
different interpreter than the shell `python` or `pip`. Check the target
environment directly, for example:

```bash
python -c "import av, OpenImageIO"
/usr/bin/python3 -c "import av, OpenImageIO"
```

On Debian/Ubuntu-like systems the current `eclipse_align` setup has used:

```text
python3-numpy
python3-openimageio
openimageio-tools
openexr
```

The exact packaging may differ if AstroIO is later split into a standalone
module or installed with Python package extras.

## Reader API

Typical usage:

```python
import astroio

with astroio.open_reader("IMG_*.exr") as reader:
    print(reader.width, reader.height, reader.frame_count)
    print(reader.dtype, reader.channels)

    for frame in reader:
        process(frame)
```

Readers expose:

- `width`
- `height`
- `frame_count`
- `dtype`
- `channels`
- `supports_random_access`
- `fps`
- `timestamps`
- `metadata`
- `frame_info(index)`

## EXR Writing

EXR writing is available through the generic writer API:

```python
import numpy as np
import astroio

frame = np.zeros((100, 100, 3), dtype=np.float32)

with astroio.open_writer(
    "frame.exr",
    width=100,
    height=100,
    dtype=np.float32,
    channels=3,
) as writer:
    writer.write(frame)
```

The convenience helper is also available for simple single-frame writes:

```python
from astroio.exr import write_exr

write_exr("frame.exr", frame, half=True)
```

`half=True` writes HALF pixels, matching the current `eclipse_align` output
default. Use `half=False` for FLOAT output.

`frame_count` is always known for a `FrameReader`. If a source cannot provide a
known count cheaply, it should be handled by a future streaming API rather than
this base reader interface.

## FrameInfo

`frame_info(index)` returns side-channel information without wrapping each NumPy
array:

```python
FrameInfo(
    index=2,
    source_path=Path("IMG_0004.exr"),
    source_number=4,
)
```

The AstroIO `index` is always dense and 0-based. `source_number` is metadata
inferred from filenames or containers when available.

## Image Sequence Gaps

AstroIO treats discovered image files as the ordered frames to process. If a
numbered sequence appears to skip a source number, the reader should warn by
default but continue:

```text
reader[0] -> IMG_0001.exr
reader[1] -> IMG_0002.exr
reader[2] -> IMG_0004.exr
```

In this example `IMG_0003.exr` is informationally missing, but `reader[2]` is
valid and returns `IMG_0004.exr`.

## Video Notes

MP4 and many AVI files are decoded from compressed video rather than preserved
as native astronomy image data. AstroIO should expose the decoded pixel
representation clearly, record codec/container/source pixel-format metadata, and
avoid hidden conversion beyond what the decoder requires.
