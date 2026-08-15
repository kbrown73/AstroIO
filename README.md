# AstroIO

AstroIO is a small Python module for reading and writing astronomy and
image-sequence formats through a common frame-oriented API.

It is designed for processing pipelines that want to work with NumPy arrays
without baking format-specific code into the processing layer.

## Goals

- Provide a common reader interface for frame-based sources.
- Return frames as NumPy arrays.
- Preserve source pixel data as closely as the backend permits.
- Keep IO concerns separate from processing concerns.
- Make format-specific metadata available without forcing every format into one
  rigid schema.

AstroIO does not implicitly normalize images, debayer raw data, convert colour
spaces, resize frames, apply gamma, or promote data to a working dtype. Those
steps belong in the calling application.

## Current Status

Implemented:

- EXR single-frame reading through OpenImageIO.
- EXR image-sequence reading.
- EXR writing through OpenImageIO.
- Reader and writer factory functions.
- Frame side-channel metadata through `FrameInfo`.

Planned:

- MP4 / AVI / other video containers through PyAV.
- TIFF and PNG image sequences.
- FITS.
- SER.

## Requirements

Core:

```text
numpy
```

Current EXR support:

```text
OpenImageIO
```

Planned optional backends:

```text
av          # MP4 / AVI / video containers
astropy     # FITS
tifffile    # TIFF sequences
Pillow      # PNG/JPEG fallback
```

On Debian/Ubuntu-like systems, OpenImageIO may be available through packages
such as:

```text
python3-openimageio
openimageio-tools
openexr
```

Make sure dependencies are installed into the same Python environment that runs
AstroIO. A quick check:

```bash
python3 -c "import numpy, OpenImageIO"
```

## Reading

Use `open_reader()` for normal application code:

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

`frame_count` is always known for a `FrameReader`. Sources that cannot provide a
known count cheaply should use a future streaming API rather than this base
reader interface.

## Frame Shape And Dtype

Mono frames are returned as 2D arrays:

```python
frame.shape == (height, width)
```

RGB/RGBA frames are returned with an explicit channel dimension:

```python
frame.shape == (height, width, channels)
```

The returned dtype reflects the decoded source representation exposed by the
backend. For example, EXR HALF data is returned as `float16`, and EXR FLOAT data
is returned as `float32`.

Applications that need a specific working format should convert explicitly:

```python
working = frame.astype("float32", copy=False)
```

## FrameInfo

`frame_info(index)` returns per-frame side-channel information without wrapping
the NumPy array itself:

```python
info = reader.frame_info(2)
print(info.source_path)
print(info.source_number)
```

The AstroIO frame index is dense and 0-based. `source_number`, where available,
is metadata inferred from filenames or containers.

Example:

```text
reader[0] -> IMG_0001.exr, source_number=1
reader[1] -> IMG_0002.exr, source_number=2
reader[2] -> IMG_0004.exr, source_number=4
```

In this example `IMG_0003.exr` is missing from the source numbering, but
`reader[2]` is still valid and returns `IMG_0004.exr`.

## Image Sequences

AstroIO treats discovered image files as the ordered frames to process.
Numbering gaps are non-fatal by default: the reader warns and continues.

Supported forms:

```python
astroio.open_reader("IMG_*.exr")
astroio.open_reader("IMG_####.exr")
astroio.open_reader(["IMG_0001.exr", "IMG_0002.exr", "IMG_0004.exr"], format="exr")
```

Sequence frames must currently have consistent dimensions, dtype, and channel
count.

## Writing EXR

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

For simple single-frame EXR output, use the convenience helper:

```python
from astroio.exr import write_exr

write_exr("frame.exr", frame, half=True)
```

`half=True` writes HALF pixels. Use `half=False` for FLOAT output.

## Video Roadmap

MP4 and many AVI files are decoded from compressed video rather than preserved
as native astronomy image data. The planned video reader will expose decoded
frames, record codec/container/source pixel-format metadata, and avoid extra
hidden conversion beyond what the decoder requires.
