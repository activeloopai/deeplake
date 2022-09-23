import os
import io
import pytest
import numpy as np
from PIL import Image  # type: ignore

from deeplake.core.tiling.optimizer import get_tile_shape


def test_tile_shape():
    x = np.random.random((1000, 500, 3))
    tile_shape = get_tile_shape(
        sample_shape=x.shape,
        sample_size=x.nbytes,
        chunk_size=x.nbytes / 16,
        exclude_axes=-1,
    )
    assert np.prod(tile_shape) * x.itemsize == x.nbytes / 16
    assert tile_shape[2] == 3


@pytest.mark.parametrize("compression", ["jpeg", "png", "tiff"])
def test_tile_shape_compressed(compression, compressed_image_paths):
    path = compressed_image_paths[compression][0]
    sample_size = os.path.getsize(path)
    arr = np.array(Image.open(path))
    sample_shape = arr.shape
    chunk_size = sample_size / 10
    tile_shape = get_tile_shape(
        sample_shape=sample_shape,
        sample_size=sample_size,
        chunk_size=chunk_size,
        exclude_axes=-1,
    )
    tile = arr[: tile_shape[0], : tile_shape[1]]
    bio = io.BytesIO()
    Image.fromarray(tile).save(bio, compression)
    bio.seek(0)
    nbytes_compressed = len(bio.read())
    assert (
        nbytes_compressed < chunk_size
    )  # tiles wont be optimal for small compressed samples.


@pytest.mark.parametrize("compression", ["jpeg", "png"])
def test_tile_shape_large_compressed(compression):
    arr = np.random.randint(0, 256, (1000, 1000, 3), dtype=np.uint8)
    bio = io.BytesIO()
    Image.fromarray(arr).save(bio, compression)
    bio.seek(0)
    sample_size = len(bio.read())
    chunk_size = sample_size / 10
    tile_shape = get_tile_shape(
        sample_shape=arr.shape,
        sample_size=sample_size,
        chunk_size=chunk_size,
        exclude_axes=-1,
    )
    tile = arr[: tile_shape[0], : tile_shape[1]]
    bio = io.BytesIO()
    Image.fromarray(tile).save(bio, compression)
    bio.seek(0)
    tile_size = len(bio.read())
    assert 1 <= chunk_size / tile_size < 2
