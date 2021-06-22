from typing import Sequence, Tuple

from numpy.core.fromnumeric import shape
from hub.core.storage.provider import StorageProvider
import pytest
import numpy as np
from hub.core.meta.shape_meta import ShapeMetaEncoder


def _add_shapes_and_assert_expected(
    shape_meta_encoder: ShapeMetaEncoder, expected_shapes: Sequence[Tuple[int]]
):
    start_num_samples = shape_meta_encoder.num_samples
    shape_meta_encoder.add_shapes(expected_shapes)
    end_num_samples = shape_meta_encoder.num_samples

    assert end_num_samples - start_num_samples == len(expected_shapes)

    for i in range(start_num_samples, end_num_samples):
        np.testing.assert_array_equal(
            shape_meta_encoder[i], expected_shapes[i - start_num_samples]
        )

    with pytest.raises(IndexError):
        shape_meta_encoder[end_num_samples]


def test_first_add(memory_storage: StorageProvider):
    shape_meta_encoder = ShapeMetaEncoder(memory_storage)

    _add_shapes_and_assert_expected(
        shape_meta_encoder,
        [
            (28, 28, 3),
            (28, 28, 3),
            (28, 28, 3),
            (28, 28, 3),
            (30, 28, 3),
            (30, 28, 3),
            (30, 28, 3),
        ],
    )

    _add_shapes_and_assert_expected(
        shape_meta_encoder,
        [
            (30, 45, 2),
            (30, 45, 2),
            (30, 45, 2),
            (30, 45, 3),
        ],
    )


def test_empty(memory_storage: StorageProvider):
    shape_meta_encoder = ShapeMetaEncoder(memory_storage)
    _add_shapes_and_assert_expected(shape_meta_encoder, [])
    _add_shapes_and_assert_expected(shape_meta_encoder, [(28, 28), (28, 28)])
    _add_shapes_and_assert_expected(shape_meta_encoder, [])
    _add_shapes_and_assert_expected(shape_meta_encoder, [(28, 28), (28, 28)])


def test_scalars(memory_storage: StorageProvider):
    shape_meta_encoder = ShapeMetaEncoder(memory_storage)
    _add_shapes_and_assert_expected(
        shape_meta_encoder,
        [
            (0,),
            (8,),
            (8,),
            (6,),
            (8,),
            (8,),
            (8,),
        ],
    )

    _add_shapes_and_assert_expected(
        shape_meta_encoder,
        [
            (8,),
            (8,),
            (8,),
            (7,),
            (8,),
            (8,),
            (8,),
        ],
    )


def test_failures(memory_storage: StorageProvider):
    shape_meta_encoder = ShapeMetaEncoder(memory_storage)

    _add_shapes_and_assert_expected(shape_meta_encoder, [(8, 1, 1)])

    with pytest.raises(ValueError):
        _add_shapes_and_assert_expected(shape_meta_encoder, [(8, 2, 1), (8, 1)])

    with pytest.raises(ValueError):
        _add_shapes_and_assert_expected(shape_meta_encoder, [(8, 1)])
