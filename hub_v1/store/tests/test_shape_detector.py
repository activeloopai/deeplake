"""
License:
This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""

from hub_v1.exceptions import HubException
from hub_v1.store.shape_detector import ShapeDetector
import pytest


def test_shape_detector():
    s = ShapeDetector((10, 10, 10), 10)
    assert str(s.dtype) == "float64"
    assert s.chunks[1:] == (10, 10)


def test_shape_detector_2():
    s = ShapeDetector((10, 10, 10), 10, compressor="png")
    assert str(s.dtype) == "float64"
    assert s.chunks[1:] == (10, 10)


def test_shape_detector_wrong_shape():
    with pytest.raises(HubException):
        ShapeDetector((10, 10, 10), (10, 10, 20))


def test_shape_detector_wrong_shape_2():
    with pytest.raises(AssertionError):
        ShapeDetector((10, 10, 10), 20)


def test_shape_detector_wrong_shape_3():
    with pytest.raises(HubException):
        ShapeDetector((10, 10, None), (10, 10, None))


def test_shape_detector_wrong_chunk_shape():
    with pytest.raises(Exception):
        ShapeDetector((10, 10, 10), (10, 10, 10), (10, 10))


def test_shape_detector_wrong_chunk_value():
    with pytest.raises(Exception):
        ShapeDetector((10, 10, 10), (10, 10, 10), (2, 10, 10))
