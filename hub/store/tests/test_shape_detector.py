"""
License:
This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""

from hub.exceptions import HubException
from hub.store.shape_detector import ShapeDetector


def test_shape_detector():
    s = ShapeDetector((10, 10, 10), 10)
    assert str(s.dtype) == "float64"
    assert s.chunks[1:] == (10, 10)


def test_shape_detector_2():
    s = ShapeDetector((10, 10, 10), 10, compressor="png")
    assert str(s.dtype) == "float64"
    assert s.chunks[1:] == (10, 10)


def test_shape_detector_wrong_shape():
    try:
        ShapeDetector((10, 10, 10), (10, 10, 20))
    except HubException:
        return


def test_shape_detector_wrong_shape_2():
    try:
        ShapeDetector((10, 10, 10), 20)
    except AssertionError:
        return
    assert False


def test_shape_detector_wrong_shape_3():
    try:
        ShapeDetector((10, 10, None), (10, 10, None))
    except HubException:
        return
    assert False
