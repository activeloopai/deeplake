"""
License:
This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""

import numpy as np
import pytest

from .png_numcodec import PngCodec


@pytest.mark.parametrize("from_config", [False, True])
def test_png_codec(from_config: bool) -> None:
    codec = PngCodec()
    if from_config:
        config = codec.get_config()
        codec = PngCodec.from_config(config)
    arr = np.ones((1000, 1000, 3), dtype="uint8")
    bytes_ = codec.encode(arr)
    arr_ = codec.decode(bytes_)
    assert (arr == arr_).all()


@pytest.mark.parametrize("solo_channel", [False, True])
def test_png_codec_config(solo_channel: bool) -> None:
    codec = PngCodec(solo_channel)
    config = codec.get_config()
    assert config["id"] == "png"
    assert config["solo_channel"] == solo_channel


@pytest.mark.parametrize("solo_channel", [False, True])
def test_png_codec_solo_channel(solo_channel: bool) -> None:
    codec = PngCodec(solo_channel)
    if solo_channel:
        arr = np.ones((1000, 1000, 1), dtype="uint8")
    else:
        arr = np.ones((1000, 1000), dtype="uint8")
    bytes_ = codec.encode(arr)
    arr_ = codec.decode(bytes_)
    assert (arr == arr_).all()
