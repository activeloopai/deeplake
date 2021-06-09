import io
import pytest
import numpy as np
from PIL import Image
from hub.core.tensor import read
from hub.util.dataset import get_compressor
import hub

EXTENSIONS = ("PNG", "JPEG")
CHECK_META = (True, False)
parametrize_extension = pytest.mark.parametrize("extension", EXTENSIONS)
parametrize_check_meta = pytest.mark.parametrize("check_meta", CHECK_META)


@parametrize_extension
@parametrize_check_meta
def test_read(extension: str, check_meta: bool):
    img_arr = np.ones((100, 100, 3)).astype("uint8")
    img = Image.fromarray(img_arr)
    img.save(f"/tmp/test_read.{extension.lower()}", format=extension)
    ds = hub.Dataset("/tmp/test_read")
    ds.create_tensor("image", dtype=img_arr.dtype)
    image_dict = read(f"/tmp/test_read.{extension.lower()}", check_meta=check_meta)
    assert image_dict["size"] == img_arr.shape[:2]
    assert image_dict["channels"] == img_arr.shape[2]
    assert image_dict["dtype"] == img_arr.dtype
    assert image_dict["compression"] == extension

    compressor = get_compressor(extension)
    image_decompressed = compressor.decode_single_image(image_dict["bytes"])
    assert np.all(image_decompressed == img_arr)
