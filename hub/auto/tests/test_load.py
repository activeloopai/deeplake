import os
from hub.auto.load import load
from hub.auto.tests.common import get_dummy_data_path


def load_from_dummy_data(path: str):
    dummy_path = get_dummy_data_path(path)
    return load(dummy_path, symbolic=True)


def test_load_image():
    img = load_from_dummy_data("cat.jpeg")
    # assert img.shape == (900, 900, 3)
    # print(img.load())