import numpy as np

import hub
from hub.schema import Image, Mask, ClassLabel, Text
from hub.compute.generic.ds_transforms import (
    horizonatal_flip,
    vertical_flip,
    shift_scale_rotate,
    transpose,
    gausse_noise,
    blur,
    random_brightness_contrast,
)


def test_transforms():
    url = "./tmp/test_transforms"
    schema = {
        "img": Image(shape=(100, 100, 3)),
        "mask": Mask(shape=(100, 100, 1)),
        "label": ClassLabel(num_classes=3),
        "id": "int8",
    }
    ds = hub.Dataset(url, mode="w", shape=(140,), schema=schema)
    for i, _ in enumerate(ds):
        ds[i]["img"] = np.ones((100, 100, 3))
        ds[i]["mask"] = np.ones((100, 100, 1))
        ds[i]["label"] = np.random.choice([0, 1, 2])
        ds[i]["id"] = i
    ds_1 = horizonatal_flip(ds, keys=["img"])
    ds_1.store(url + "_1")
    ds_1 = hub.Dataset(url + "_1")
    assert not np.all(ds[0]["img"] == ds_1[0]["img"].compute())
    ds_2 = gausse_noise(vertical_flip(ds, keys=["mask"])[0:20], keys=["mask"], mean=0.2)
    ds_2.store(url + "_2")
    ds_2 = hub.Dataset(url + "_2")
    assert len(ds_2) == 20
    assert not np.all(ds[0]["mask"] == ds_2[0]["mask"].compute())
    ds_3 = blur(ds[:100], keys=["img"], p=0.5)
    ds_3.store(url + "_3")
    ds_3 = hub.Dataset(url + "_3")
    assert len(ds_3) == 100
    assert np.any(
        sample["img"] != sample_1["img"].compute() for sample in ds for sample_1 in ds_3
    )
    ds_4 = hub.Dataset(url + "_3")
    ds_4 = transpose(
        random_brightness_contrast(ds_3, keys=["mask"], contrast_limit=0.7)[:30],
        keys=["mask"],
        p=0.5,
    )
    ds_4.store(url + "_4")
    ds_4 = hub.Dataset(url + "_4")
    assert not np.all(ds_3[0]["img"] == ds_4[0]["img"].compute())
    ds_1.delete()
    ds_2.delete()
    ds_3.delete()
    ds_4.delete()


def test_transforms_input():
    schema = {
        "label": ClassLabel(num_classes=3),
        "text": Text((None,), max_shape=(10,)),
    }
    url = "./tmp/test_transforms"
    ds = hub.Dataset(url, mode="w", shape=(5,), schema=schema)
    for i, _ in enumerate(ds):
        ds[i]["label"] = np.random.choice([0, 1, 2])
        ds[i]["text"] = "text_label"
    try:
        ds_1 = horizonatal_flip(ds, keys=["label"])
        ds_1.store(url + "_1")
    except Exception as e:
        assert isinstance(e, IndexError)
    try:
        ds_1 = horizonatal_flip(ds, keys=["text"])
        ds_1.store(url + "_1")
    except Exception as e:
        assert isinstance(e, AttributeError)
    ds.delete()
    ds_1.delete()


if __name__ == "__main__":
    test_transforms()
    test_transforms_input()
