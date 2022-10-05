import deeplake
import numpy as np
import pytest
from unittest.mock import patch


all_non_image_compressions = pytest.mark.parametrize(
    "compression",
    [
        {"sample_compression": None},
        {"sample_compression": "lz4"},
        {"chunk_compression": "lz4"},
    ],
)


@patch("deeplake.constants._ENABLE_RANDOM_ASSIGNMENT", True)
@all_non_image_compressions
@pytest.mark.parametrize("insert_first", [True, False])
def test_insertion_array(memory_ds, compression, insert_first):
    with memory_ds as ds:
        ds.create_tensor("abc", **compression)
        first = np.random.rand(100, 100, 3)
        tenth = np.random.rand(100, 100, 3)
        empty_sample = np.random.rand(0, 0, 0)
        if insert_first:
            ds.abc.append(first)
        ds.abc[10] = tenth

        if insert_first:
            np.testing.assert_array_equal(ds.abc[0].numpy(), first)
        else:
            np.testing.assert_array_equal(ds.abc[0].numpy(), empty_sample)

        for i in range(1, 10):
            np.testing.assert_array_equal(ds.abc[i].numpy(), empty_sample)
        np.testing.assert_array_equal(ds.abc[10].numpy(), tenth)


@patch("deeplake.constants._ENABLE_RANDOM_ASSIGNMENT", True)
@pytest.mark.parametrize("sample_compression", ["png", "jpeg"])
@pytest.mark.parametrize("insert_first", [True, False])
def test_insertion_array_img_compressed(memory_ds, sample_compression, insert_first):
    with memory_ds as ds:
        ds.create_tensor("abc", sample_compression=sample_compression)
        first = np.random.randint(0, 256, (200, 300, 3), dtype=np.uint8)
        tenth = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        empty_sample = np.random.randint(0, 256, (0, 0, 0), dtype=np.uint8)
        if insert_first:
            ds.abc.append(first)
        ds.abc[10] = tenth

        if insert_first:
            if sample_compression == "png":
                np.testing.assert_array_equal(ds.abc[0].numpy(), first)
            else:
                assert ds.abc[0].numpy().shape == first.shape
        else:
            np.testing.assert_array_equal(ds.abc[0].numpy(), empty_sample)

        for i in range(1, 10):
            np.testing.assert_array_equal(ds.abc[i].numpy(), empty_sample)

        if sample_compression == "png":
            np.testing.assert_array_equal(ds.abc[10].numpy(), tenth)
        else:
            assert ds.abc[10].numpy().shape == tenth.shape


@patch("deeplake.constants._ENABLE_RANDOM_ASSIGNMENT", True)
@all_non_image_compressions
@pytest.mark.parametrize("insert_first", [True, False])
def test_insertion_json(memory_ds, compression, insert_first):
    with memory_ds as ds:
        ds.create_tensor("abc", **compression)
        first = {"a": 1, "b": 2}
        tenth = {"a": 3, "b": 4}
        empty_sample = {}
        if insert_first:
            ds.abc.append(first)
        ds.abc[10] = tenth

        if insert_first:
            assert ds.abc[0].numpy()[0] == first
        else:
            assert ds.abc[0].numpy()[0] == empty_sample

        for i in range(1, 10):
            assert ds.abc[i].numpy() == empty_sample

        assert ds.abc[10].numpy() == tenth


@patch("deeplake.constants._ENABLE_RANDOM_ASSIGNMENT", True)
@all_non_image_compressions
@pytest.mark.parametrize("insert_first", [True, False])
def test_insertion_text(memory_ds, compression, insert_first):
    with memory_ds as ds:
        ds.create_tensor("abc", **compression)
        first = "hi"
        tenth = "if ur reading this ur a nerd"
        empty_sample = ""
        if insert_first:
            ds.abc.append(first)
        ds.abc[10] = tenth

        if insert_first:
            assert ds.abc[0].numpy()[0] == first
        else:
            assert ds.abc[0].numpy()[0] == empty_sample

        for i in range(1, 10):
            assert ds.abc[i].numpy() == empty_sample

        assert ds.abc[10].numpy() == tenth


@patch("deeplake.constants._ENABLE_RANDOM_ASSIGNMENT", True)
@all_non_image_compressions
@pytest.mark.parametrize("insert_first", [True, False])
def test_insertion_list(memory_ds, compression, insert_first):
    with memory_ds as ds:
        ds.create_tensor("abc", **compression, htype="list")
        first = [1, 2, 3]
        tenth = [4, 5, 6]
        empty_sample = np.array([], dtype="object")
        if insert_first:
            ds.abc.append(first)
        ds.abc[10] = tenth

        if insert_first:
            np.testing.assert_array_equal(ds.abc[0].numpy(), np.array(first))
        else:
            np.testing.assert_array_equal(ds.abc[0].numpy(), empty_sample)

        for i in range(1, 10):
            np.testing.assert_array_equal(ds.abc[i].numpy(), empty_sample)

        np.testing.assert_array_equal(ds.abc[10].numpy(), np.array(tenth))


@patch("deeplake.constants._ENABLE_RANDOM_ASSIGNMENT", True)
def test_updation_bug(memory_ds):
    with memory_ds as ds:
        labels = ds.create_tensor("labels", "class_label")
        labels[0] = [0, 1]
        np.testing.assert_array_equal(labels[0].numpy(), [0, 1])
        labels[0] = [1, 2]
        np.testing.assert_array_equal(labels[0].numpy(), [1, 2])


@patch("deeplake.constants._ENABLE_RANDOM_ASSIGNMENT", True)
@pytest.mark.parametrize("insert_first", [False, True])
def test_insertion_link(memory_ds, insert_first, cat_path, flower_path):
    with memory_ds as ds:
        ds.create_tensor("abc", htype="link[image]", sample_compression="jpg")
        first = deeplake.link(cat_path)
        tenth = deeplake.link(flower_path)
        empty_sample = np.ones((0,))
        if insert_first:
            ds.abc.append(first)
        ds.abc[10] = tenth

        if insert_first:
            assert ds.abc[0].numpy().shape == (900, 900, 3)
        else:
            np.testing.assert_array_equal(ds.abc[0].numpy(), empty_sample)

        for i in range(1, 10):
            np.testing.assert_array_equal(ds.abc[i].numpy(), empty_sample)

        assert ds.abc[10].numpy().shape == (513, 464, 4)


# removing this test until we fix sequence updates
# @patch("deeplake.constants._ENABLE_RANDOM_ASSIGNMENT", True)
# @pytest.mark.parametrize("insert_first", [True])
# def test_insertion_sequence(memory_ds, insert_first):
#     with memory_ds as ds:
#         ds.create_tensor("abc", htype="sequence")
#         first = [[1, 2, 3, 4], [5, 6, 7, 8]]
#         tenth = [[9, 10, 11, 12], [13, 14, 15, 16], [17, 18, 19, 20]]
#         empty_sample = np.ones((0, 0))
#         if insert_first:
#             ds.abc.append(first)
#         ds.abc[10] = tenth
#         ds.abc[0].numpy()

#         if insert_first:
#             np.testing.assert_array_equal(ds.abc[0].numpy(), first)
#         else:
#             np.testing.assert_array_equal(ds.abc[0].numpy(), empty_sample)

#         for i in range(1, 10):
#             np.testing.assert_array_equal(ds.abc[i].numpy(), empty_sample)

#         np.testing.assert_array_equal(ds.abc[10].numpy(), tenth)
