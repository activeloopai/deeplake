import numpy as np
import pytest

from hub.util.exceptions import MergeMismatchError, MergeNotSupportedError


def test_merge(local_ds):
    with local_ds as ds:
        ds.create_tensor("image")
        ds.image.append(1)
        a = ds.commit()
        ds.image[0] = 2
        b = ds.commit()
        assert ds.image[0].numpy() == 2
        ds.checkout(a)
        ds.checkout("alt", create=True)
        ds.image[0] = 3
        assert ds.image[0].numpy() == 3
        f = ds.commit()

        ds.checkout("main")
        assert ds.image[0].numpy() == 2

        ds.merge(f, conflict_resolution="theirs")
        assert ds.image[0].numpy() == 3

        ds.image[0] = 4
        assert ds.image[0].numpy() == 4
        d = ds.commit()
        ds.checkout("alt")
        assert ds.image[0].numpy() == 3

        ds.image[0] = 0
        assert ds.image[0].numpy() == 0
        g = ds.commit()

        ds.merge("main", conflict_resolution="theirs")
        assert ds.image[0].numpy() == 4

        ds.image[0] = 5
        assert ds.image[0].numpy() == 5
        ds.image.append(10)
        i = ds.commit()

        ds.image[0] = 6
        assert ds.image[0].numpy() == 6

        ds.checkout("main")
        assert ds.image[0].numpy() == 4

        ds.merge("alt", conflict_resolution="theirs")
        assert ds.image[0].numpy() == 6
        assert ds.image[1].numpy() == 10


def test_complex_merge(local_ds):
    with local_ds as ds:
        ds.create_tensor("image")
        ds.create_tensor("label")
        for i in range(10):
            ds.image.append(i * np.ones((200, 200, 3)))
            ds.label.append(i)
        a = ds.commit("added 10 images and labels")
        ds.checkout("other", create=True)
        for i in range(10, 15):
            ds.image.append(i * np.ones((200, 200, 3)))
        for i in range(3, 7):
            ds.label[i] = 2 * i

        b = ds.commit("added 5 more images and changed 4 labels")
        assert len(ds.image) == 15
        assert len(ds.label) == 10
        commit_id = ds.commit_id
        ds.merge("main")
        assert len(ds.image) == 15
        assert len(ds.label) == 10
        assert ds.commit_id == commit_id

        ds.checkout("main")
        ds.merge("other")
        assert len(ds.image) == 15
        assert len(ds.label) == 10
        for i in range(10):
            target = 1 if i not in range(3, 7) else 2
            assert ds.label[i].numpy() == target * i
        for i in range(15):
            np.testing.assert_array_equal(
                ds.image[i].numpy(), i * np.ones((200, 200, 3))
            )

        ds.checkout("other")
        ds.merge("main")

        assert len(ds.image) == 15
        assert len(ds.label) == 10
        for i in range(10):
            target = 1 if i not in range(3, 7) else 2
            assert ds.label[i].numpy() == target * i
        for i in range(15):
            np.testing.assert_array_equal(
                ds.image[i].numpy(), i * np.ones((200, 200, 3))
            )


def test_merge_not_supported(local_ds):
    with local_ds as ds:
        ds.create_tensor("image", create_id_tensor=False)
        ds.create_tensor("label")
        for i in range(10):
            ds.image.append(i * np.ones((200, 200, 3)))
            ds.label.append(i)
        a = ds.commit("added 10 images and labels")
        ds.checkout("other", create=True)
        for i in range(10, 15):
            ds.image.append(i * np.ones((200, 200, 3)))
        for i in range(3, 7):
            ds.label[i] = 2 * i

        b = ds.commit("added 5 more images and changed 4 labels")
        assert len(ds.image) == 15
        assert len(ds.label) == 10
        ds.checkout("main")
        with pytest.raises(MergeNotSupportedError):
            ds.merge("other")


def test_tensor_mismatch(local_ds):
    with local_ds as ds:
        ds.create_tensor("image")
        ds.checkout("alt", create=True)
        ds.create_tensor("xyz", htype="bbox")
        ds.checkout("main")
        ds.create_tensor("xyz", htype="class_label")
        with pytest.raises(MergeMismatchError):
            ds.merge("alt")


def test_new_tensor_creation_merge(local_ds):
    with local_ds as ds:
        ds.create_tensor("image")
        ds.checkout("alt", create=True)
        ds.create_tensor("xyz")
        for i in range(100):
            ds.xyz.append(i)
        ds.commit()
        ds.checkout("main")
        ds.merge("alt")
        assert "xyz" in ds.tensors
        assert len(ds.xyz) == 100
        for i in range(100):
            assert ds.xyz[i].numpy() == i

        # merging twice to confirm, that extra items are not added
        ds.merge("alt")
        assert "xyz" in ds.tensors
        assert len(ds.xyz) == 100
        for i in range(100):
            assert ds.xyz[i].numpy() == i


def test_tensor_deletion_merge(local_ds):
    with local_ds as ds:
        ds.create_tensor("image")
        ds.create_tensor("label")
        for i in range(10):
            ds.image.append(i * np.ones((200, 200, 3)))
            ds.label.append(i)

        a = ds.commit("added 10 images and labels")
        ds.checkout("other", create=True)

        ds.delete_tensor("image")
        ds.label.append(10)
        ds.commit()
        ds.checkout("main")
        ds.merge("other")
        assert "image" in ds.tensors
        for i in range(10):
            np.testing.assert_array_equal(
                ds.image[i].numpy(), i * np.ones((200, 200, 3))
            )
            assert ds.label[i].numpy() == i
        assert ds.label[10].numpy() == 10

        ds.merge("other", delete_removed_tensors=True)
        assert "image" not in ds.tensors
        assert "label" in ds.tensors
        for i in range(11):
            assert ds.label[i].numpy() == i
