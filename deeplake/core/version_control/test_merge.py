import numpy as np
import pytest
import deeplake

from deeplake.util.testing import assert_array_equal, compare_version_info
from deeplake.util.version_control import rebuild_version_info
from deeplake.util.exceptions import (
    MergeConflictError,
    MergeMismatchError,
    MergeNotSupportedError,
)


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


def test_tensor_revival(local_ds):
    with local_ds as ds:
        ds.create_tensor("image")
        ds.create_tensor("label")

        for i in range(10):
            ds.image.append(i * np.ones((200, 200, 3)))
            ds.label.append(i)

        a = ds.commit("added 10 images and labels")
        ds.delete_tensor("image")
        ds.delete_tensor("label")

        ds.checkout(a)
        ds.checkout("other", create=True)

        assert "image" in ds.tensors
        assert "label" in ds.tensors
        for i in range(10, 15):
            ds.label.append(i)
        ds.commit()
        ds.checkout("main")
        assert "image" not in ds.tensors
        assert "label" not in ds.tensors

        ds.merge("other")

        # not revived as this had no changes in other
        assert "image" not in ds.tensors

        # revived as this had changes in other
        assert "label" in ds.tensors
        for i in range(15):
            assert ds.label[i].numpy() == i


def test_conflicts(local_ds):
    with local_ds as ds:
        ds.create_tensor("image")
        for i in range(10):
            ds.image.append(i * np.ones((200, 200, 3)))

        a = ds.commit("added 10 images")
        ds.checkout("other", create=True)
        for i in range(10, 15):
            ds.image.append(i * np.ones((200, 200, 3)))
        ds.image[4] = 25 * np.ones((200, 200, 3))

        b = ds.commit("added 5 more images and changed 4th")
        ds.checkout("main")
        ds.image[4] = 50 * np.ones((200, 200, 3))

        with pytest.raises(MergeConflictError):
            ds.merge("other")

        ds.merge("other", conflict_resolution="theirs")
        np.testing.assert_array_equal(ds.image[4].numpy(), 25 * np.ones((200, 200, 3)))


def test_rename_merge(local_ds):
    with local_ds as ds:
        # no conflicts
        ds.create_tensor("abc")
        ds.abc.append([1, 2, 3])
        ds.commit()
        ds.checkout("alt", create=True)
        ds.rename_tensor("abc", "xyz")
        ds.xyz.append([3, 4, 5])
        ds.commit()
        ds.checkout("main")
        ds.merge("alt")
        assert "abc" not in ds.tensors
        np.testing.assert_array_equal(ds.xyz.numpy(), np.array([[1, 2, 3], [3, 4, 5]]))

        # tensor with same name on main
        ds.create_tensor("red")
        ds.red.append([2, 3, 4])
        ds.commit()
        ds.checkout("alt2", create=True)
        ds.rename_tensor("red", "blue")
        ds.blue.append([1, 0, 0])
        ds.commit()
        ds.checkout("main")
        ds.create_tensor("blue")
        ds.blue.append([0, 0, 1])
        ds.commit()
        with pytest.raises(MergeConflictError):
            ds.merge("alt2")
        # resolve
        ds.merge("alt2", force=True)
        np.testing.assert_array_equal(ds.red.numpy(), np.array([[2, 3, 4]]))
        np.testing.assert_array_equal(
            ds.blue.numpy(), np.array([[0, 0, 1], [2, 3, 4], [1, 0, 0]])
        )
        ds.delete_tensor("blue")
        ds.commit()

        # rename to same name (no conflict)
        ds.rename_tensor("red", "blue")
        ds.merge("alt2")
        np.testing.assert_array_equal(ds.blue.numpy(), np.array([[2, 3, 4], [1, 0, 0]]))

        # renamed on both branches
        ds.create_tensor("image")
        ds.create_tensor("video")
        ds.image.append([1, 2, 3])
        ds.video.append([1, 0, 0])
        ds.commit()
        ds.checkout("alt3", create=True)
        ds.rename_tensor("image", "images")
        ds.rename_tensor("video", "videos")
        ds.images.append([3, 4, 5])
        ds.videos.append([0, 1, 0])
        ds.commit()
        ds.checkout("main")
        ds.rename_tensor("image", "cat_images")
        ds.rename_tensor("video", "sample_videos")
        ds.cat_images.append([5, 6, 7])
        ds.sample_videos.append([7, 8, 9])
        ds.commit()
        with pytest.raises(MergeConflictError):
            ds.merge("alt3")
        # resolve (through rename and force)
        ds.checkout("alt3")
        ds.rename_tensor("images", "cat_images")
        ds.commit()
        ds.checkout("main")
        ds.merge("alt3", force=True)
        assert "image" not in ds.tensors
        assert "images" not in ds.tensors
        np.testing.assert_array_equal(
            ds.cat_images.numpy(), np.array([[1, 2, 3], [5, 6, 7], [3, 4, 5]])
        )
        np.testing.assert_array_equal(
            ds.videos.numpy(), np.array([[1, 0, 0], [0, 1, 0]])
        )
        np.testing.assert_array_equal(
            ds.sample_videos.numpy(), np.array([[1, 0, 0], [7, 8, 9]])
        )

        # tensor missing from head
        ds.create_tensor("t1")
        ds.t1.append([1, 2, 3])
        ds.commit()
        ds.checkout("alt4", create=True)
        ds.rename_tensor("t1", "t2")
        ds.t2.append([4, 5, 6])
        ds.commit()
        ds.checkout("main")
        ds.delete_tensor("t1")
        ds.commit()
        with pytest.raises(MergeConflictError):
            ds.merge("alt4")


def test_clear_merge(local_ds):
    with local_ds as ds:
        ds.create_tensor("abc")
        ds.abc.append([1, 2, 3])
        a = ds.commit()

        ds.checkout("alt", create=True)
        ds.abc.append([2, 3, 4])
        b = ds.commit()
        ds.abc.clear()
        c = ds.commit()

        ds.checkout("main")
        ds.abc.append([5, 6, 3])
        d = ds.commit()
        ds.merge("alt")

        np.testing.assert_array_equal(ds.abc.numpy(), np.array([[1, 2, 3], [5, 6, 3]]))


def test_merge_pop(local_ds):
    with local_ds as ds:
        ds.create_tensor("abc")
        ds.abc.extend([1, 2])
        a = ds.commit()

        ds.checkout("alt", create=True)
        ds.abc.extend([3, 4])
        b = ds.commit()
        ds.abc.pop(1)
        c = ds.commit()

        ds.checkout("main")
        ds.abc.append(5)
        d = ds.commit()
        ds.merge("alt")

        np.testing.assert_array_equal(
            ds.abc.numpy().squeeze(), np.array([1, 2, 5, 3, 4])
        )

        ds.checkout("alt")
        ds.abc[2] = 8  # change value from 4 to 8
        np.testing.assert_array_equal(ds.abc.numpy().squeeze(), np.array([1, 3, 8]))
        e = ds.commit()
        ds.checkout("main")
        ds.abc.pop(4)  # remove value 4
        np.testing.assert_array_equal(ds.abc.numpy().squeeze(), np.array([1, 2, 5, 3]))
        ds.merge("alt")
        np.testing.assert_array_equal(
            ds.abc.numpy().squeeze(), np.array([1, 2, 5, 3, 8])
        )


def test_merge_class_labels(local_ds):
    with local_ds as ds:
        ds.create_tensor("labels", "class_label")
        ds.labels.extend(["a", "b", "c", "d"])
        ds.commit()

        ds.checkout("alt", create=True)
        ds.labels.extend(["e", "f", "a"])
        ds.commit()

        ds.checkout("main")
        ds.labels.extend(["g", "f", "h"])
        ds.merge("alt")

        np.testing.assert_array_equal(
            np.array(ds.labels.data()["text"]).squeeze(),
            ["a", "b", "c", "d", "g", "f", "h", "e", "f", "a"],
        )
        assert set(ds.labels.info["class_names"]) == {
            "a",
            "b",
            "c",
            "d",
            "e",
            "f",
            "g",
            "h",
        }


def test_merge_class_labels_no_classnames(local_ds):
    with local_ds as ds:
        ds.create_tensor("labels", "class_label")
        ds.labels.extend([0, 1, 2, 3])
        ds.commit()

        ds.checkout("alt", create=True)
        ds.labels.extend([4, 5, 0])
        ds.commit()

        ds.checkout("main")
        ds.labels.extend([6, 5, 7])
        ds.merge("alt")

        np.testing.assert_array_equal(
            np.array(ds.labels.numpy()).squeeze(),
            [0, 1, 2, 3, 6, 5, 7, 4, 5, 0],
        )


def test_merge_class_labels_different_class_names(memory_ds):
    with memory_ds as ds:
        ds.checkout("alt1", create=True)
        ds.create_tensor("labels", "class_label")
        ds.labels.extend(["a", "b", "c", "d"])
        ds.checkout("main")
        ds.checkout("alt2", create=True)
        ds.create_tensor("labels", "class_label")
        ds.labels.extend(["d", "e", "f", "g"])
        ds.merge("alt1")
        np.testing.assert_array_equal(
            np.array(ds.labels.numpy()).squeeze(),
            [0, 1, 2, 3, 4, 5, 6, 0],
        )


def test_merge_class_labels_subset_class_names(memory_ds):
    with memory_ds as ds:
        ds.checkout("alt1", create=True)
        ds.create_tensor("labels", "class_label")
        ds.labels.extend(["a", "b", "c", "d"])
        ds.checkout("main")
        ds.checkout("alt2", create=True)
        ds.create_tensor("labels", "class_label")
        ds.labels.extend(["a", "b", "c"])
        ds.merge("alt1")
        np.testing.assert_array_equal(
            np.array(ds.labels.numpy()).squeeze(),
            [0, 1, 2, 0, 1, 2, 3],
        )


def test_merge_class_labels_subset_class_names_2(memory_ds):
    with memory_ds as ds:
        ds.checkout("alt1", create=True)
        ds.create_tensor("labels", "class_label")
        ds.labels.extend(["a", "b", "c"])
        ds.checkout("main")
        ds.checkout("alt2", create=True)
        ds.create_tensor("labels", "class_label")
        ds.labels.extend(["a", "b", "c", "d"])
        ds.merge("alt1")
        np.testing.assert_array_equal(
            np.array(ds.labels.numpy()).squeeze(),
            [0, 1, 2, 3, 0, 1, 2],
        )


def test_merge_sequence_htype(memory_ds):
    with memory_ds as ds:
        ds.create_tensor("abc", htype="sequence")
        ds.abc.extend([[1, 2, 3], [4, 5], [6, 7, 8, 9]])
        ds.commit()

        ds.checkout("alt", create=True)
        ds.abc.extend([[1, 2, 3, 4], [1, 2], [7, 5, 4, 2]])

        ds.checkout("main")
        ds.merge("alt")
        ds.create_tensor("expected")
        ds.expected.extend([[1, 2, 3], [4, 5], [6, 7, 8, 9]])
        ds.expected.extend([[1, 2, 3, 4], [1, 2], [7, 5, 4, 2]])
        for x, y in zip(ds.abc.numpy(aslist=True), ds.expected.numpy(aslist=True)):
            assert_array_equal(x, y)


def test_merge_tiled(memory_ds):
    arr = np.random.random((3, 1713, 1918))
    with memory_ds as ds:
        ds.create_tensor("abc")
        ds.abc.extend(arr)
        assert list(ds.abc.chunk_engine.tile_encoder.entries.keys()) == list(range(3))
        ds.commit()
        ds.checkout("alt", create=True)
        ds.abc.extend(arr * 0.3 + 0.7)
        ds.checkout("main")
        ds.merge("alt")
        ds.create_tensor("expected")
        ds.expected.extend(arr)
        ds.expected.extend(arr * 0.3 + 0.7)
        assert_array_equal(ds.abc.numpy(aslist=True), ds.expected.numpy(aslist=True))


def test_merge_tiled_new_tensor(memory_ds):
    arr = np.random.random((3, 1394, 1503))
    with memory_ds as ds:
        ds.checkout("alt", create=True)
        ds.create_tensor("abc")
        ds.abc.extend(arr)
        assert list(ds.abc.chunk_engine.tile_encoder.entries.keys()) == list(range(3))
        ds.checkout("main")
        ds.merge("alt")
        ds.create_tensor("expected")
        ds.expected.extend(arr)
        assert_array_equal(ds.abc.numpy(aslist=True), ds.expected.numpy(aslist=True))


def test_merge_groups(memory_ds):
    with memory_ds as ds:
        ds.create_tensor("a")
        ds.checkout("alt", create=True)
        ds.create_tensor("b/c")
        ds.checkout("main")
        ds.merge("alt")
        assert list(ds.groups) == ["b"]


def test_merge_linked(memory_ds, cat_path):
    with memory_ds as ds:
        ds.create_tensor("abc", htype="link[image]", sample_compression="jpg")
        ds.add_creds_key("creds1")
        ds.populate_creds("creds1", {})
        ds.abc.extend([deeplake.link(cat_path, creds_key="creds1") for _ in range(10)])
        ds.checkout("alt", create=True)
        ds.add_creds_key("creds2")
        ds.populate_creds("creds2", {})
        ds.abc.extend([deeplake.link(cat_path, creds_key="creds2") for _ in range(5)])
        ds.checkout("main")
        ds.merge("alt")

        assert len(ds.abc) == 15
        assert set(ds.link_creds.creds_keys) == {"creds1", "creds2"}


def test_get_required_chunks(memory_ds):
    from deeplake.util.merge import _get_required_chunks_for_range as get_chunks

    half_chunk = np.random.randn(500, 1000)  # 4MB
    ds = memory_ds

    abc = ds.create_tensor("abc")
    abc.extend([half_chunk, half_chunk])

    assert get_chunks(abc, 0, 1) == (None, (0, 1), None)
    assert get_chunks(abc, 0, 2) == ((0, 1), None, None)

    abc.extend([half_chunk, half_chunk])

    assert get_chunks(abc, 1, 3) == (None, (1, 3), None)
    assert get_chunks(abc, 0, 3) == ((0, 1), None, (2, 3))
    assert get_chunks(abc, 1, 4) == ((1, 2), (1, 2), None)

    abc.extend([half_chunk, half_chunk])

    assert get_chunks(abc, 0, 5) == ((0, 2), None, (4, 5))
    assert get_chunks(abc, 1, 6) == ((1, 3), (1, 2), None)
    assert get_chunks(abc, 1, 5) == ((1, 2), (1, 2), (4, 5))
    assert get_chunks(abc, 0, 6) == ((0, 3), None, None)


def test_merge_with_padding(memory_ds):
    with memory_ds as ds:
        ds.create_tensor("x")
        cid = ds.commit()
        ds.checkout("branch1", create=True)
        ds.x[100] = 2
        for i in range(0, 100):
            assert ds.x.chunk_engine.pad_encoder.is_padded(i), i
        ds.x[200] = 3
        ds.x[300] = 4
        ds.checkout(cid)
        ds.checkout("branch2", create=True)
        ds.x[150] = 10
        ds.x[250] = 20
        ds.x[350] = 30
        ds.checkout("branch1")
        ds.merge("branch2")
        assert len(ds.x) == 304


def test_merge_with_pop(memory_ds):
    with memory_ds as ds:
        ds.create_tensor("x")
        ds.x.extend([1, 2, 3, 4, 5])
        cid = ds.commit()
        ds.checkout("branch1", create=True)
        ds.pop(2)
        ds.checkout(cid)
        ds.checkout("branch2", create=True)
        ds.pop(3)
        ds.x.append(6)
        ds.checkout("branch1")
        ds.merge("branch2")
        np.testing.assert_array_equal(ds.x.numpy().flatten(), [1, 2, 4, 5, 6])
