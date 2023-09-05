import pytest


def test_dataset(local_ds_generator):
    ds = local_ds_generator()

    assert len(ds.info) == 0

    ds.info.update(my_key=0)
    ds.info.update(my_key=1)

    ds.info.update(another_key="hi")
    ds.info.update({"another_key": "hello"})

    ds.info.update({"something": "aaaaa"}, something="bbbb")

    ds.info.test = [1, 2, "5"]

    test_list = ds.info.test
    with ds:
        ds.info.update({"test2": (1, 5, (1, "2"), [5, 6, (7, 8)])})
        ds.info.update(xyz="abc")
        test_list.extend(["user made change without `update`"])

    ds.info.update({"1_-+": 5})
    assert len(ds.info) == 7

    ds = local_ds_generator()

    assert len(ds.info) == 7

    assert ds.info.another_key == "hello"
    assert ds.info.something == "bbbb"

    assert ds.info.test == [1, 2, "5", "user made change without `update`"]
    assert ds.info.test2 == [1, 5, [1, "2"], [5, 6, [7, 8]]]

    assert ds.info.xyz == "abc"
    assert ds.info["1_-+"] == 5  # key can't be accessed with `.` syntax

    ds.info.update(test=[99])

    ds = local_ds_generator()

    assert len(ds.info) == 7
    assert ds.info.test == [99]

    ds.info.pop("test")
    assert len(ds.info) == 6

    ds.info.pop("1_-+")
    ds.info.pop("xyz")
    assert len(ds.info) == 4

    ds.info.clear()
    assert len(ds.info) == 0


def test_tensor(local_ds_generator):
    ds = local_ds_generator()

    t1 = ds.create_tensor("tensor1")
    t2 = ds.create_tensor("tensor2")

    t1.info.update(key=0)
    t2.info.update(key=1, key1=0)
    t2.info.key2 = 2
    t2.info["key3"] = 3

    ds = local_ds_generator()

    t1 = ds.tensor1
    t2 = ds.tensor2

    assert len(t1.info) == 1
    assert len(t2.info) == 4

    assert t1.info.key == t1.info["key"] == 0
    assert t2.info.key == t2.info["key"] == 1
    assert t2.info.key1 == t2.info["key1"] == 0
    assert t2.info.key2 == t2.info["key2"] == 2
    assert t2.info.key3 == t2.info["key3"] == 3

    with ds:
        t1.info.update(key=99)

    ds = local_ds_generator()

    t1 = ds.tensor1
    t2 = ds.tensor2

    assert len(t1.info) == 1
    assert len(t2.info) == 4

    assert t1.info.key == 99

    t2.info.pop("key")
    assert len(t2.info) == 3

    t2.info.pop("key2")
    t2.info.pop("key3")
    assert len(t2.info) == 1

    t2.info.clear()
    assert len(t2.info) == 0


def test_update_reference_manually(local_ds_generator):
    """Right now synchronization can only happen when you call `info.update`."""

    ds = local_ds_generator()

    ds.info.update(key=[1, 2, 3])

    ds = local_ds_generator()

    l = ds.info.key
    assert l == [1, 2, 3]

    # un-registered update
    l.append(5)
    assert ds.info.key == [1, 2, 3, 5]

    ds = local_ds_generator()

    l = ds.info.key
    assert l == [1, 2, 3]

    # registered update
    l.append(99)
    ds.info.update()

    assert l == [1, 2, 3, 99]


def test_class_label(local_ds_generator):
    ds = local_ds_generator()
    ds.create_tensor("labels", htype="class_label", class_names=["a", "b", "c"])
    ds.create_tensor("labels2", htype="class_label")
    assert len(ds.labels.info) == 1
    assert len(ds.labels2.info) == 1
    assert (
        ds.labels.info.class_names == ds.labels.info["class_names"] == ["a", "b", "c"]
    )
    assert ds.labels2.info.class_names == ds.labels2.info["class_names"] == []
    ds.labels.info.class_names = ["c", "b", "a"]
    ds = local_ds_generator()
    assert len(ds.labels.info) == 1
    assert len(ds.labels2.info) == 1
    assert (
        ds.labels.info.class_names == ds.labels.info["class_names"] == ["c", "b", "a"]
    )
    assert ds.labels2.info.class_names == ds.labels2.info["class_names"] == []


def test_bbox(local_ds_generator):
    ds = local_ds_generator()
    ds.create_tensor("bboxes", htype="bbox", coords={"type": 0, "mode": 2})
    ds.create_tensor("bboxes1", htype="bbox", coords={"type": 1})
    ds.create_tensor("bboxes2", htype="bbox")
    assert len(ds.bboxes.info) == 1
    assert len(ds.bboxes2.info) == 1
    assert ds.bboxes.info.coords == ds.bboxes.info["coords"] == {"type": 0, "mode": 2}
    assert ds.bboxes1.info.coords == ds.bboxes1.info["coords"] == {"type": 1}
    assert ds.bboxes2.info.coords == ds.bboxes2.info["coords"] == {}
    ds.bboxes.info.coords = {"type": 3}
    ds = local_ds_generator()
    assert len(ds.bboxes.info) == 1
    assert len(ds.bboxes2.info) == 1
    assert ds.bboxes.info.coords == ds.bboxes.info["coords"] == {"type": 3}
    assert ds.bboxes2.info.coords == ds.bboxes2.info["coords"] == {}
    with pytest.raises(TypeError):
        ds.create_tensor("bboxes3", htype="bbox", coords=[1, 2, 3])

    with pytest.raises(KeyError):
        ds.create_tensor("bboxes4", htype="bbox", coords={"random": 0})


def test_info_new_methods(local_ds_generator):
    ds = local_ds_generator()
    ds.create_tensor("x")

    ds.info[0] = "hello"
    ds.info[1] = "world"
    assert len(ds.info) == 2
    assert set(ds.info.keys()) == {0, 1}
    assert 0 in ds.info
    assert 1 in ds.info

    assert ds.info[0] == "hello"
    assert ds.info[1] == "world"

    del ds.info[0]
    assert len(ds.info) == 1
    assert 1 in ds.info
    assert ds.info[1] == "world"

    for it in ds.info:
        assert it == 1

    ds.info.setdefault(0, "yo")
    assert len(ds.info) == 2
    assert 0 in ds.info
    assert 1 in ds.info
    assert ds.info[0] == "yo"
    assert ds.info[1] == "world"

    ds.info.popitem()
    assert len(ds.info) == 1
    assert 1 in ds.info
    assert ds.info[1] == "world"

    for k, v in ds.info.items():
        assert k == 1
        assert v == "world"

    for v in ds.info.values():
        assert v == "world"

    ds.info = {"a": "b"}
    assert len(ds.info) == 1
    assert "a" in ds.info
    assert ds.info["a"] == "b"

    ds.x.info = {"x": "y", "z": "w"}
    assert len(ds.x.info) == 2
    assert "x" in ds.x.info
    assert "z" in ds.x.info
    assert ds.x.info["x"] == "y"
    assert ds.x.info["z"] == "w"

    with pytest.raises(TypeError):
        ds.info = ["abc"]

    with pytest.raises(TypeError):
        ds.x.info = ["abc"]


def test_info_persistence_bug(local_ds_generator):
    ds = local_ds_generator()
    with ds:
        ds.create_tensor("xyz")
    ds.commit()
    ds.xyz.info.update(abc=123)
    assert ds.xyz.info.abc == 123
    ds = local_ds_generator()
    assert ds.xyz.info.abc == 123


def test_info_has_head_changes_bug(local_ds):
    with local_ds as ds:
        ds.info["tst"] = 42
    assert ds.has_head_changes == True
