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

    ds.info.delete("test")
    assert len(ds.info) == 6

    ds.info.delete(["1_-+", "xyz"])
    assert len(ds.info) == 4

    ds.info.delete()
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

    t2.info.delete("key")
    assert len(t2.info) == 3

    t2.info.delete(["key2", "key3"])
    assert len(t2.info) == 1

    t2.info.delete()
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

    ds = local_ds_generator()

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
