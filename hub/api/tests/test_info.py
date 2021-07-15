def test_dataset(local_ds_generator):
    ds = local_ds_generator()

    assert len(ds.info) == 0

    ds.info.update(my_key=0)
    ds.info.update(my_key=1)

    ds.info.update(another_key="hi")
    ds.info.update({"another_key": "hello"})

    ds.info.update({"something": "aaaaa"}, something="bbbb")

    ds.info.update(test=[1, 2, "5"])

    with ds:
        ds.info.update({"test2": (1, 5, (1, "2"), [5, 6, (7, 8)])})
        ds.info.update(xyz="abc")

    ds.info.update({"1_-+", 5})

    ds = local_ds_generator()

    assert len(ds.info) == 7

    assert ds.info.another_key == "hello"
    assert ds.info.something == "bbbb"
    assert ds.info.test == (1, 2, "5"), "needs to convert to tuple"
    assert ds.info.test2 == (1, 5, (1, "2"), (5, 6, (7, 8)))
    assert ds.info.xyz == "abc"
    assert ds.info["1_-+"] == 5  # key can't be accessed with `.` syntax

    ds.info.update(test=[99])

    ds = local_ds_generator()

    assert len(ds.info) == 8
    assert ds.info.test == (99,)


def test_tensor(local_ds_generator):
    ds = local_ds_generator()

    t1 = ds.create_tensor("tensor1")
    t2 = ds.create_tensor("tensor2")

    assert len(t1.info) == 0
    assert len(t2.info) == 0

    t1.info.update(key=0)
    t2.info.update(key=1, key1=0)

    ds = local_ds_generator()

    t1 = ds.tensor1
    t2 = ds.tensor2

    assert len(t1.info) == 1
    assert len(t2.info) == 2

    assert t1.info.key == 0
    assert t2.info.key == 1
    assert t2.info.key1 == 0

    with ds:
        t1.info.update(key=99)

    ds = local_ds_generator()

    t1 = ds.tensor1
    t2 = ds.tensor2

    assert len(t1.info) == 2
    assert len(t2.info) == 2

    assert t1.info.key == 99


