import hub


def test_dynamic_array():
    conn = hub.fs("./data/cache").connect()
    arr = conn.array_create(
        "test/dynamic_array_3",
        shape=(10, 8, 4, 12),
        chunk=(5, 4, 2, 6),
        dtype="uint8",
        dsplit=2,
    )
    arr.darray[0:10, 0:5] = (2, 12)
    arr.darray[0:10, 5:8] = (6, 14)

    assert arr[5, 3, :, :].shape == (2, 12)
    assert arr[5, 6, :, :].shape == (6, 14)
    assert arr[5, 4:6, :, :].shape == (2, 6, 14)
