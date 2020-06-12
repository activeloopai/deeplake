import hub


def test_dataset():
    datahub = hub.fs("./data/cache").connect()
    x = datahub.array(
        name="test/example:input", shape=(100, 25, 25), chunk=(20, 5, 5), dtype="uint8"
    )
    y = datahub.array(
        name="test/example:label", shape=(100, 4), chunk=(20, 2), dtype="uint8"
    )

    ds = datahub.dataset(
        components={"input": x, "label": y}, name="test/dataset:train3"
    )
    assert ds[0]["input"].shape == (25, 25)
    assert ds["input"].shape[0] == 100  # return single array
    assert ds["label", 0].mean() == 0  # equivalent ds['train'][0]


def test_load_dataset():
    datahub = hub.fs("./data/cache").connect()
    ds = datahub.open(name="test/dataset:train3")  # return the dataset object
    assert list(map(lambda x: x[0], ds.items())) == ["input", "label"]
    assert ds["label", 0].mean() == 0
