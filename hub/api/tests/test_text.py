import hub
import pytest
from hub.tests.dataset_fixtures import enabled_non_gcs_datasets


def test_text(memory_ds):
    ds = memory_ds
    ds.create_tensor("text", htype="text")
    items = ["abcd", "efgh", "0123456"]
    with ds:
        for x in items:
            ds.text.append(x)
        ds.text.extend(items)
    assert ds.text.shape == (6, 1)
    for i in range(6):
        assert ds.text[i].numpy()[0] == items[i % 3]


@enabled_non_gcs_datasets
def test_text_transform(ds, scheduler="threaded"):
    ds.create_tensor("text", htype="text")

    @hub.compute
    def upload(some_str, ds):
        ds.text.append(some_str)
        return ds

    upload().eval(
        ["hi", "if ur reading this ur a nerd"], ds, num_workers=2, scheduler=scheduler
    )

    assert len(ds) == 2
    assert ds.text.data() == ["hi", "if ur reading this ur a nerd"]


@pytest.mark.parametrize(
    "args", [{}, {"sample_compression": "lz4"}, {"chunk_compression": "lz4"}]
)
def test_text_update(memory_ds, args):
    ds = memory_ds
    with ds:
        ds.create_tensor("x", htype="text", **args)
        for _ in range(10):
            ds.x.append("cat")
    for i in range(0, 10, 2):
        ds.x[i] = "flower"
    assert ds.x.data() == ["flower", "cat"] * 5
