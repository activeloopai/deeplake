import hub
import pytest


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
