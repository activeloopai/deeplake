from datasets import load_dataset  # type: ignore
from datasets import Dataset  # type: ignore
from hub.integrations.huggingface import from_huggingface
from hub.integrations.huggingface.huggingface import _is_seq_convertible
from numpy.testing import assert_array_equal


def test_before_split():
    ds = load_dataset("glue", "mrpc")
    hub_ds = from_huggingface(ds, "mem://xyz")

    splits = ds.keys()
    columns = ds["train"].column_names

    assert set(hub_ds.meta.tensors) == {
        f"{split}/{column}" for split in splits for column in columns
    }

    for split in splits:
        for column in columns:
            assert_array_equal(
                hub_ds[f"{split}/{column}"].numpy().reshape(-1), ds[split][column]
            )


def test_split():
    ds = load_dataset("glue", "mrpc", split="train[:5%]")
    hub_ds = from_huggingface(ds, "mem://xyz")

    assert hub_ds.meta.tensors == ds.column_names

    for column in ds.column_names:
        assert_array_equal(hub_ds[column].numpy().reshape(-1), ds[column])


def test_seq_with_dict():
    ds = load_dataset("squad", split="train[:5%]")
    hub_ds = from_huggingface(ds, "mem://xyz")

    keys = set(ds.column_names) - {"answers"} | {"answers/text", "answers/answer_start"}

    assert set(hub_ds.meta.tensors) == keys

    for key in ("id", "title", "context", "question"):
        assert_array_equal(hub_ds[key].numpy().reshape(-1), ds[key])

    answers = {"text": [], "answer_start": []}
    for answer in ds["answers"]:
        answers["text"].extend(answer["text"])
        answers["answer_start"].extend(answer["answer_start"])

    assert_array_equal(hub_ds["answers/text"].numpy().reshape(-1), answers["text"])
    assert_array_equal(
        hub_ds["answers/answer_start"].numpy().reshape(-1), answers["answer_start"]
    )


def test_seq():
    arr1 = [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]
    arr2 = [[[5, 6], [7, 8]], [[9, 10], [11, 12]]]

    data = {"id": [0, 1], "seq": [arr1, arr2]}
    ds = Dataset.from_dict(data)

    hub_ds = from_huggingface(ds, "mem://xyz")

    assert set(hub_ds.meta.tensors) == {"id", "seq"}
    assert_array_equal(hub_ds["seq"], [arr1, arr2])

    arr = [["abcd"], ["efgh"]]

    data = {"id": [0, 1], "seq": arr}
    ds = Dataset.from_dict(data)

    assert not _is_seq_convertible(ds.features["seq"])
