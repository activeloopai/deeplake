from hub.util.callbacks import CallbackList
from hub.util.exceptions import MetaInvalidKey, MetaInvalidRequiredMetaKey
import pytest
from hub.core.meta.dataset_meta import DatasetMeta
from hub.core.meta.meta import Meta
import hub


TEST_META_KEY = "test_meta.json"


def test_meta(local_storage):
    meta = Meta(
        TEST_META_KEY,
        local_storage,
        {
            "list": CallbackList,
            "nested_list": CallbackList,
            "number": 9,
            "string": "uhhh",
        },
    )
    meta.list.append(1)
    meta.list.extend([5, 6])
    meta.nested_list.append([1, 2, 3])
    meta.nested_list.extend([[9, 8, 7], [0]])
    meta.nested_list.append({"05": 8})
    meta.custom_meta["stayin"] = "alive"  # custom_meta is implicit with every `Meta`
    del meta

    meta = Meta(TEST_META_KEY, local_storage)
    assert meta.list == [1, 5, 6]
    assert meta.nested_list == [[1, 2, 3], [9, 8, 7], [0], {"05": 8}]
    assert meta.custom_meta == {"stayin": "alive"}
    meta.list[1] = 99
    meta.list += [0]
    meta.string += "3"
    meta.number += 3
    meta.number = meta.number * 10
    meta.nested_list[1] = [4, 5, 99, 6]
    meta.nested_list[-1]["05"] -= 5
    meta.nested_list[-1]["5"] = 1
    meta.custom_meta.update({"ha": "ha", "haha": {"f": 9}})
    del meta

    meta = Meta(TEST_META_KEY, local_storage)
    assert meta.list == [1, 99, 6, 0]
    assert meta.nested_list == [[1, 2, 3], [4, 5, 99, 6], [0], {"05": 3, "5": 1}]
    assert meta.string == "uhhh3"
    assert meta.number == 120
    assert meta.custom_meta == {"stayin": "alive", "ha": "ha", "haha": {"f": 9}}
    assert meta.version == hub.__version__, "meta.version should be implicitly created"


def test_dataset_meta(local_storage):
    dataset_meta = DatasetMeta.create(local_storage)
    assert dataset_meta.tensors == []
    assert dataset_meta.custom_meta == {}
    assert dataset_meta.version == hub.__version__
    del dataset_meta

    dataset_meta = DatasetMeta.load(local_storage)
    dataset_meta.tensors.append("tensor1")
    del dataset_meta

    dataset_meta = DatasetMeta.load(local_storage)
    assert dataset_meta.tensors == ["tensor1"]


@pytest.mark.xfail(raises=MetaInvalidKey, strict=True)
def test_invalid_meta_key(local_storage):
    meta = Meta(TEST_META_KEY, local_storage, required_meta={})
    meta.some_key


@pytest.mark.xfail(raises=MetaInvalidRequiredMetaKey, strict=True)
def test_invalid_required_meta(local_storage):
    # "version" should not be passed into `required_meta` (auto-populated)
    meta = Meta(TEST_META_KEY, local_storage, required_meta={"version": hub.__version__})
