from hub.constants import DEFAULT_HTYPE
from hub.core.meta.tensor_meta import TensorMeta
from hub.core.meta.index_meta import IndexMeta
from hub.util.exceptions import (
    MetaInvalidKey,
    MetaInvalidRequiredMetaKey,
    TensorMetaInvalidHtype,
    TensorMetaInvalidHtypeOverwriteKey,
)
import pytest
from hub.core.meta.dataset_meta import DatasetMeta
from hub.core.meta.meta import Meta
from hub.core.tests.common import (
    parametrize_all_storages_and_caches,
)
import hub


TEST_META_KEY = "test_meta.json"


@parametrize_all_storages_and_caches
def test_meta(storage):
    meta = Meta(
        TEST_META_KEY,
        storage,
        {
            "list": [],
            "nested_list": [],
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

    meta = Meta(TEST_META_KEY, storage)
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

    meta = Meta(TEST_META_KEY, storage)
    assert meta.list == [1, 99, 6, 0]
    assert meta.nested_list == [[1, 2, 3], [4, 5, 99, 6], [0], {"05": 3, "5": 1}]
    assert meta.string == "uhhh3"
    assert meta.number == 120
    assert meta.custom_meta == {"stayin": "alive", "ha": "ha", "haha": {"f": 9}}
    assert meta.version == hub.__version__, "meta.version should be implicitly created"


@parametrize_all_storages_and_caches
def test_dataset_meta(storage):
    dataset_meta = DatasetMeta.create(storage)
    assert dataset_meta.tensors == []
    assert dataset_meta.custom_meta == {}
    assert dataset_meta.version == hub.__version__
    del dataset_meta

    dataset_meta = DatasetMeta.load(storage)
    dataset_meta.tensors.append("tensor1")
    del dataset_meta

    dataset_meta = DatasetMeta.load(storage)
    assert dataset_meta.tensors == ["tensor1"]


@parametrize_all_storages_and_caches
def test_tensor_meta(storage):
    tensor_meta = TensorMeta.create(TEST_META_KEY, storage)
    assert tensor_meta.htype == DEFAULT_HTYPE
    assert tensor_meta.dtype == None
    assert tensor_meta.length == 0
    assert tensor_meta.min_shape == []
    assert tensor_meta.max_shape == []
    assert tensor_meta.custom_meta == {}
    tensor_meta.length += 10
    tensor_meta.min_shape = [1, 2, 3]
    del tensor_meta

    tensor_meta = TensorMeta.load(TEST_META_KEY, storage)
    assert tensor_meta.length == 10
    assert tensor_meta.min_shape == [1, 2, 3]
    tensor_meta.min_shape[2] = 99
    tensor_meta.length += 1
    del tensor_meta

    tensor_meta = TensorMeta.load(TEST_META_KEY, storage)
    assert tensor_meta.min_shape == [1, 2, 99]
    assert tensor_meta.length == 11


@parametrize_all_storages_and_caches
def test_tensor_meta_htype_overwrite(storage):
    tensor_meta = TensorMeta.create(TEST_META_KEY, storage, dtype="bool")
    del tensor_meta

    tensor_meta = TensorMeta.load(TEST_META_KEY, storage)
    assert tensor_meta.dtype == "bool"


@parametrize_all_storages_and_caches
def test_index_meta(storage):
    index_meta = IndexMeta.create(TEST_META_KEY, storage)
    with pytest.raises(MetaInvalidKey):
        index_meta.custom_meta
    assert index_meta.entries == []
    index_meta.entries.append({"start_byte": 0})
    del index_meta

    index_meta = IndexMeta.load(TEST_META_KEY, storage)
    assert index_meta.entries == [{"start_byte": 0}]


@parametrize_all_storages_and_caches
@pytest.mark.xfail(raises=TensorMetaInvalidHtype, strict=True)
def test_invalid_htype(storage):
    TensorMeta.create(TEST_META_KEY, storage, htype="bad_htype")


@parametrize_all_storages_and_caches
@pytest.mark.xfail(raises=TensorMetaInvalidHtypeOverwriteKey, strict=True)
def test_invalid_htype_overwrite_key(storage):
    TensorMeta.create(TEST_META_KEY, storage, non_existent_htype_key="some_value")


@parametrize_all_storages_and_caches
@pytest.mark.xfail(raises=MetaInvalidKey, strict=True)
def test_read_invalid_meta_key(storage):
    meta = Meta(TEST_META_KEY, storage, required_meta={})
    meta.some_key


@parametrize_all_storages_and_caches
@pytest.mark.xfail(raises=MetaInvalidRequiredMetaKey, strict=True)
def test_invalid_required_meta(storage):
    # "version" should not be passed into `required_meta` (auto-populated)
    meta = Meta(TEST_META_KEY, storage, required_meta={"version": hub.__version__})
