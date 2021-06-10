from hub.api import dataset
from hub.constants import DEFAULT_CHUNK_SIZE, DEFAULT_DTYPE, DEFAULT_HTYPE
from hub.tests.common import TENSOR_KEY
from hub.core.meta.tensor_meta import create_tensor_meta, load_tensor_meta
from hub.util.keys import get_dataset_meta_key, get_tensor_meta_key
import hub
from hub.core.meta.dataset_meta import create_dataset_meta, load_dataset_meta


def test_dataset_meta_updates(local_storage):
    # TODO: load data and create data as methods
    # dataset_meta = DatasetMeta(key=get_dataset_meta_key(), storage=local_storage)
    dataset_meta = create_dataset_meta(get_dataset_meta_key(), local_storage)

    assert dataset_meta.tensors == []
    assert dataset_meta.custom_meta == {}
    assert dataset_meta.version == hub.__version__

    dataset_meta.custom_meta["string"] = "test"

    dataset_meta.tensors.append("tensor1")
    dataset_meta.tensors += ["tensor2"]
    dataset_meta.tensors.extend(["tensor3"])

    dataset_meta.custom_meta["something"] = ["brainzzz"]
    dataset_meta.custom_meta["something"].append("i am zombie")
    dataset_meta.custom_meta.update({"another_thing": 123})

    dataset_meta.custom_meta["nested_thing"] = {"a": 123, "b": {"b.0": [1,2,3], "b.1": {"b.1.something": [1]}}}

    del dataset_meta

    # dataset_meta = DatasetMeta(key=get_dataset_meta_key(), storage=local_storage)
    dataset_meta = load_dataset_meta(get_dataset_meta_key(), local_storage)
    assert dataset_meta.tensors == ["tensor1", "tensor2", "tensor3"]

    assert dataset_meta.custom_meta["string"] == "test"
    assert dataset_meta.custom_meta["something"] == ["brainzzz", "i am zombie"]
    assert dataset_meta.custom_meta["another_thing"] == 123
    assert dataset_meta.custom_meta["nested_thing"] == {"a": 123, "b": {"b.0": [1,2,3], "b.1": {"b.1.something": [1]}}}

    dataset_meta.custom_meta["another_thing"] = "AAAAAAHHHHH"
    dataset_meta.custom_meta["nested_thing"]["b"]["b.1"]["b.1.something"].append(5)
    dataset_meta.custom_meta["nested_thing"]["b"]["b.1"]["WHY WOULD YOU DO THIS??"] = "test"
    dataset_meta.custom_meta["string"] += "test"

    del dataset_meta

    # dataset_meta = DatasetMeta(key=get_dataset_meta_key(), storage=local_storage)
    dataset_meta = load_dataset_meta(get_dataset_meta_key(), local_storage)
    assert dataset_meta.custom_meta["another_thing"] == "AAAAAAHHHHH"
    assert dataset_meta.custom_meta["nested_thing"] == {"a": 123, "b": {"b.0": [1,2,3], "b.1": {"b.1.something": [1, 5], "WHY WOULD YOU DO THIS??": "test"}}}
    assert dataset_meta.custom_meta["string"] == "testtest"


def test_tensor_meta_updates(local_storage):
    # TODO: generalize these tests

    # tensor_meta = (get_tensor_meta_key(local_storage.root), local_storage)
    tensor_meta = create_tensor_meta(get_tensor_meta_key(TENSOR_KEY), local_storage)

    assert tensor_meta.dtype == DEFAULT_DTYPE
    assert tensor_meta.htype == DEFAULT_HTYPE
    assert tensor_meta.chunk_size == DEFAULT_CHUNK_SIZE
    assert tensor_meta.length == 0
    assert tensor_meta.version == hub.__version__
    assert tensor_meta.custom_meta == {}

    tensor_meta.custom_meta["nested_thing"] = {"test": [1,2,3]}
    tensor_meta.length = tensor_meta.length + 1

    del tensor_meta

    tensor_meta = load_tensor_meta(get_tensor_meta_key(TENSOR_KEY), local_storage)

    assert tensor_meta.custom_meta["nested_thing"] == {"test": [1,2,3]}

    tensor_meta.custom_meta["nested_thing"]["test"].append(999)
    assert tensor_meta.length == 1

    tensor_meta.length += 1

    del tensor_meta

    tensor_meta = load_tensor_meta(get_tensor_meta_key(TENSOR_KEY), local_storage)

    assert tensor_meta.custom_meta["nested_thing"] == {"test": [1,2,3, 999]}
    assert tensor_meta.length == 2