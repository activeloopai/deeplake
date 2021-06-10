from hub.util.keys import get_dataset_meta_key
import hub
from hub.core.meta.dataset_meta import DatasetMeta


def test_dataset_meta(memory_storage):
    dataset_meta = DatasetMeta(get_dataset_meta_key(), memory_storage)

    assert len(dataset_meta.tensors) == 0

    dataset_meta.tensors.append("tensor1")
    dataset_meta.tensors += ["tensor2"]
    dataset_meta.tensors.extend(["tensor3"])

    dataset_meta.custom_meta["something"] = ["brainzzz"]
    dataset_meta.custom_meta["something"].append("i am zombie")
    dataset_meta.custom_meta.update({"another_thing": 123})

    dataset_meta.custom_meta["nested_thing"] = {"a": 123, "b": {"b.0": [1,2,3], "b.1": {"b.1.something": [1]}}}

    del dataset_meta

    dataset_meta = DatasetMeta(get_dataset_meta_key(), memory_storage)
    assert dataset_meta.version == hub.__version__
    assert dataset_meta.tensors == ["tensor1", "tensor2", "tensor3"]

    assert dataset_meta.custom_meta["something"] == ["brainzzz", "i am zombie"]
    assert dataset_meta.custom_meta["another_thing"] == 123
    assert dataset_meta.custom_meta["nested_thing"] == {"a": 123, "b": {"b.0": [1,2,3], "b.1": {"b.1.something": [1]}}}

    dataset_meta.custom_meta["another_thing"] = "AAAAAAHHHHH"
    dataset_meta.custom_meta["nested_thing"]["b"]["b.1"]["b.1.something"].append(5)
    dataset_meta.custom_meta["nested_thing"]["b"]["b.1"]["WHY WOULD YOU DO THIS??"] = "test"

    del dataset_meta

    dataset_meta = DatasetMeta(get_dataset_meta_key(), memory_storage)
    assert dataset_meta.custom_meta["another_thing"] == "AAAAAAHHHHH"
    assert dataset_meta.custom_meta["nested_thing"] == {"a": 123, "b": {"b.0": [1,2,3], "b.1": {"b.1.something": [1, 5], "WHY WOULD YOU DO THIS??": "test"}}}