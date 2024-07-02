import deeplake
from deeplake.core.vectorstore.vector_search import filter as filter_utils

import pytest


def test_attribute_based_filtering():
    ds = deeplake.empty("mem://deeplake_test")
    ds.create_tensor("metadata", htype="json")
    ds.create_tensor("metadata2", htype="json")
    ds.create_tensor("text", htype="text")
    ds.create_tensor("text2", htype="text")
    ds.metadata.extend([{"k": 1}, {"k": 2}, {"k": 3}, {"k": 4}])
    ds.metadata2.extend([{"kk": "a"}, {"kk": "b"}, {"kk": "c"}, {"kk": "d"}])
    ds.text.extend(["AA", "BB", "CC", "DD"])
    ds.text2.extend(["11", "22", "33", "DD"])

    # Test basic filter
    filter_dict = {"metadata": {"k": 1}, "metadata2": {"kk": "a"}, "text": "AA"}

    def filter_udf(x):
        metadata = x["metadata"].data()["value"]
        return metadata["k"] == 1

    view_dict = filter_utils.attribute_based_filtering_python(ds, filter=filter_dict)

    view_udf = filter_utils.attribute_based_filtering_python(ds, filter=filter_udf)

    view_tql, _ = filter_utils.attribute_based_filtering_tql(ds, filter=filter_dict)

    assert view_dict.metadata.data()["value"][0] == filter_dict["metadata"]
    assert view_dict.metadata2.data()["value"][0] == filter_dict["metadata2"]
    assert view_dict.text.data()["value"][0] == filter_dict["text"]

    assert view_udf.metadata.data()["value"][0] == filter_dict["metadata"]

    assert len(view_tql) == len(ds)

    # Test filter with list
    filter_dict_list = {"text2": ["11", "DD"]}

    view_dict_list = filter_utils.attribute_based_filtering_python(
        ds, filter=filter_dict_list
    )

    view_tql_list, _ = filter_utils.attribute_based_filtering_tql(
        ds, filter=filter_dict_list
    )

    assert view_dict_list.text2.data()["value"][0] in filter_dict_list["text2"]
    assert view_dict_list.text2.data()["value"][1] in filter_dict_list["text2"]

    assert len(view_tql_list) == len(ds)

    # Test bad tensor
    filter_dict_bad_tensor = {
        "metadata_bad": {"k": 1},
        "metadata2": {"kk": "a"},
        "text": "AA",
    }
    with pytest.raises(ValueError):
        filter_utils.attribute_based_filtering_python(ds, filter=filter_dict_bad_tensor)
    with pytest.raises(ValueError):
        filter_utils.attribute_based_filtering_tql(ds, filter=filter_dict_bad_tensor)


# No longer used in user-facing features but we still have it in the repo and will continue to test it
def test_exact_text_search():
    view = deeplake.empty("mem://deeplake_test")
    view.create_tensor("text", htype="text")
    view.text.extend(["abcd", "avc", "anv", "abc"])

    with pytest.warns(
        UserWarning,
        match="Exact text search wasn't able to find any files. Try other search options like embedding search.",
    ):
        filter_utils.exact_text_search(view=view, query="amk")

    (filtered_view, filtered_scores, filtered_index) = filter_utils.exact_text_search(
        view=view, query="abc"
    )
    assert filtered_scores == [1.0, 1.0]
    assert filtered_index == [0, 3]


def test_get_id_indices():
    view = deeplake.empty("mem://deeplake_test")
    view.create_tensor("ids")
    view.ids.extend(["ac", "bs", "cd", "fd"])

    ids = ["ac", "cd"]

    converted_ids = filter_utils.get_id_indices(view, ids)
    assert converted_ids == [0, 2]

    with pytest.raises(Exception):
        converted_ids = filter_utils.get_id_indices(view, ["ac", "cde"])


def test_get_ids_that_does_not_exist():
    ids = ["ac", "bs", "cd", "fd"]
    filtered_ids = ["ac", "bs"]

    targ_ids_that_doesnt_exist = "`cd`, `fd`"

    ids_that_doesnt_exist = filter_utils.get_ids_that_does_not_exist(ids, filtered_ids)
    assert targ_ids_that_doesnt_exist == ids_that_doesnt_exist


def test_get_filtered_ids():
    view = deeplake.empty("mem://deeplake_test")
    view.create_tensor("metadata", htype="json")
    view.metadata.extend([{"abc": 1}, {"cd": 2}, {"se": 3}])

    ids = filter_utils.get_filtered_ids(view, filter={"metadata": {"se": 3}})
    assert ids == [2]

    with pytest.raises(ValueError):
        ids = filter_utils.get_filtered_ids(view, filter={"metadata": {"se0": 3}})


def test_get_converted_ids():
    view = deeplake.empty("mem://deeplake_test")
    view.create_tensor("metadata", htype="json")
    view.metadata.extend([{"abc": 1}, {"cd": 2}, {"se": 3}])
    view.create_tensor("ids")
    view.ids.extend(["ac", "bs", "cd"])

    ids = ["cd"]
    filter = {"metadata": {"se": 3}}

    with pytest.raises(ValueError):
        ids = filter_utils.get_converted_ids(view, filter, ids)

    ids = filter_utils.get_converted_ids(view, filter=None, ids=ids)
    assert ids == [2]

    ids = filter_utils.get_converted_ids(view, filter=filter, ids=None)
    assert ids == [2]
