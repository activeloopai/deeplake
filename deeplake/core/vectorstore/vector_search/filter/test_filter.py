import deeplake
from deeplake.core.vectorstore.vector_search import filter as filter_utils

import pytest


def test_attribute_based_filtering():
    view = deeplake.empty("mem://deeplake_test")
    view.create_tensor("metadata", htype="json")
    view.metadata.extend([{"abcd": 1}, {"abcd123": 2}, {"abcd32": 3}, {"abcrd": 4}])
    exec_otion = "compute_engine"
    filter_dict = {"abcd": 1}

    with pytest.raises(NotImplementedError):
        view = filter_utils.attribute_based_filtering(
            view, filter=filter_dict, exec_option="compute_engine"
        )

    with pytest.raises(NotImplementedError):
        view = filter_utils.attribute_based_filtering(
            view, filter=filter_dict, exec_option="tensor_db"
        )

    view = filter_utils.attribute_based_filtering(
        view, filter=filter_dict, exec_option="python"
    )

    assert view.metadata.data()["value"][0] == filter_dict

    with pytest.raises(ValueError):
        view = filter_utils.attribute_based_filtering(
            view, filter={"aaaccc": 2}, exec_option="python"
        )


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

    ids = filter_utils.get_filtered_ids(view, filter={"se": 3})
    assert ids == [2]

    with pytest.raises(ValueError):
        ids = filter_utils.get_filtered_ids(view, filter={"se0": 3})


def test_get_converted_ids():
    view = deeplake.empty("mem://deeplake_test")
    view.create_tensor("metadata", htype="json")
    view.metadata.extend([{"abc": 1}, {"cd": 2}, {"se": 3}])
    view.create_tensor("ids")
    view.ids.extend(["ac", "bs", "cd"])

    ids = ["cd"]
    filter = {"se": 3}

    with pytest.raises(ValueError):
        ids = filter_utils.get_converted_ids(view, filter, ids)

    ids = filter_utils.get_converted_ids(view, filter=None, ids=ids)
    assert ids == [2]

    ids = filter_utils.get_converted_ids(view, filter=filter, ids=None)
    assert ids == [2]
