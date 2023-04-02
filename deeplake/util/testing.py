import numpy as np


def assert_array_equal(x, y, *args, **kwargs):
    if isinstance(x, (list, tuple)) or isinstance(y, (list, tuple)):
        assert len(x) == len(y)
        for xi, yi in zip(x, y):
            return np.testing.assert_array_equal(xi, yi, *args, **kwargs)
    else:
        return np.testing.assert_array_equal(x, y, *args, **kwargs)


def compare_version_info(info1, info2):
    for commit_id in info1["commits"]:
        commit_info_1 = info1["commits"][commit_id]
        commit_info_2 = info2["commits"][commit_id]
        for key in commit_info_1:
            if key == "children":
                if set(commit_info_1[key]) != set(commit_info_2[key]):
                    print(commit_info_1[key], commit_info_2[key])
                assert set(commit_info_1[key]) == set(commit_info_2[key])
            else:
                assert commit_info_1[key] == commit_info_2[key]

    assert info1["branches"] == info2["branches"]
