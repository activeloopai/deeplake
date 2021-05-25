from pytest_cases import parametrize_with_cases  # type: ignore

from hub.util.slice import merge_slices


class MergeSlicesCases:
    def case_first_longer(self):
        return slice(0, 4), slice(1, 3)

    def case_second_longer(self):
        return slice(0, 4), slice(1, 7)

    def case_start_none(self):
        return slice(None, 4), slice(1, 7)

    def case_stop_first_none(self):
        return slice(1, None), slice(3, 7)

    def case_stop_second_none(self):
        return slice(2, 5), slice(1, None)

    def case_stop_both_none(self):
        return slice(1, None), slice(2, None)

    def case_irregular_steps(self):
        return slice(4, 101, 2), slice(10, 20, 3)


@parametrize_with_cases("first,second", cases=MergeSlicesCases)
def test_merge_slices(first: slice, second: slice):
    r = range(100)
    assert r[first][second] == r[merge_slices(first, second)]
