from deeplake.core.index import merge_slices, slice_at_int
from pytest_cases import parametrize_with_cases  # type: ignore
import pytest


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

    def case_negative_step(self):
        return slice(None), slice(None, None, -3)


@parametrize_with_cases("first,second", cases=MergeSlicesCases)
def test_merge_slices(first: slice, second: slice):
    r = range(100)
    assert r[first][second] == r[merge_slices(first, second)]


def test_slice_at_int():
    assert slice_at_int(slice(2, 10, 2), 3) == 8
    assert slice_at_int(slice(9, 2, -2), 0) == 9
    assert slice_at_int(slice(9, 2, -2), 2) == 5
    assert slice_at_int(slice(None, None, -3), 0) == -1
    assert slice_at_int(slice(None, None, -3), 2) == -7
    assert slice_at_int(slice(None, 9, -3), 2) == -7
    assert slice_at_int(slice(-1, -8, -2), 3) == -7
    assert slice_at_int(slice(-1, None, -2), 3) == -7
    assert slice_at_int(slice(None, -10, -3), 2) == -7
    assert slice_at_int(slice(2, 5, None), 2) == 4

    with pytest.raises(IndexError):
        slice_at_int(slice(2, 9, -1), 0)

    with pytest.raises(NotImplementedError):
        slice_at_int(slice(2, 6, 2), -3)
