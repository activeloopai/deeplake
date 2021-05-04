from hub.util.slice import merge_slices


RANGE = range(100)


def test_merge_slices_first_longer():
    first = slice(0, 4)
    second = slice(1, 3)
    assert RANGE[first][second] == RANGE[merge_slices(first, second)]


def test_merge_slices_second_longer():
    first = slice(0, 4)
    second = slice(1, 7)
    assert RANGE[first][second] == RANGE[merge_slices(first, second)]


def test_merge_slices_start_none():
    first = slice(None, 4)
    second = slice(1, 7)
    assert RANGE[first][second] == RANGE[merge_slices(first, second)]


def test_merge_slices_stop_both_none():
    first = slice(1, None)
    second = slice(2, None)
    assert RANGE[first][second] == RANGE[merge_slices(first, second)]


def test_merge_slices_stop_first_none():
    first = slice(1, None)
    second = slice(3, 7)
    assert RANGE[first][second] == RANGE[merge_slices(first, second)]


def test_merge_slices_stop_second_none():
    first = slice(2, 5)
    second = slice(1, None)
    assert RANGE[first][second] == RANGE[merge_slices(first, second)]


def test_merge_slices_irregular_steps():
    first = slice(4, 101, 2)
    second = slice(10, 20, 3)
    assert RANGE[first][second] == RANGE[merge_slices(first, second)]
