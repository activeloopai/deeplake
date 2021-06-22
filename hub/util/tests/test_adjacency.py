import numpy as np
from hub.util.adjacency import calculate_adjacent_runs


def test_1():
    L = [0]
    counts, elems = calculate_adjacent_runs(L)
    np.testing.assert_array_equal(counts, [1])
    np.testing.assert_array_equal(elems, [0])


def test_2():
    L = [1, 1, 1, 1, 1, 1, 1]
    counts, elems = calculate_adjacent_runs(L)
    np.testing.assert_array_equal(counts, [7])
    np.testing.assert_array_equal(elems, [1])


def test_3():
    L = [1, 5, 5, 5, 5, 5]
    counts, elems = calculate_adjacent_runs(L)
    np.testing.assert_array_equal(counts, [1, 5])
    np.testing.assert_array_equal(elems, [1, 5])


def test_4():
    L = [1, 2, 3, 4, 5]
    counts, elems = calculate_adjacent_runs(L)
    np.testing.assert_array_equal(counts, [1, 1, 1, 1, 1])
    np.testing.assert_array_equal(elems, [1, 2, 3, 4, 5])


def test_5():
    L = [1, 1, 1, 2, 1, 2, 2, 2, 2, 2, 2, 5, 5]
    counts, elems = calculate_adjacent_runs(L)
    np.testing.assert_array_equal(counts, [3, 1, 1, 6, 2])
    np.testing.assert_array_equal(elems, [1, 2, 1, 2, 5])


def test_6():
    L = [1, 1, 1, 2, 1, 2, 2, 2, 2, 2, 2, 5]
    counts, elems = calculate_adjacent_runs(L)
    np.testing.assert_array_equal(counts, [3, 1, 1, 6, 1])
    np.testing.assert_array_equal(elems, [1, 2, 1, 2, 5])


def test_tuples():
    L = [(10, 10), (10, 10), (10, 11)]
    counts, elems = calculate_adjacent_runs(L)
    np.testing.assert_array_equal(counts, [2, 1])
    np.testing.assert_array_equal(elems, [(10, 10), (10, 11)])


def test_single_tuple():
    L = [(99, 99)]
    counts, elems = calculate_adjacent_runs(L)
    np.testing.assert_array_equal(counts, [1])
    np.testing.assert_array_equal(elems, [(99, 99)])


def test_empty():
    L = []
    counts, elems = calculate_adjacent_runs(L)
    np.testing.assert_array_equal(counts, [])
    np.testing.assert_array_equal(elems, [])
