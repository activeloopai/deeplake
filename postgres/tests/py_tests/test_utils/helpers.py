"""
Helper functions for test data generation.
"""
import random
from typing import List


def generate_random_float_array(length: int) -> List[float]:
    """
    Generate a random float array.

    Args:
        length: Length of the array

    Returns:
        List of random floats between 0 and 1
    """
    return [random.random() for _ in range(length)]


def generate_random_float_2d_array(rows: int, cols: int) -> List[List[float]]:
    """
    Generate a random 2D float array.

    Args:
        rows: Number of rows
        cols: Number of columns

    Returns:
        2D list of random floats between 0 and 1
    """
    return [[random.random() for _ in range(cols)] for _ in range(rows)]


def float_arrays_equal(arr1: List[float], arr2: List[float], tolerance: float = 1e-6) -> bool:
    """
    Compare two float arrays with tolerance.

    Args:
        arr1: First array
        arr2: Second array
        tolerance: Maximum allowed difference

    Returns:
        True if arrays are equal within tolerance, False otherwise
    """
    if len(arr1) != len(arr2):
        return False

    return all(abs(a - b) < tolerance for a, b in zip(arr1, arr2))
