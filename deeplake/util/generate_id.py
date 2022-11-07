from uuid import uuid4
import numpy as np
import os


def generate_id(dtype: type):
    shift = 128 - (8 * dtype(1).itemsize)
    return dtype(uuid4().int >> shift)
