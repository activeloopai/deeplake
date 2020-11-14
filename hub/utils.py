from math import gcd
import time


def _flatten(list_):
    """
    Helper function to flatten the list
    """
    return [item for sublist in list_ for item in sublist]


def gcp_creds_exist():
    """Checks if credentials exists"""

    try:
        import os

        env = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
        if env is not None:
            return True
        from google.cloud import storage

        storage.Client()
    except Exception:
        return False
    return True


def s3_creds_exist():
    import boto3

    sts = boto3.client("sts")
    try:
        sts.get_caller_identity()
    except Exception:
        return False
    return True


def azure_creds_exist():
    """Checks if credentials exists"""

    import os
    env = os.getenv("ACCOUNT_KEY")
    if env is not None:
        return True
    return False


def pytorch_loaded():
    try:
        import torch

        torch.__version__
    except ImportError:
        return False
    return True


def tensorflow_loaded():
    try:
        import tensorflow

        tensorflow.__version__
    except ImportError:
        return False
    return True


def tfds_loaded():
    try:
        import tensorflow_datasets

        tensorflow_datasets.__version__
    except ImportError:
        return False
    return True


def compute_lcm(a):
    """
    Lowest Common Multiple of a list a
    """
    lcm = a[0]
    for i in a[1:]:
        lcm = lcm * i / gcd(lcm, i)
    return int(lcm)


def batch(iterable, n=1):
    """
    Batchify an iteratable
    """
    ls = len(iterable)
    for ndx in range(0, ls, n):
        yield iterable[ndx : min(ndx + n, ls)]


class Timer:
    def __init__(self, text):
        self._text = text

    def __enter__(self):
        self._start = time.time()

    def __exit__(self, *args):
        print(f"{self._text}: {time.time() - self._start}s")
