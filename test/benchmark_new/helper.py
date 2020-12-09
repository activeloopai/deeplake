import numpy as np
import hub
from hub.schema import Tensor
from hub.log import logger


def generate_dataset(shape=(10,), size=(1024, 1024), chunksize=None):
    """
    Generates a datasets with random tensors
    """
    my_schema = {"img": Tensor(shape=shape, chunks=chunksize)}
    ds = hub.Dataset("kristina/benchmarking", shape=(10,), schema=my_schema)
    for i in range(shape):
        ds[i] = np.random.rand(size)
    return ds


def report(logs):
    """
    Print logs generated from benchmarks
    """
    print(" ")
    for log in logs:
        logger.info(f"~~~~ {log['name']} ~~~~")
        del log["name"]
        for k, v in log.items():
            logger.info(f"{k}: {v}")
