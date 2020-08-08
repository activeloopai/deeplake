import numpy as np
from hub import tensor, dataset as ds
from hub.log import logger

def generate_dataset(shapes=[(10, 1024, 1024)], chunksize=None):
    """
    Generates a datasets with random tensors
    """
    ds_dict = {}
    for shape in shapes:
        data = np.random.rand(*shape)
        ds_dict[f"ds{str(shape)}"] = tensor.from_array(data, chunksize=chunksize)
    _ds = ds.from_tensors(ds_dict)
    return _ds

def report(logs):
    """
    Print logs generated from benchmarks
    """
    print(" ")
    for log in logs:
        logger.info(f"~~~~ {log['name']} ~~~~")
        del log["name"]
        for k,v in log.items(): 
            logger.info(f"{k}: {v}")