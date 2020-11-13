from hub.compute.transform import Transform
from hub.compute.pathos import PathosTransform
from hub.compute.ray import RayTransform


def transform(schema, scheduler="single", processes=1):
    """
    Transform is a decorator of a function that outputs an dictionary for transforming a dataset
    
    Parameters
        ----------
        schema: Schema
            The output format of the transformed dataset
        scheduler: str
            "single" - for single threaded, "pathos" using multiprocessing, "ray" using ray scheduler, "dask" scheduler 
    """
    def wrapper(func):
        def inner(ds):
            if scheduler == "pathos":
                return PathosTransform(func, schema, ds)

            if scheduler == "ray":
                return RayTransform(func, schema, ds)
            
            if scheduler == "dask":
                raise NotImplementedError

            return Transform(func, schema, ds)
        return inner

    return wrapper
