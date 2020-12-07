from hub.compute.transform import Transform
from hub.compute.ray import RayTransform, RayGeneratorTransform
from collections.abc import Iterable
from hub.exceptions import NotIterable


def transform(schema, scheduler="single", workers=1):
    """| Transform is a decorator of a function. The function should output a dictionary per sample.

    Parameters:
    ----------
    schema: Schema
        The output format of the transformed dataset
    scheduler: str
        "single" - for single threaded, "threaded" using multiple threads, "processed", "ray" scheduler, "dask" scheduler
    workers: int
        how many workers will be started for the process
    """

    def wrapper(func):
        def inner(ds, **kwargs):
            if not isinstance(ds, Iterable) and not isinstance(ds, str):
                raise NotIterable

            if scheduler == "ray":
                return RayTransform(
                    func, schema, ds, scheduler=scheduler, workers=workers, **kwargs
                )

            if scheduler == "ray_generator":
                return RayGeneratorTransform(
                    func, schema, ds, scheduler=scheduler, workers=workers, **kwargs
                )

            return Transform(
                func, schema, ds, scheduler=scheduler, workers=workers, **kwargs
            )

        return inner

    return wrapper
