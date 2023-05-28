import pathlib
from typing import Dict, Optional, Union

from deeplake.client.log import logger
from deeplake.util.path import is_hub_cloud_path


def parse_runtime_parameters(
    path: Union[str, pathlib.Path], runtime: Optional[Dict] = None, verbose: bool = True
):
    """Parse runtime parameters from a dictionary.
    Will become more helpful as clutter in the paramter increases

    Args:
        runtime (Optional[Dict]): A dictionary containing runtime parameters.

    Returns:
        A dictionary containing parsed runtime parameters.
    """
    if isinstance(path, pathlib.Path):
        path = str(path)

    if runtime is None:
        runtime = {}

    db_engine = runtime.get(
        "db_engine", False
    )  # DB engine is kept for backward compatibility, can be removed if not used=
    tensor_db = runtime.get("tensor_db", False) or db_engine

    if tensor_db and not is_hub_cloud_path(path):
        raise ValueError(
            f"Path {path} is not a valid hub cloud path."
            "{'tensor_db': True} can only be used with hub cloud paths."
        )

    if verbose:
        invalid_keys = set(runtime.keys()) - {"db_engine", "tensor_db"}
        if len(invalid_keys):
            logger.warning(
                f"Invalid runtime parameters: {invalid_keys}. They will be ignored."
            )

    return {"tensor_db": tensor_db}
