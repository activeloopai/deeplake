import math
from typing import Optional


def get_indexes(
    dataset,
    rank: Optional[int] = None,
    num_replicas: Optional[int] = None,
    drop_last: Optional[bool] = None,
):
    """
    Generates a slice for a given rank in a distributed setting, dividing
    the dataset evenly across multiple replicas.

    Parameters:
        dataset (Dataset): The dataset to split across distributed replicas.
        rank (Optional[int]): The rank of the current process. If not specified,
                              the function will use the distributed package to get the current rank.
        num_replicas (Optional[int]): Total number of replicas (i.e., processes) involved in distributed training.
                                      If not specified, the function will determine the number based on the world size.
        drop_last (Optional[bool]): If True, drop the extra data not evenly divisible among replicas.
                                    This is useful for maintaining equal batch sizes across replicas.

    Returns:
        slice: A slice object representing the start and end indices for the current rank's portion of the dataset.

    Raises:
        RuntimeError: If the distributed package is not available when `rank` or `num_replicas` are not specified.
        ValueError: If the specified `rank` is out of range based on the number of replicas.

    Notes:
        This function requires the `torch.distributed` package to determine the number of replicas and
        rank when they are not provided. It is useful in distributed data loading to ensure each process
        gets a specific subset of the data.
    """
    import torch.distributed as dist

    if num_replicas is None:
        if not dist.is_available():
            raise RuntimeError("Requires distributed package to be available")
        num_replicas = dist.get_world_size()
    if rank is None:
        if not dist.is_available():
            raise RuntimeError("Requires distributed package to be available")
        rank = dist.get_rank()
    if rank >= num_replicas or rank < 0:
        raise ValueError(
            "Invalid rank {}, rank should be in the interval"
            " [0, {}]".format(rank, num_replicas - 1)
        )

    dataset_length = len(dataset)

    if drop_last:
        total_size = (dataset_length // num_replicas) * num_replicas
        per_process = total_size // num_replicas
    else:
        per_process = math.ceil(dataset_length / num_replicas)
        total_size = per_process * num_replicas

    start_index = rank * per_process
    end_index = min(start_index + per_process, total_size)

    end_index = min(end_index, dataset_length)

    return slice(start_index, end_index)
