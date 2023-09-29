import math
from typing import List, Sequence, Union, Any
import warnings
import numpy as np
from collections import defaultdict
from deeplake.core.meta.encode.chunk_id import ChunkIdEncoder
from deeplake.core.seed import DeeplakeRandom
import deeplake.core.dataset


def find_primary_tensor(dataset):
    current_max_size = 0
    primary_tensor_name = None
    for tensor_key, tensor in dataset.tensors.items():
        max_shape = tensor.meta.max_shape
        max_size = np.prod(max_shape)
        if max_size > current_max_size:
            current_max_size = max_size
            primary_tensor_name = tensor_key

    return primary_tensor_name


def create_fetching_schedule(
    dataset: "deeplake.core.dataset.Dataset",
    primary_tensor_name: str,
    shuffle_within_chunks: bool = False,
):
    slice_ = dataset.index.values[0].value
    index_struct: Union[set, dict, None] = None

    if isinstance(slice_, int):
        return None
    elif isinstance(slice_, slice):
        start = slice_.start if slice_.start is not None else 0
        stop = slice_.stop if slice_.stop is not None else dataset.min_len
        step = slice_.step if slice_.step is not None else 1
        index_struct = set(range(start, stop, step))
    elif isinstance(slice_, (list, tuple)):
        index_struct = defaultdict(lambda: 0)
        for item in slice_:
            index_struct[item] += 1
    primary_tensor = dataset[primary_tensor_name]
    try:
        chunk_id_encoder: ChunkIdEncoder = primary_tensor.chunk_engine.chunk_id_encoder
    except NotImplementedError:
        return None
    enc_array = chunk_id_encoder.array
    num_chunks = chunk_id_encoder.num_chunks
    # pick chunks randomly, one by one
    prev_state = np.random.get_state()
    np.random.seed(DeeplakeRandom().get_seed())
    chunk_order = np.random.choice(num_chunks, num_chunks, replace=False)
    schedule: List[Any] = []
    for chunk_idx in chunk_order:
        start_index = int(enc_array[chunk_idx - 1][1]) + 1 if chunk_idx > 0 else 0
        last_index = int(enc_array[chunk_idx][1]) + 1
        indexes = np.arange(start_index, last_index)
        if shuffle_within_chunks:
            np.random.shuffle(indexes)
        schedule.extend(indexes)

    if isinstance(index_struct, set):
        schedule = [int(idx) for idx in schedule if idx in index_struct]
    elif isinstance(index_struct, dict):
        idxs = filter(lambda idx: idx in index_struct, schedule)  # type: ignore
        schedule = [int(idx) for idx in idxs for _ in range(index_struct[idx])]
    np.random.set_state(prev_state)
    return schedule


def calculate_absolute_lengths(
    percent_lengths: Sequence[Union[int, float]], absolute_length: int
):
    subset_lengths: List[int] = []
    for i, frac in enumerate(percent_lengths):
        if frac < 0 or frac > 1:
            raise ValueError(f"Fraction at index {i} is not between 0 and 1")
        n_items_in_split = int(
            math.floor(absolute_length * frac)  # type: ignore[arg-type]
        )
        subset_lengths.append(n_items_in_split)
    remainder = absolute_length - sum(subset_lengths)  # type: ignore[arg-type]
    # add 1 to all the percent_lengths in round-robin fashion until the remainder is 0
    for i in range(remainder):
        idx_to_add_at = i % len(subset_lengths)
        subset_lengths[idx_to_add_at] += 1
    percent_lengths = subset_lengths
    for i, length in enumerate(percent_lengths):
        if length == 0:
            warnings.warn(
                f"Length of split at index {i} is 0. "
                f"This might result in an empty dataset."
            )
    return percent_lengths


def create_random_split_views(dataset, lengths):
    from deeplake.enterprise.convert_to_libdeeplake import import_indra_api

    import_indra_api()
    if math.isclose(sum(lengths), 1) and sum(lengths) <= 1:
        lengths = calculate_absolute_lengths(lengths, len(dataset))
    if sum(lengths) != len(dataset):  # type: ignore[arg-type]
        raise ValueError(
            "Sum of input lengths does not equal the length of the input dataset!"
        )

    primary_tensor = find_primary_tensor(dataset)
    schedule = create_fetching_schedule(
        dataset, primary_tensor, shuffle_within_chunks=True
    )
    ds = dataset.no_view_dataset
    sliced_ds = ds[schedule]
    views = []
    start = 0
    for length in lengths:
        end = start + length
        view = sliced_ds[start:end]
        views.append(view)
        start = end
    return views
