from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
from functools import partial
from tqdm import tqdm
from deeplake.core.index import Index
from deeplake.hooks import dataset_read

import numpy as np

import bisect


class MultiView:
    def __init__(self, items):
        self.items = items
        self.cumulative_sizes = self.cumsum(self.items)

    def cumsum(self):
        raise NotImplementedError

    def item_len(self, item):
        raise NotImplementedError

    def get_ij(self, idx):
        if idx < 0:
            idx = len(self) + idx
        i = bisect.bisect_right(self.cumulative_sizes, idx)
        if i == 0:
            j = idx
        else:
            j = idx - self.cumulative_sizes[i - 1]
        return i, j

    def __add__(self, item):
        return self.__class__(self.items + [item])

    def __iter__(self):
        for x in self.items:
            for y in x:
                yield y

    def __len__(self):
        return self.cumulative_sizes[-1]

    def __getitem__(
        self,
        item: Union[
            str, int, slice, List[int], Tuple[Union[int, slice, Tuple[int]]], Index
        ],
    ):
        if isinstance(item, str):
            raise NotImplementedError
        else:
            idx_obj = Index(item)
            sample_idx, rest = idx_obj.values[0], Index(idx_obj.values[1:])
            sample_idx.validate(len(self))
            if sample_idx.is_trivial():
                return self.__class__([item[rest] for item in self.items])
            elif sample_idx.subscriptable():
                items = []
                idx = sample_idx.value
                if isinstance(idx, tuple):
                    for x in idx:
                        i, j = self.get_ij(x)
                        items.append(self.items[i][j][rest])
                    return self.__class__(items)
                elif isinstance(idx, slice):
                    start, stop, step = idx.start, idx.stop, idx.step
                    if step is None:
                        step = 1
                    if step >= 0:
                        if start is None:
                            start = 0
                        if stop is None:
                            stop = -1
                        if start > stop:
                            return self.__class__([])
                    else:
                        if start is None:
                            start = -1
                        if stop is None:
                            stop = 0
                        if start < stop:
                            return self.__class__([])
                    start_i, start_j = self.get_ij(start)
                    stop_i, stop_j = self.get_ij(stop)
                    if start_i == stop_i:
                        return self.__class__([self.items[start_i][idx][rest]])
                    elif stop_i > start_i:
                        next_start = start_j
                        for i in range(start_i, stop_i):
                            items.append(self.items[i][next_start::step][rest])
                            next_start = step - (
                                (self.item_len(self.items[i]) - next_start) % step
                            )
                            if next_start == step:
                                next_start = 0
                        items.append(self.items[stop_i][next_start:stop_j:step][rest])
                    else:
                        next_start = start_j
                        for i in range(start_i, stop_i, -1):
                            items.append(self.items[i][next_start::step][rest])
                            next_start = step + (next_start % -step)
                            if next_start == step:
                                next_start = -1
                        items.append(self.items[stop_i][next_start:stop_j:step][rest])
                    return self.__class__(items)
            else:
                i, j = self.get_ij(sample_idx.value)
                return self.__class__([self.items[i][j][rest]])


class MultiDatasetView(MultiView):
    def __init__(self, datasets):
        super().__init__(datasets)

    def cumsum(self, sequence):
        r, s = [], 0
        for e in sequence:
            l = e.max_len
            r.append(l + s)
            s += l
        return r

    def item_len(self, item):
        return item.max_len

    @staticmethod
    def is_compatible(ds1, ds2):
        if set(ds1.tensors) == set(ds2.tensors):
            for key in ds1.tensors:
                config1 = ds1[key]._config
                config2 = ds2[key]._config
                if config1 != config2:
                    return False
            return True
        return False

    def __iter__(self):
        for d in self.items:
            for x in d.max_view:
                yield x

    def __str__(self):
        if not self.items:
            return "MultiDatasetView([])"
        res = "MultiDatasetView(["
        for dataset in self.items:
            path_str = f"'{dataset.path}', " if dataset.path else ""
            index_str = (
                f"index={dataset.index}, " if not dataset.index.is_trivial() else ""
            )
            group_str = (
                f"group_index='{dataset.group_index}'" if dataset.group_index else ""
            )
            if not group_str:
                index_str = index_str[:-2]  # remove comma
            dataset_str = f"Dataset({path_str}{index_str}{group_str})"
            res += f"\n\t{dataset_str}, "

        return f"{res}\n])"

    __repr__ = __str__

    def __getitem__(self, item):
        if isinstance(item, str):
            return MultiTensorView([dataset[item] for dataset in self.items])
        return super().__getitem__(item)

    def pytorch(
        self,
        transform: Optional[Callable] = None,
        tensors: Optional[Sequence[str]] = None,
        tobytes: Union[bool, Sequence[str]] = False,
        num_workers: int = 1,
        batch_size: int = 1,
        drop_last: bool = False,
        collate_fn: Optional[Callable] = None,
        pin_memory: bool = False,
        shuffle: bool = False,
        buffer_size: int = 2048,
        use_local_cache: bool = False,
        use_progress_bar: bool = False,
        return_index: bool = True,
        pad_tensors: bool = False,
        transform_kwargs: Optional[Dict[str, Any]] = None,
    ):
        from deeplake.integrations import datasets_to_pytorch

        if transform and transform_kwargs:
            transform = partial(transform, **transform_kwargs)

        dataloader = datasets_to_pytorch(
            self.items,
            transform=transform,
            tensors=tensors,
            tobytes=tobytes,
            num_workers=num_workers,
            batch_size=batch_size,
            drop_last=drop_last,
            collate_fn=collate_fn,
            pin_memory=pin_memory,
            shuffle=shuffle,
            buffer_size=buffer_size,
            use_local_cache=use_local_cache,
            return_index=return_index,
            pad_tensors=pad_tensors,
        )

        if use_progress_bar:
            dataloader = tqdm(dataloader, desc=self.path, total=len(self) // batch_size)
        for dataset in self.items:
            dataset_read(dataset)
        return dataloader

    def query(self, query_string: str):
        return MultiDatasetView([dataset.query(query_string) for dataset in self.items])


class MultiTensorView(MultiView):
    def __init__(self, tensors):
        super().__init__(tensors)

    def cumsum(self, sequence):
        r, s = [], 0
        for e in sequence:
            l = len(e)
            r.append(l + s)
            s += l
        return r

    def item_len(self, item):
        return len(item)

    def __getitem__(self, item):
        if isinstance(item, str):
            return MultiTensorView([tensor[item] for tensor in self.tensors])
        return super().__getitem__(item)

    def __str__(self):
        if not self.items:
            return "MultiTensorView([])"
        res = "MultiTensorView(["
        for tensor in self.items:
            ds_str = f"{tensor.dataset.path}, " if tensor.dataset.path else ""
            index_str = (
                f", index={tensor.index}" if not tensor.index.is_trivial() else ""
            )
            res += f"\n\tTensor({ds_str}key={repr(tensor.meta.name or tensor.key)}{index_str}),"
        return f"\n{res}])"

    __repr__ = __str__

    def numpy(self, aslist=False):
        if not aslist:
            return np.vstack([tensor.numpy() for tensor in self.items])
        return [tensor.numpy(aslist=True) for tensor in self.items]
