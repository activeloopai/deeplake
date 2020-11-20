import os
import zarr
from typing import Dict, Iterable
from hub.api.dataset import Dataset
from tqdm import tqdm
from collections.abc import MutableMapping
from hub.utils import batchify
from hub.api.dataset_utils import slice_extract_info, slice_split
import collections.abc as abc
from hub.api.datasetview import DatasetView
from pathos.pools import ProcessPool, ThreadPool
from hub.features import Primitive


try:
    from ray.util.multiprocessing import Pool as RayPool
except Exception:
    pass


class Transform:
    def __init__(
        self, func, schema, ds, scheduler: str = "single", nodes: int = 1, **kwargs
    ):
        """
        Transform applies a user defined function to each sample in single threaded manner

        Parameters
        ----------
        func: function
            user defined function func(x, **kwargs)
        schema: dict of dtypes
            the structure of the final dataset that will be created
        ds: Iterative
            input dataset or a list that can be iterated
        scheduler: str
            choice between "single", "threaded", "processed"
        nodes: int
            how many threads or processes to use
        **kwargs:
            additional arguments that will be passed to func as static argument for all samples
        """
        self._func = func
        self.schema = schema
        self._ds = ds
        self.kwargs = kwargs

        if scheduler == "threaded" or (scheduler == "single" and nodes > 1):
            self.map = ThreadPool(nodes=nodes).map
        elif scheduler == "processed":
            self.map = ProcessPool(nodes=nodes).map
        elif scheduler == "single":
            self.map = map
        elif scheduler == "ray":
            self.map = RayPool().map
        else:
            raise Exception(
                f"Scheduler {scheduler} not understood, please use 'single', 'threaded', 'processed'"
            )

    @classmethod
    def _flatten_dict(self, d: Dict, parent_key="", schema=None):
        """
        Helper function to flatten dictionary of a recursive tensor

        Parameters
        ----------
        d: dict
        """
        items = []
        for k, v in d.items():
            new_key = parent_key + "/" + k if parent_key else k
            if isinstance(v, MutableMapping) and not isinstance(
                self.dtype_from_path(new_key, schema), Primitive
            ):
                items.extend(
                    self._flatten_dict(v, parent_key=new_key, schema=schema).items()
                )
            else:
                items.append((new_key, v))
        return dict(items)

    @classmethod
    def dtype_from_path(cls, path, schema):
        """
        Helper function to get the dtype from the path
        """
        path = path.split("/")
        cur_type = schema
        for subpath in path[:-1]:
            cur_type = cur_type[subpath]
            cur_type = cur_type.dict_
        return cur_type[path[-1]]

    def _split_list_to_dicts(self, xs):
        """
        Helper function that transform list of dicts into dicts of lists

        Parameters
        ----------
        xs: list of dicts

        Returns
        ----------
        xs_new: dicts of lists
        """
        xs_new = {}
        for x in xs:
            if isinstance(x, list):
                x = dict(
                    zip(self._flatten_dict(self.schema, schema=self.schema).keys(), x)
                )

            for key, value in x.items():
                if key in xs_new:
                    xs_new[key].append(value)
                else:
                    xs_new[key] = [value]
        return xs_new

    def upload(self, results, url: str, token: dict, progressbar: bool = True):
        """Batchified upload of results
        For each tensor batchify based on its chunk and upload
        If tensor is dynamic then still upload element by element

        Parameters
        ----------
        dataset: hub.Dataset
            Dataset object that should be written to
        results:
            Output of transform function
        progressbar: bool
        Returns
        ----------
        ds: hub.Dataset
            Uploaded dataset
        """

        shape = (len(list(results.values())[0]),)
        ds = Dataset(
            url,
            mode="w",
            shape=shape,
            schema=self.schema,
            token=token,
            fs=zarr.storage.MemoryStore() if "tmp" in url else None,
            cache=False,
        )

        for key, value in results.items():

            length = ds[key].chunksize[0]
            if length == 0:
                length = 1

            batched_values = batchify(value, length)

            def upload_chunk(i_batch):
                i, batch = i_batch
                if not ds[key].is_dynamic:
                    if len(batch) != 1:
                        ds[key, i * length : (i + 1) * length] = batch
                    else:
                        ds[key, i * length] = batch[0]
                else:
                    for k, el in enumerate(batch):
                        ds[key, i * length + k] = el

            index_batched_values = list(
                zip(list(range(len(batched_values))), batched_values)
            )
            index_batched_values = self._pbar(progressbar)(
                index_batched_values,
                desc=f"Storing {key} tensor",
                total=len(value) // length,
            )
            list(self.map(upload_chunk, index_batched_values))

        ds.commit()
        return ds

    def _pbar(self, show: bool = True):
        """
        Returns a progress bar, if empty then it function does nothing
        """

        def _empty_pbar(xs, **kwargs):
            return xs

        single_threaded = self.map == map
        return tqdm if show and single_threaded else _empty_pbar

    def _unwrap(self, results):
        """
        If there is any list then unwrap it into its elements
        """
        items = []
        for r in results:
            if isinstance(r, dict):
                items.append(r)
            else:
                items.extend(r)
        return items

    def store(
        self,
        url: str,
        token: dict = None,
        length: int = None,
        ds: Iterable = None,
        progressbar: bool = True,
    ):
        """
        The function to apply the transformation for each element in batchified manner

        Parameters
        ----------
        url: str
            path where the data is going to be stored
        token: str or dict, optional
            If url is refering to a place where authorization is required,
            token is the parameter to pass the credentials, it can be filepath or dict
        length: int
            in case shape is None, user can provide length
        ds: Iterable
        progressbar: bool
            Show progress bar
        Returns
        ----------
        ds: hub.Dataset
            uploaded dataset
        """

        _ds = ds or self._ds

        def _func_argd(item):
            return self._func(item, **self.kwargs)

        results = self.map(
            _func_argd, self._pbar(progressbar)(_ds, desc="Computing the transormation")
        )
        results = self._unwrap(results)
        results = self.map(lambda x: self._flatten_dict(x, schema=self.schema), results)
        results = self._split_list_to_dicts(results)

        ds = self.upload(results, url=url, token=token, progressbar=progressbar)
        return ds

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, slice_):
        """
        Get an item to be computed without iterating on the whole dataset
        Creates a dataset view, then a temporary dataset to apply the transform

        slice_: slice
            Gets a slice or slices from dataset
        """
        if not isinstance(slice_, abc.Iterable) or isinstance(slice_, str):
            slice_ = [slice_]

        slice_ = list(slice_)
        subpath, slice_list = slice_split(slice_)

        if len(slice_list) == 0:
            slice_list = [slice(None, None, None)]
        
        num, ofs = slice_extract_info(slice_list[0], self.shape[0])
        
        ds_view = DatasetView(
            dataset=self._ds,
            num_samples=num,
            offset=ofs,
            squeeze_dim=isinstance(slice_list[0], int),
        )

        path = os.path.expanduser("~/.activeloop/tmparray")
        new_ds = self.store(path, length=num, ds=ds_view, progressbar=False)

        index = 1 if len(slice_) > 1 else 0
        slice_[index] = (
            slice(None, None, None) if not isinstance(slice_list[0], int) else 0
        )  # Get all shape dimension since we already sliced
        return new_ds[slice_]

    def __iter__(self):
        for index in range(len(self)):
            yield self[index]

    @property
    def shape(self):
        return self._ds.shape
