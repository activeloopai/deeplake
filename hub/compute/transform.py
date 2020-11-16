from typing import Dict
import hub
from collections.abc import MutableMapping
from hub.features.features import Primitive
from hub.utils import batchify
from hub.exceptions import AdvancedSlicingNotSupported


class Transform:
    def __init__(self, func, schema, ds, **kwargs):
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
        **kwargs:
            additional arguments that will be passed to func as static argument for all samples
        """
        self._func = func
        self.schema = schema
        self._ds = ds
        self.kwargs = kwargs

    def _determine_shape(self, length: int):
        """
        Helper function to determine the shape of the newly created dataset

        Parameters
        ----------
        length: int
            in case shape is None, user can provide length
        """
        shape = self._ds.shape if hasattr(self._ds, "shape") else None
        if shape is None:
            if length is not None:
                shape = (length,)
            else:
                try:
                    shape = (len(self._ds),)
                except Exception as e:
                    raise e
        return shape

    def _flatten_dict(self, d: Dict, parent_key=''):
        """
        Helper function to flatten dictionary of a recursive tensor

        Parameters
        ----------
        d: dict
        """
        items = []
        for k, v in d.items():
            new_key = parent_key + '/' + k if parent_key else k
            if isinstance(v, MutableMapping) and not isinstance(self.dtype_from_path(new_key), Primitive):
                items.extend(self._flatten_dict(v, new_key).items())
            else:
                items.append((new_key, v))
        return dict(items)

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
            for key, value in x.items():
                if key in xs_new:
                    xs_new[key].append(value)
                else: 
                    xs_new[key] = [value]
        return xs_new

    def dtype_from_path(self, path):
        """
        Helper function to get the dtype from the path
        """
        path = path.split('/')
        cur_type = self.schema
        for subpath in path[:-1]:
            cur_type = cur_type[subpath]
            cur_type = cur_type.dict_
        return cur_type[path[-1]]

    def upload(self, ds, results):
        """ Batchified upload of results
        For each tensor batchify based on its chunk and upload
        If tensor is dynamic then still upload element by element

        Parameters
        ----------
        dataset: hub.Dataset
            Dataset object that should be written to
        results:
            Output of transform function

        Returns
        ----------
        ds: hub.Dataset
            Uploaded dataset
        """
        for key, value in results.items():
            length = ds[key].chunksize[0]
            batched_values = batchify(value, length)

            for i, batch in enumerate(batched_values):
                # FIXME replace below 8 lines with ds[key, i * length : (i + 1) * length] = batchs
                if not ds[key].is_dynamic:
                    if len(batch) != 1:
                        ds[key, i * length : (i + 1) * length] = batch
                    else:
                        ds[key, i * length] = batch[0]
                else:
                    for k, el in enumerate(batch):
                        ds[key, i * length + k] = el
        return ds

    def store(self, url, token=None, length=None):
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
        Returns
        ----------
        ds: hub.Dataset
            uploaded dataset
        """
        shape = self._determine_shape(length)
        ds = hub.Dataset(
            url, mode="w", shape=shape, schema=self.schema, token=token, cache=False,
        )

        results = [self._func(item, **self.kwargs) for item in self._ds]
        results = [self._flatten_dict(r) for r in results]
        results = self._split_list_to_dicts(results)
        ds = self.upload(ds, results)

        return ds

    def __getitem__(self, slice_):
        """
        Get an item to be compute without iterating on the whole dataset

        slice_: int
            currently stands for index, but need to add advanced slicing as if it is a dataset
        """
        # TODO add advanced slicing as if the transform of the dataset has access to any element
        if not isinstance(slice_, int):
            raise AdvancedSlicingNotSupported

        return self._func(self._ds[slice_], **self.kwargs)

    def __len__(self):
        return self.shape[0]

    def __iter__(self):
        for item in self._ds:
            yield self._func(item, **self.kwargs)

    @property
    def shape(self):
        return self._ds.shape

