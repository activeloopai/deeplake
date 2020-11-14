import hub

try:
    import ray
except:
    pass

from collections.abc import MutableMapping
from hub.features.features import Primitive
from tqdm import tqdm
from hub.utils import batchify


class Transform:
    def __init__(self, func, schema, ds):
        self._func = func
        self._schema = schema
        self._ds = ds

    def __iter__(self):
        for item in self._ds:
            yield self._func(item)

    def store(self, url, token=None, length=None):
        shape = self._ds.shape if hasattr(self._ds, "shape") else None
        if shape is None:
            if length is not None:
                shape = (length,)
            else:
                try:
                    shape = (len(self._ds),)
                except Exception as e:
                    raise e

        ds = hub.Dataset(
            url, mode="w", shape=shape, schema=self._schema, token=token, cache=False, 
        )

        # apply transformation and some rewrapping
        results = [self._func(item) for item in self._ds]
        results = [self.flatten_dict(r) for r in results]
        results = self.split_list_to_dicts(results)
 
        # batchified upload
        for key, value in results.items():
            length = ds[key].chunksize[0]
            batched_values = batchify(value, length)
            for i, batch in enumerate(batched_values):
                ds[key, i * length:(i + 1) * length] = batch
        return ds

    def flatten_dict(self, d, parent_key=''):
        items = []
        for k, v in d.items():
            new_key = parent_key + '/' + k if parent_key else k
            if isinstance(v, MutableMapping) and not isinstance(self.dtype_from_path(new_key), Primitive):
                items.extend(self.flatten_dict(v, new_key).items())
            else:
                items.append((new_key, v))
        return dict(items)

        
    def split_list_to_dicts(self, xs):
        """
        Transform list of dicts into dicts of lists
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
        path = path.split('/')
        cur_type = self._schema
        for subpath in path[:-1]:
            cur_type = cur_type[subpath]
            cur_type = cur_type.dict_
        return cur_type[path[-1]]

    def _transfer_batch(self, ds, i, results):
        for j, result in enumerate(results[0]):
            for key in result:
                ds[key, i * ds.chunksize + j] = result[key]

    def _transfer(self, from_, to):
        assert isinstance(from_, dict)
        for key, value in from_.items():
            to[key] = from_[key]

    def __getitem__(self, slice_):
        raise NotImplementedError()

    def __len__(self):
        return self.shape[0]

    @property
    def shape(self):
        return self._ds.shape

    @property
    def schema(self):
        return self._schema