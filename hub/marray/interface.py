import warnings
from hub.utils.store_control import StoreControlClient
from hub.marray.array import HubArray
import numpy as np
from hub.exceptions import WrongTypeError
from hub.backend.storage import StorageFactory
from hub.marray.dataset import Dataset


def _get_path(name, public=False):
    if len(name.split('/')) == 1:
        name = '{}/{}'.format(name, name)
    user = name.split('/')[0]
    dataset = name.split('/')[1].split(':')[0]
    tag = name.split(':')
    if len(tag) == 1:
        tag.append('latest')
    tag = tag[1]
    path = user+'/'+dataset+'/'+tag
    return path


def array(shape=None, name=None, dtype='float', chunk_size=None, backend='s3', caching=False):

    if not name:
        raise Exception(
            'No name provided, please name your array - hub.array(..., name="username/dataset:version") '
        )
    path = _get_path(name)

    if not shape:
        return load(name)

    try:
        dtype = np.dtype(dtype).name
    except:
        raise WrongTypeError('Dtype {} is not supported '.format(dtype))

    # auto chunking
    if chunk_size is None:
        chunk_size = list(shape)
        chunk_size[0] = 1

    # Input checking
    assert len(chunk_size) == len(shape)
    assert np.array(shape).dtype in np.sctypes['int']
    assert np.array(chunk_size).dtype in np.sctypes['int']

    storage = StorageFactory(protocols=backend, caching=caching)

    return HubArray(
        shape=shape,
        dtype=dtype,
        chunk_shape=chunk_size,
        key=path,
        protocol=storage.protocol,
        storage=storage
    )


def dataset(arrays=None, name=None):
    # TODO check inputs validity
    name = _get_path(name)
    if arrays is None:
        return Dataset(key=name)
    return Dataset(arrays, name)


def load(name, backend='s3'):
    is_public = name in ['imagenet', 'cifar', 'coco', 'mnist']
    path = _get_path(name, is_public)

    storage = StorageFactory(protocols=backend)

    return HubArray(key=path, public=is_public, storage=storage)


# FIXME implement deletion of repositories
def delete(name):
    path = _get_path(name)
    bucket = StoreControlClient().get_config()['BUCKET']
    s3.Object(bucket, path.split(bucket)[-1]).delete()
