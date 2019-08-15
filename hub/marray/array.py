import warnings
from hub.utils.store_control import StoreControlClient
from hub.marray.interface import HubArray
import numpy as np
from hub.exceptions import WrongTypeError

def _get_path(name, public=False):
    if len(name.split('/')) == 1:
        name = '{}/{}'.format(name,name)
    user = name.split('/')[0]
    dataset = name.split('/')[1].split(':')[0]
    tag = name.split(':')
    if len(tag) == 1:
        tag.append('latest')
    tag = tag[1]
    bucket = StoreControlClient().get_config(public)['BUCKET']
    if bucket=='':
        exit()
    path = 's3://'+bucket+'/'+user+'/'+dataset+'/'+tag
    #print(path)
    return path

def load(name):
    is_public = name in ['imagenet', 'cifar', 'coco', 'mnist']
    path = _get_path(name, is_public)
    return HubArray(key=path, public=is_public)

def array(shape=None, name=None, dtype='float', chunk_size=None):

    if not name:
        raise Exception('No name provided, please name your array - hub.array(..., name="username/dataset:version") ')

    path = _get_path(name)

    if not shape:
        return load(name)

    return create(path, shape, dtype, chunk_size)
    
def create(path, dim=[50000, 28, 28], dtype='float', chunk_size=None):
    # auto chunking
    if chunk_size is None:
        chunk_size = list(dim)
        chunk_size[0] = 1

    # Input checking
    assert len(chunk_size) == len(dim)
    assert np.array(dim).dtype in np.sctypes['int']
    assert np.array(chunk_size).dtype in np.sctypes['int']
    
    try:
        np.zeros((1), dtype=dtype)
    except:
        raise WrongTypeError('Dtype {} is not supported '.format(dtype))

    return HubArray(shape=dim, dtype=dtype, chunk_shape=chunk_size, key=path, protocol=None)
    
def delete(name):
    path = _get_path(name)
    bucket = StoreControlClient().get_config()['BUCKET']
    s3.Object(bucket, path.split(bucket)[-1]).delete()
