import warnings
from hub.utils.store_control import StoreControlClient
from hub.marray.interface import HubArray
import numpy as np
from hub.exceptions import WrongTypeError

try:
    from cloudvolume import CloudVolume
    cloudvolume_exist = False
except ImportError as e:
    cloudvolume_exist = False

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
    return path

def array(dim=None, name='username/test:latest', dtype='uint8', chunk_size=None):
    path = _get_path(name)
    if not dim and cloudvolume_exist:
        return CloudVolume(path, parallel=True, progress=False, fill_missing=True, non_aligned_writes=True)

    return create(path, dim, dtype, chunk_size, cloudvolume_exist)
    
def create(path, dim=[50000, 28, 28], dtype='uint8', chunk_size=None, cloudvolume=False):
    # TODO to remvoe
    if cloudvolume:
        if len(dim)==1:
            dim = (dim[0],1,1,1)
            warnings.warn('hub arrays support only 3D/4D arrays. Expanding shape to {}'.format(str(dim)))

        if len(dim)==2:
            dim = (dim[0], dim[1], 1, 1)
            warnings.warn('hub arrays support only 3D/4D arrays. Expanding shape to {}'.format(str(dim)))
 
        if len(dim)==4:
            num_channels = dim[3]
            dim = dim[:3]
        
        if len(dim)>4:
            raise Exception('hub arrays support up to 4D')

    # auto chunking
    if not chunk_size:
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

    if cloudvolume_exist:
        info = CloudVolume.create_new_info(
            num_channels    = num_channels,
            layer_type      = 'image',
            data_type       = dtype, # Channel images might be 'uint8'
            encoding        = 'raw', # raw, jpeg, compressed_segmentation, fpzip, kempressed
            resolution      = [1, 1, 1], # Voxel scaling, units are in nanometers
            voxel_offset    = [0, 0, 0], # x,y,z offset in voxels from the origin

            # Pick a convenient size for your underlying chunk representation
            # Powers of two are recommended, doesn't need to cover image exactly
            chunk_size      = chunk_size, # units are voxels
            volume_size     = dim, # e.g. a cubic millimeter dataset
        )

        vol = CloudVolume(path, info=info, parallel=True, fill_missing=True, progress=False,  non_aligned_writes=True, autocrop=True, green_threads=False)
        vol.commit_info()
        return vol
    else:
        return HubArray(shape=dim, dtype=dtype, chunk_shape=chunk_size, key=path, protocol=None)

def load(name):
    is_public = name in ['imagenet', 'cifar', 'coco', 'mnist']
    path = _get_path(name, is_public)
    if cloudvolume_exist:
        return CloudVolume(path, parallel=True, fill_missing=True, progress=False,  non_aligned_writes=True, autocrop=True, green_threads=False)
    else:
        return HubArray(key=path, public=is_public)
    
def delete(name):
    path = _get_path(name)
    bucket = StoreControlClient().get_config()['BUCKET']
    s3.Object(bucket, path.split(bucket)[-1]).delete()
