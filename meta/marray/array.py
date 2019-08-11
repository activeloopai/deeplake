from cloudvolume import CloudVolume

import warnings
from meta.utils.store_control import StoreControlClient

def _get_path(name):
    if len(name.split('/')) == 1:
        raise Exception('array name should be specified the following format (username/dataset:version)')
        return 
    user = name.split('/')[0]
    dataset = name.split('/')[1].split(':')[0]
    tag = name.split(':')
    if len(tag) == 1:
        tag.append('latest')
    tag = tag[1]
    bucket = StoreControlClient().get_config()['bucket']
    path = 's3://'+bucket+'/'+user+'/'+dataset+'/'+tag
    return path

def array(dim=None, name='username/test:latest', dtype='uint8'):
    path = _get_path(name)
    if not dim:
        return CloudVolume(path, parallel=True, progress=False, fill_missing=True, non_aligned_writes=True)

    return create(path, dim, dtype)
    
def create(path, dim=[50000, 28, 28], dtype='uint8', num_channels = 1):
    if len(dim)==1:
        dim = (dim[0],1,1,1)
        warnings.warn('Meta arrays support only 3D/4D arrays. Expanding shape to {}'.format(str(dim)))

    if len(dim)==2:
        dim = (dim[0], dim[1], 1, 1)
        warnings.warn('Meta arrays support only 3D/4D arrays. Expanding shape to {}'.format(str(dim)))

    if len(dim)==4:
        num_channels = dim[3]
        dim = dim[:3]
    
    if len(dim)>4:
        raise Exception('Meta arrays support up to 4D')

    chunk_size = list(dim)
    chunk_size[0] = 1
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

def load(name):
    path = _get_path(name)
    return CloudVolume(path, parallel=True, fill_missing=True, progress=False,  non_aligned_writes=True, autocrop=True, green_threads=False)
    
# TODO implement a cloudvolume simple wrapper that translates errors into Meta errors
def delete(name):
    path = _get_path(name)
    bucket = StoreControlClient().get_config()['bucket']
    s3.Object(bucket, path.split(bucket)[-1]).delete()
