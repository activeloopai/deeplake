import argparse
import hub
import zarr
import time
import numpy as np
import os

def get_args():
    parser = argparse.ArgumentParser(description='Hub vs Zar')
    parser.add_argument('--type', default='hub', metavar='T')
    parser.add_argument('--path', default='/drive/hub_vs_zarr/', metavar='P')
    return parser.parse_args()


def main():
    # another_test()
    args = get_args()
    arr = zarr_array(args)
    print(arr.chunks)
    run_test(args, arr, 'Zarr')
    p = os.path.join(args.path, '_zarr')
    zarr.save(p, arr)
    arr = zarr.open(p, 'r')
    print(arr.chunks)
    arr = hub_array(args)
    run_test(args, arr, 'Hub')


# def another_test():
#     arr1 = zarr.ones(shape=(10, 10))
#     arr2 = arr1 + arr1
#     print(arr2)

def run_test(args, arr, name: str):
    t = time.time()
    rand_arr = np.random.randint(0, high=50000, size=(1000, 1000))
    for i in range(0, 50):
        for j in range(0, 50):
            arr[1000 * i: 1000 * i + 1000, 1000 * j : 1000 * j + 1000] = rand_arr
    print(f'{name}, Time: {time.time() - t}')
    
def zarr_array(args):
    storage = zarr.storage.NestedDirectoryStore(os.path.join(args.path, 'zarr'))
    arr = zarr.create(shape=(50000, 50000), chunks=(1000, 1000), dtype='int32', store=storage, overwrite=True)
    return arr

def hub_array(args):
    bucket = hub.fs(os.path.join(args.path, 'hub')).connect()
    arr = bucket.array('my_array', shape=(50000, 50000), chunk=(1000, 1000), dtype='int32')
    return arr

if __name__ == '__main__':
    main()