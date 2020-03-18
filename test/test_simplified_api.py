import pytest
import hub
import numpy as np

arrname = 'test/array'

def main():
    test_array_create()
    test_array_open()
    test_array_delete()
    delete_item(arrname)

def _connect(bucket: str, creds: str):
    return hub.s3(bucket, aws_creds_filepath=creds).connect()

def connect():
    return _connect('snark-intelinair-export', '.creds/aws.json')

def delete_item(path: str, throw: bool = False):
    try:
        bucket = connect()
        bucket.delete(path)
    except Exception as ex:
        if throw:
            raise ex

def array_create(path: str):
    delete_item(path)
    bucket = connect()
    arr = bucket.array(path, shape=(100, 100, 2), chunk=(20, 20, 1), dtype=np.int32)
    return arr

def array_open(path: str):
    delete_item(path)
    array_create(path)
    bucket = connect()
    arr = bucket.open(path)
    arr[3,50] = np.array([1, 2])
    # print(arr[3, 50])
    return arr

def array_delete(path: str):
    delete_item(path)
    array_create(path)
    bucket = connect()
    bucket.delete(path)

def test_array_create():
    array_create(arrname)

def test_array_open():
    array_open(arrname)

def test_array_delete():
    array_delete(arrname)


if __name__ == "__main__":
    main()