import pytest
import hub
import os

def test_load():
    cid = 'ipfs://QmTPWJ3JWHqCHKHADV3BLmc6baiL7GyBSDg7m3Ga5dtWwP'
    gw = 'https://ipfs.infura.io:5001/api/v0'
    ds = hub.load(cid, creds={"gw": gw}, read_only=True)
    assert ds.tensors['data'].key == 'data'

def test_copy():
    ds = hub.dataset('./test/')
    p = os.path.abspath("./test/")
    ds_copy = hub.copy('./test', 'ipfs://', dest_creds={"fpath": p})
    assert ds.path == 'ipfs://'
    return
