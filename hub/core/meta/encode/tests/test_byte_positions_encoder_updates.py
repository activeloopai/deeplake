import numpy as np
from hub.core.meta.encode.byte_positions import BytePositionsEncoder
from .common import assert_encoded



def test_update_no_change():
    enc = BytePositionsEncoder(np.array([[8, 0, 10], [4, 88, 15]]))

    enc[0] = 8
    assert_encoded(enc, [[8, 0, 10], [4, 88, 15]])

    enc[1] = 8
    assert_encoded(enc, [[8, 0, 10], [4, 88, 15]])

    enc[10] = 8
    assert_encoded(enc, [[8, 0, 10], [4, 88, 15]])

    enc[11] = 4
    assert_encoded(enc, [[8, 0, 10], [4, 88, 15]])

    enc[12] = 4
    assert_encoded(enc, [[8, 0, 10], [4, 88, 15]])

    enc[15] = 4
    assert_encoded(enc, [[8, 0, 10], [4, 88, 15]])

    assert enc.num_samples == 16


# def test_update_squeeze_trivial():
#     enc = BytePositionsEncoder(np.array([[28, 0, 2], [100, 100, 3], [28, 0, 5]]))
# 
#     enc[3] = (28, 0)
#     assert_encoded(enc, [[28, 0, 5]])
# 
#     assert enc.num_samples == 6
# 
# 
# def test_update_squeeze_complex():
#     enc = BytePositionsEncoder(
#         np.array([[10, 10, 1], [28, 0, 2], [100, 100, 3], [28, 0, 5], [10, 10, 7]])
#     )
# 
#     enc[3] = (28, 0)
#     assert_encoded(enc, [[10, 10, 1], [28, 0, 5], [10, 10, 7]])
# 
#     assert enc.num_samples == 8
# 
# 
# def test_update_move_up():
#     enc = BytePositionsEncoder(np.array([[101, 100, 0], [100, 101, 5]]))
# 
#     enc[1] = (101, 100)
#     assert_encoded(enc, [[101, 100, 1], [100, 101, 5]])
# 
#     enc[2] = (101, 100)
#     assert_encoded(enc, [[101, 100, 2], [100, 101, 5]])
# 
#     assert enc.num_samples == 6
# 
# 
# def test_update_move_down():
#     enc = BytePositionsEncoder(np.array([[101, 100, 5], [100, 101, 10]]))
# 
#     enc[5] = (100, 101)
#     assert_encoded(enc, [[101, 100, 4], [100, 101, 10]])
# 
#     enc[4] = (100, 101)
#     assert_encoded(enc, [[101, 100, 3], [100, 101, 10]])
# 
#     enc[3] = (100, 101)
#     assert_encoded(enc, [[101, 100, 2], [100, 101, 10]])
# 
#     assert enc.num_samples == 11
# 
# 
# def test_update_replace():
#     enc = BytePositionsEncoder(np.array([[100, 100, 0]]))
#     enc[0] = (100, 101)
#     assert enc.num_samples == 1
# 
# 
# def test_update_split_up():
#     enc = BytePositionsEncoder(np.array([[100, 101, 5]]))
# 
#     enc[0] = (101, 100)
#     assert_encoded(enc, [[101, 100, 0], [100, 101, 5]])
# 
# 
# def test_update_split_down():
#     enc = BytePositionsEncoder(np.array([[100, 101, 5]]))
# 
#     enc[5] = (101, 100)
#     assert_encoded(enc, [[100, 101, 4], [101, 100, 5]])
# 
# 
# def test_update_split_middle():
#     enc = BytePositionsEncoder(np.array([[28, 0, 5]]))
# 
#     enc[3] = (100, 100)
#     assert_encoded(enc, [[28, 0, 2], [100, 100, 3], [28, 0, 5]])
# 