import pytest
from hub.core.meta.encode.byte_positions import BytePositionsEncoder


def test_trivial():
    enc = BytePositionsEncoder()

    assert enc.num_samples == 0

    enc.add_byte_position(8, 100)
    enc.add_byte_position(8, 100)

    assert enc.num_samples == 200
    assert len(enc._encoded) == 1
    assert enc.num_bytes_encoded_under_row(-1) == 1600

    enc.add_byte_position(1, 1000)

    assert enc.num_samples == 1200
    assert len(enc._encoded) == 2
    assert enc.num_bytes_encoded_under_row(-1) == 2600

    assert enc.get_byte_position(0) == (0, 8)
    assert enc.get_byte_position(1) == (8, 16)
    assert enc.get_byte_position(199) == (1592, 1600)
    assert enc.get_byte_position(200) == (1600, 1601)
    assert enc.get_byte_position(201) == (1601, 1602)
    assert enc.get_byte_position(1199) == (2599, 2600)

    enc.add_byte_position(16, 32)

    assert enc.num_samples == 1232
    assert len(enc._encoded) == 3
    assert enc.num_bytes_encoded_under_row(-1) == 3112

    assert enc.get_byte_position(1200) == (2600, 2616)

    with pytest.raises(IndexError):
        enc.get_byte_position(1232)


def test_failures():
    enc = BytePositionsEncoder()

    with pytest.raises(Exception):
        # num_samples cannot be 0
        enc.add_byte_position(8, 0)
