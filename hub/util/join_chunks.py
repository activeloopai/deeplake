from typing import List


def join_chunks(chunks: List[bytes], start_byte: int, end_byte: int) -> bytes:
    if len(chunks) == 1:
        return memoryview(chunks[0])[start_byte:end_byte]
    b = bytearray()
    for i, chunk in enumerate(chunks):
        actual_start_byte, actual_end_byte = 0, len(chunk)
        if i <= 0:
            actual_start_byte = start_byte
        if i >= len(chunks) - 1:
            actual_end_byte = end_byte
        b += chunk[actual_start_byte:actual_end_byte]
    return b
