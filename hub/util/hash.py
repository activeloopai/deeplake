import hashlib


def hash_inputs(*args) -> str:
    sha = hashlib.sha3_256()
    for item in args:
        sha.update(str(item).encode("utf-8"))
    return sha.hexdigest()


def hash_str_to_int32(string: str):
    hash_ = int(hashlib.sha256(string.encode("utf-8")).hexdigest(), 16) >> 224
    return hash_
