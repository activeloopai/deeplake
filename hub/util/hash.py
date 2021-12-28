import hashlib


def hash_inputs(*args) -> str:
    sha = hashlib.sha3_256()
    for item in args:
        sha.update(str(item).encode("utf-8"))
    return sha.hexdigest()
