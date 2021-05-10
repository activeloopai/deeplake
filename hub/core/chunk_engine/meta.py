import os
import pickle

import hub


def default_meta():
    return {"hub_version": hub.__version__}


def meta_func(func):
    def wrapper(key, *args, **kwargs):
        if not key.endswith("/meta.json"):
            key = os.path.join(key, "meta.json")
        return func(key, *args, **kwargs)

    return wrapper


@meta_func
def has_meta(key, storage):
    return key in storage.mapper


@meta_func
def get_meta(key, storage):
    # TODO: don't use pickle
    return pickle.loads(storage[key])


@meta_func
def set_meta(key, storage, meta):
    # TODO: don't use pickle
    storage[key] = pickle.dumps(meta)


@meta_func
def validate_meta(key, storage, **kwargs):
    if has_meta(key, storage):
        meta = get_meta(key, storage)

        for arg_key, arg_v in kwargs.items():
            if meta[arg_key] != arg_v:
                # TODO: move into exceptions.py
                raise Exception("`%s` mismatch." % arg_key)


@meta_func
def update_meta(key, storage, length=0, **kwargs):
    if "chunk_size" in kwargs:
        if kwargs["chunk_size"] <= 0:
            # TODO: move into exceptions.py
            raise Exception("Chunk size too small")

    if has_meta(key, storage):
        meta = get_meta(key, storage)

        meta["length"] += length
    else:
        meta = default_meta()
        meta.update(
            {
                "length": length,
                **kwargs,
            }
        )

    return meta


@meta_func
def validate_and_update_meta(key, storage, **kwargs):
    validate_meta(key, storage, **kwargs)
    meta = update_meta(
        key,
        storage,
        **kwargs,
    )

    return meta
