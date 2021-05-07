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
    storage[key] = pickle.dumps(meta)
