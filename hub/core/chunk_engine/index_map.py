import os
import pickle


def default_index_map():
    return []


def index_map_func(func):
    def wrapper(key, *args, **kwargs):
        if not key.endswith("/index_map.json"):
            key = os.path.join(key, "index_map.json")
        return func(key, *args, **kwargs)

    return wrapper


@index_map_func
def has_index_map(key, storage):
    return key in storage.mapper


@index_map_func
def get_index_map(key, storage):
    if has_index_map(key, storage):
        # TODO: don't use pickle
        return pickle.loads(storage[key])
    return default_index_map()


# TODO: not set function, should be append (so we're not rewriting the index map every time)
@index_map_func
def set_index_map(key, storage, index_map):
    # TODO: don't use pickle for index_map/meta
    storage[key] = pickle.dumps(index_map)
