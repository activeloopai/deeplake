# TODO: merge cache_chain -> storage_chain. we can merge them to reduce params
# TODO: create class to handle this merge, also the class should be moved from `chunk_engine/` to `storage/`


def write_bytes_with_caching(key, b, cache_chain, storage):
    if len(cache_chain) <= 0:
        # TODO: move into exceptions.py
        raise Exception("At least one cache inside of `cache_chain` is required.")

    # prioritize cache storage over main storage.
    cache_success = write_to_cache(key, b, cache_chain)

    if not cache_success:
        flush_cache(cache_chain, storage)
        cache_success = write_to_cache(key, b, cache_chain)

        if not cache_success:
            # TODO move into exceptions.py
            raise Exception("Caching failed even after flushing.")


def write_to_cache(key, b, cache_chain):
    # TODO: cross-cache storage (maybe the data doesn't fit in 1 cache, should we do so partially?)
    for cache in cache_chain:
        if cache.has_space(len(b)):
            cache[key] = b
            return True

    return False


def write_to_storage(key, b, storage):
    storage[key] = b


def flush_cache(cache_chain, storage):
    # TODO: send all cached data -> storage & clear the caches.

    for cache in cache_chain:
        keys = []
        for key, chunk in cache:
            storage[key] = chunk
            keys.append(key)

        for key in keys:
            del cache[key]

        # TODO: test flushing to make sure cache.used_space will return 0
