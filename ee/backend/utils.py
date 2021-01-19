def redis_loaded():
    try:
        from redis import Redis

        r = Redis()
        r.set("foo", "bar")
    except ImportError:
        return False
    return True