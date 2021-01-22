from shutil import Error


def redis_loaded():
    try:
        from redis import Redis

        r = Redis()
        r.set("foo", "bar")
        assert r.get("foo") == "bar"
    except:
        return False
    return True