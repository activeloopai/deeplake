def ensure_concurrent_mode(f):
    def g(self, *args, **kwargs):
        if not self._concurrent_mode:
            raise ValueError(f"ds.{f.__name__}() is available only in concurrent write mode.")
        return f(self, *args, **kwargs)
    return g

def ensure_not_concurrent_mode(f):
    def g(self, *args, **kwargs):
        ignore = kwargs.pop("_ignore_concurrent_mode_check", None)
        if not ignore and self._concurrent_mode:
            raise ValueError(f"ds.{f.__name__}() is not available in concurrent write mode.")
        return f(self, *args, **kwargs)
    return g
