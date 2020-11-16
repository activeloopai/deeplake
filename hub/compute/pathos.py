from hub.compute.transform import Transform

try:
    from pathos.pools import ProcessPool, ThreadPool
except Exception:
    pass


class PathosTransform(Transform):
    def __init__(self, func, schema, ds):
        Transform.__init__(self, func, schema, ds)
        Pool = ThreadPool or ProcessPool
        self.map = Pool(nodes=2).map