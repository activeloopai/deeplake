import hub

class HubBucket():
    def __init__(self, storage, backend):
        self._storage = storage
        self._backend = backend

    def arraynew(self, shape=None, name=None, dtype='float', chunk=None, compress='zlib', compresslevel=6):
        return hub.array(shape, name, dtype, chunk, backend=self._backend, storage=self._storage, compression=compress, compression_level=compresslevel)

    def arrayload(self, name):
        return hub.load(name, backend=self._backend, storage=self._storage)

    def arraydelete(self, name):
        raise NotImplementedError()


