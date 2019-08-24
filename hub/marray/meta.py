import json
from hub.exceptions import NotFound
from hub.utils import StoreControlClient
from hub.backend.storage import Storage, S3, FS

class MetaObject(object):
    def __init__(self, key=None, storage=None, create=None):
        super().__init__()
        self._info = {
            'key': key
        }
        self.storage = storage if storage is not None else \
            S3(StoreControlClient.get_config()['BUCKET'])
        self.dclass = 'meta'
        self.create = create

    @property
    def info(self):
        return self._info

    @info.setter
    def info(self, value):
        self._info = value

    @property
    def key(self):
        return self.info['key']

    def not_found(self):
        name, dataset, version = self.key.split('/')[-3:]
        raise NotFound(
            'Could not identify {} with name {}/{}:{}. Please make sure the array name is correct.'.\
                    format(self.dclass, name, dataset, version))


    def initialize(self, path):
        cloudpath = "{}/info.txt".format(path)
        info = self.storage.get(cloudpath)
        
        if info is None and self.create is None:
            self.not_found()

        if info:
            info = json.loads(info.decode('utf-8'))
            if 'dclass' in info and info['dclass'] != self.dclass:
                self.not_found()
            self.info = info   
        else:
            info = json.dumps(self.info).encode('utf-8')
            self.storage.put(cloudpath, info)

    def __setitem__(self, slices):
        raise NotImplementedError

    def __getitem__(self, slices):
        raise NotImplementedError