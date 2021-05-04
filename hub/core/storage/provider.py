from collections.abc import MutableMapping


class Provider(MutableMapping):
    def __init__(self):
        raise NotImplementedError

    def get_credentials(self):
        raise NotImplementedError

    def __getitem__(self, path):
        raise NotImplementedError

    def __setitem__(self, path):
        raise NotImplementedError

    def __delitem__(self, path):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError
