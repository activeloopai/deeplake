from hub.core.storage.mapped_provider import MappedProvider
import os


class LocalProvider(MappedProvider):
    """Provider class for using the local filesystem."""

    def __init__(self, root: str):
        """Initializes the LocalProvider.

        Example:
            local_provider = LocalProvider("/home/ubuntu/Documents/")

        Args:
            root (str): The root of the provider. All read/write request keys will be appended to root."
        """
        if os.path.isfile(root):
            raise Exception  # TODO better exception
        self.root = root

    def __getitem__(self, path: str):
        full_path = self._check_is_file(path)
        try:
            file = open(full_path, "rb")
            return file.read()
        except Exception:
            raise KeyError

    def __setitem__(self, path: str, value: bytes):
        full_path = self._check_is_file(path)
        file = open(full_path, "wb")
        file.write(value)

    def __delitem__(self, path: str):
        full_path = self._check_is_file(path)
        try:
            os.remove(full_path)
        except Exception:
            raise KeyError

    def __iter__(self):
        yield from self._list_keys()

    def __len__(self):
        return len(self._list_keys)

    def _list_keys(self):
        ls = []
        for r, d, f in os.walk(self.root):
            for file in f:
                ls.append(os.path.relpath(os.path.join(r, file), self.root))
        return ls

    def _check_is_file(self, path: str):
        full_path = os.path.join(self.root, path)
        if os.path.isdir(full_path):
            raise Exception  # TODO better exception
        return full_path
