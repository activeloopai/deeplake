from hub.utils import EmptyLock


class BasicSynchronizer(object):
    def __init__(self) -> None:
        super().__init__()
        self.iterator = {}

    def __getitem__(self, item: str):
        """Get empty lock
        ----------
        item: int
            get the lock
        """
        return EmptyLock()

    def append(self, key: str = "default", number: int = 0):
        """Appends the counter with the number and returns final value
        ----------
        number: int
            append the number
        """
        if key not in self.iterator:
            self.iterator[key] = 0
        self.iterator[key] += number
        return self.iterator[key]

    def get(self, key: str = "default"):
        """Gets the index"""
        return self.iterator[key] if key in self.iterator else 0

    def reset(self, key: str = "default", default: int = 0):
        """Resets counter"""
        self.iterator[key] = default
        return self.iterator[key]