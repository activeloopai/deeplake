import os
import json

from redis import StrictRedis
import redis_lock
import logging
import zarr

logger = logging.getLogger("redis_lock")
logger.setLevel(level=logging.WARNING)


class RedisSynchronizer(object):
    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        debug: bool = False,
        password: str = None,
        using_ray: bool = True,
    ):
        """Provides synchronization using redis locks
        Parameters
        ----------
        host: string
            url of the redis
        port: int
            port redis is running
        db: int
            id of the database
        debug: bool
            if debug logs should be printed
        """

        self.host = host
        self.port = port
        self.db = db
        self.conn = None
        self.password = None
        self.using_ray = using_ray

    def _get_connection(self):
        self.host = (
            os.environ["RAY_HEAD_IP"]
            if "RAY_HEAD_IP" in os.environ and self.using_ray
            else self.host
        )

        if self.password is None and self.using_ray:
            self.password = "5241590000000000"

        return StrictRedis(
            host=self.host, port=self.port, db=self.db, password=self.password
        )

    def __getitem__(self, item: str):
        conn = self._get_connection()
        lock = redis_lock.Lock(conn, item, strict=True)
        return lock

    def append(self, key: str = "default", number: int = 0):
        """Appends the counter with the number and returns final value
        ----------
        number: int
            append the number
        """
        conn = self._get_connection()
        return conn.incrby(key, amount=number)

    def get(self, key: str = "default"):
        """Gets the index"""
        conn = self._get_connection()
        value = conn.get(key)
        return int(value) if value is not None else 0

    def set(self, key: str = "default", number: int = 0):
        """Sets the index"""
        conn = self._get_connection()
        return conn.set(key, number)

    def reset(self, key: str = "default", default: int = 0):
        """Resets counter"""
        conn = self._get_connection()
        return conn.set(key, default)


class ProcessSynchronizer(zarr.ProcessSynchronizer):
    def __init__(self, path):
        self._filepath = path + "_datafile"
        super().__init__(path)

    def _read(self, key) -> int:
        if not os.path.exists(self._filepath):
            return None
        with open(self._filepath, "r") as f:
            data = json.loads(f.read())
            return data[key]

    def _write(self, key, number: int):
        with open(self._filepath, "a+") as f:
            f.seek(0)
            bytes_ = f.read()
            data = bytes_ and json.loads(bytes_) or dict()
            f.seek(0)
            data[key] = number
            bytes_ = json.dumps(data)
            f.truncate()
            f.write(bytes_)

    def append(self, key: str = "default", number: int = 0):
        with self[key]:
            ans = number + self._read(key)
            self._write(key, ans)
            return ans

    def get(self, key: str = "default") -> int:
        with self[key]:
            return self._read(key)

    def set(self, key: str = "default", number: int = 0):
        with self[key]:
            self._write(key, number)
            return number

    reset = set