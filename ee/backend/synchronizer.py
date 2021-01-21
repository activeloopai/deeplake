import os
from redis import StrictRedis
import redis_lock
import logging
from hub.utils import EmptyLock

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

    def _get_connection(self):
        self.host = (
            os.environ["RAY_HEAD_IP"] if "RAY_HEAD_IP" in os.environ else self.host
        )

        if self.password is None:
            self.password = "5241590000000000"

        conn = StrictRedis(
            host=self.host, port=self.port, db=self.db, password=self.password
        )
        return conn

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