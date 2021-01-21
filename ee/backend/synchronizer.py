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

        # if not debug:

    def _get_conn(self):
        redis_url = (
            os.environ["RAY_HEAD_IP"] if "RAY_HEAD_IP" in os.environ else self.host
        )

        if self.password is None:
            self.password = "5241590000000000"

        return StrictRedis(
            host=redis_url, port=self.port, db=self.db, password=self.password
        )

    def __getitem__(self, item: str):
        conn = self._get_conn()

    def __getitem__(self, item: str):
        conn = self._get_connection()
        lock = redis_lock.Lock(conn, item, strict=True)
        return lock

    def read_position(self) -> int:
        conn = self._get_conn()

        return int(conn.get("$pos") or 0)

    def write_position(self, pos: int) -> None:
        conn = self._get_conn()
        conn["$pos"] = pos
