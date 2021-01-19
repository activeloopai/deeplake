from redis import StrictRedis
import redis_lock
import logging


class RedisSynchronizer(object):
    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        debug: bool = False,
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

        if not debug:
            logger = logging.getLogger("redis_lock")
            logger.setLevel(level=logging.WARNING)

    def __getitem__(self, item: str):
        conn = StrictRedis(host=self.host, port=self.port, db=self.db)
        lock = redis_lock.Lock(conn, item, strict=True)
        return lock