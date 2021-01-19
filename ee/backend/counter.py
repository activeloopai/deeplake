from redis import StrictRedis


class Counter:
    def __init__(self, key, host="localhost", port=6379, db=0, debug=False):
        """Implements redis counter
        ----------
        host: string
            url of the redis
        port: int
            port redis is running
        db: int
            db id
        debug: bool
            if debug logs should be printed
        """
        self.host = host
        self.port = port
        self.db = db
        self.debug = debug
        self.conn = StrictRedis(host=self.host, port=self.port, db=self.db)
        self.key = key

    def append(self, number: int):
        """Appends the counter with the number and returns final value
        ----------
        number: int
            append the number
        """
        return self.conn.incrby(self.key, amount=number)

    def get(self):
        """Gets the index"""
        value = self.conn.get(self.key)
        return int(value) if value is not None else 0

    def reset(self):
        """Resets counter"""
        return self.conn.set(self.key, 0)
