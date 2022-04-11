import os
import pathlib
import pickle
import posixpath
import shutil
from typing import Optional, Set
from redis import StrictRedis
from config import redis_url
import redis
from common.redis_client import get_redis_client
from hub.core.storage.provider import StorageProvider
from hub.util.exceptions import DirectoryAtPathException, FileAtPathException


class LocalProvider(StorageProvider):
    """Provider class for using the local filesystem."""

    def __init__(self, redis_url):
        """Initializes the RedisProvider.

        Args:
            root (str): The root of the provider. All read/write request keys will be appended to root."

        Raises:
            FileAtPathException: If the root is a file instead of a directory.
        """
        
        self.r = redis.StrictRedis(url = redis_url, charset = "utf-8", decode_responses = True)

    def __getitem__(self, path: str):
        """Gets the object present at the path within the given byte range.

        Args:
            path (str): The path relative to the root of the provider.

        Returns:
            bytes: The bytes of the object present at the path.

        Raises:
            KeyError: If an object is not found at the path.
            DirectoryAtPathException: If a directory is found at the path.
            Exception: Any other exception encountered while trying to fetch the object.
        """
        try:
            pickled_value = super(LocalProvider, self).get(path)
            if pickled_value is None:
                return None
            return pickle.loads(pickled_value)
        except DirectoryAtPathException:
            raise
        except FileNotFoundError:
            raise KeyError(path)

    def __setitem__(self, path: str, value: bytes):
        """Sets the object present at the path with the value

        Args:
            path (str): the path relative to the root of the provider.
            value (bytes): the value to be assigned at the path.

        Raises:
            Exception: If unable to set item due to directory at path or permission or space issues.
            FileAtPathException: If the directory to the path is a file instead of a directory.
            ReadOnlyError: If the provider is in read-only mode.
        """
        self.check_readonly()
        full_path = self._check_is_file(path)
        try:
            cache = get_redis_client()
            for k,v in path, value:
                kw = {path: value}
                cache.mset(kw)
        except Exception as err:
            raise S3SetError(err)

    def __delitem__(self, path: str):
        """Delete the object present at the path.

        Args:
            path (str): the path to the object relative to the root of the provider.

        Raises:
            KeyError: If an object is not found at the path.
            DirectoryAtPathException: If a directory is found at the path.
            Exception: Any other exception encountered while trying to fetch the object.
            ReadOnlyError: If the provider is in read-only mode.
        """
        self.check_readonly()
        try:
            r.delete(path)
        except DirectoryAtPathException:
            raise
        except FileNotFoundError:
            raise KeyError

    def __iter__(self):
        """Generator function that iterates over the keys of the provider.

        Yields:
            str: the path of the object that it is iterating over, relative to the root of the provider.
        """
        yield from self._all_keys()

    def __len__(self):
        """Returns the number of files present inside the root of the provider.

        Returns:
            int: the number of files present inside the root.
        """
        keys = redis.allkeys('*')
        for key in keys:
            type = redis.type(key)
            if type == "string":
                val = redis.get(key)
            if type == "hash":
                vals = redis.hgetall(key)
            if type == "zset":
                vals = redis.zrange(key, 0, -1)
            if type == "list":
                vals = redis.lrange(key, 0, -1)
            if type == "set":
                vals = redis.smembers(key)
        return vals

    def _all_keys(self, refresh: bool = False) -> Set[str]:
        """Lists all the objects present at the root of the Provider.

        Args:
            refresh (bool): refresh keys

        Returns:
            set: set of all the objects found at the root of the Provider.
        """
        if self.files is None or refresh:
            p = self.r.pipeline()
            for key in self.r.keys():
                p.hgetall(key)
            key_set = set()
            for h in p.execute():
                key_set.add(h)
            return key_set

    def clear(self, ns: str):
        """Deletes ALL data on the local machine (under self.root). Exercise caution!"""
        """:param ns: str, namespace i.e your:prefix"""
        cache = StrictRedis()
        CHUNK_SIZE = 5000
        self.check_readonly()
        cursor = '0'
        ns_keys = ns + '*'
        while cursor != 0:
            cursor, keys = cache.scan(cursor=cursor, match=ns_keys, count=CHUNK_SIZE)
            if keys:
                cache.delete(*keys)
        return True

    def __contains__(self, key) -> bool:
        return self.r.exists(key)
