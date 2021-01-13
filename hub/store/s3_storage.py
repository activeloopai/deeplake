from typing import Iterable, List, Union
import array
from collections.abc import MutableMapping
from concurrent.futures.thread import ThreadPoolExecutor
import posixpath

from botocore.exceptions import ClientError
from s3fs import S3FileSystem

from hub.exceptions import S3Exception
from hub.log import logger


ByteLike = Union[bytearray, memoryview, bytes, array.array]


class S3Storage(MutableMapping):
    def __init__(
        self,
        s3fs: S3FileSystem,
        client,
        tpool: ThreadPoolExecutor,
        url: str = None,
    ):
        self.s3fs = s3fs
        self.client = client
        self.root = {}
        self.url = url
        self.bucket = url.split("/")[2]
        self.path = "/".join(url.split("/")[3:])
        if self.bucket == "s3:":
            # FIXME for some reason this is wasabi case here, probably url is something like wasabi://s3://...
            self.bucket = url.split("/")[4]
            self.path = "/".join(url.split("/")[5:])
        self.bucketpath = posixpath.join(self.bucket, self.path)
        self.protocol = "object"
        self.tpool = tpool

    def _setitem(self, path: str, content: ByteLike) -> None:
        try:
            content = bytearray(memoryview(content))
            path = posixpath.join(self.path, path)
            attrs = {
                "Bucket": self.bucket,
                "Body": content,
                "Key": path,
                "ContentType": ("application/octet-stream"),
            }

            self.client.put_object(**attrs)
        except Exception as err:
            logger.error(err)
            raise S3Exception(err)

    def __setitem__(self, path: str, content: ByteLike) -> None:
        self.tpool.submit(self._setitem, path, content).result()

    def _getitem(self, path: str) -> bytes:
        try:
            path = posixpath.join(self.path, path)
            resp = self.client.get_object(
                Bucket=self.bucket,
                Key=path,
            )
            x = resp["Body"].read()
            return x
        except ClientError as err:
            if err.response["Error"]["Code"] == "NoSuchKey":
                raise KeyError(err)
            else:
                raise
        except Exception as err:
            logger.error(err)
            raise S3Exception(err)

    def __getitem__(self, path: str) -> bytes:
        return self.tpool.submit(self._getitem, path).result()

    def _delitem(self, path: str) -> None:
        try:
            path = posixpath.join(self.bucketpath, path)
            self.s3fs.rm(path, recursive=False)
        except Exception as err:
            logger.error(err)
            raise S3Exception(err)

    def __delitem__(self, path: str) -> None:
        self.tpool.submit(self._delitem, path).result()

    def _len(self) -> int:
        return len(self.s3fs.ls(self.bucketpath, detail=False, refresh=True))

    def __len__(self) -> int:
        return self.tpool.submit(self._len).result()

    def _iter(self) -> List[str]:
        return self.s3fs.ls(self.bucketpath, detail=False, refresh=True)

    def __iter__(self) -> Iterable[str]:
        items = self.tpool.submit(self._iter).result()
        yield from [item[len(self.bucketpath) + 1 :] for item in items]
