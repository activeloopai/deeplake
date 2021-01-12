from collections.abc import MutableMapping
import posixpath

import boto3
import botocore
from botocore.exceptions import ClientError
from s3fs import S3FileSystem

from hub.exceptions import S3Exception
from hub.log import logger
from hub.defaults import MAX_POOL_CONNECTIONS


class S3Storage(MutableMapping):
    def __init__(
        self,
        s3fs: S3FileSystem,
        client,
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

    def __setitem__(self, path, content):
        try:
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

    def __getitem__(self, path):
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

    def __delitem__(self, path):
        try:
            path = posixpath.join(self.bucketpath, path)
            self.s3fs.rm(path, recursive=True)
        except Exception as err:
            logger.error(err)
            raise S3Exception(err)

    def __len__(self):
        return len(self.s3fs.ls(self.bucketpath, detail=False, refresh=True))

    def __iter__(self):
        items = self.s3fs.ls(self.bucketpath, detail=False, refresh=True)
        yield from [item[len(self.bucketpath) + 1 :] for item in items]
