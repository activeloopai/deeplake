from collections.abc import MutableMapping
import posixpath

import boto3
import botocore
from botocore.exceptions import ClientError
import tenacity
from s3fs import S3FileSystem

from hub.exceptions import S3Exception
from hub.log import logger

retry = tenacity.retry(
    reraise=True,
    stop=tenacity.stop_after_attempt(6),
    wait=tenacity.wait_random_exponential(0.4, 50.0),
)


class S3Storage(MutableMapping):
    def __init__(
        self,
        s3fs: S3FileSystem,
        url: str = None,
        public=False,
        aws_access_key_id=None,
        aws_secret_access_key=None,
        aws_session_token=None,
        parallel=25,
        endpoint_url=None,
    ):
        self.s3fs = s3fs
        self.root = url
        self.url = url
        self.endpoint_url = endpoint_url
        self.bucket = url.split("/")[2]
        self.path = "/".join(url.split("/")[3:])
        self.bucketpath = posixpath.join(self.bucket, self.path)
        self.protocol = "object"

        client_config = botocore.config.Config(
            max_pool_connections=parallel,
        )

        self.client = boto3.client(
            "s3",
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            aws_session_token=aws_session_token,
            config=client_config,
            endpoint_url=endpoint_url,
        )

        self.resource = boto3.resource(
            "s3",
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            aws_session_token=aws_session_token,
            config=client_config,
            endpoint_url=endpoint_url,
        )

    @retry
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

    @retry
    def __getitem__(self, path, default=None):
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
                if default is not None:
                    return default
                else:
                    raise
            else:
                raise
        except Exception as err:
            logger.error(err)
            raise S3Exception(err)

    @retry
    def __delitem__(self, path):
        try:
            path = posixpath.join(self.bucketpath, path)
            self.s3fs.rm(path, recursive=True)
        except Exception as err:
            logger.error(err)
            raise S3Exception(err)

    @retry
    def __len__(self):
        return len(self.s3fs.ls(self.bucketpath, detail=False, refresh=True))

    @retry
    def __iter__(self):
        items = self.s3fs.ls(self.bucketpath, detail=False, refresh=True)
        yield from [item[len(self.bucketpath) + 1 :] for item in items]
