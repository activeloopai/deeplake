from collections.abc import MutableMapping
import posixpath
import boto3
import botocore
from s3fs import S3FileSystem


class S3BotoStorage(MutableMapping):
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
        aws_region=None,
    ):
        self.s3fs = s3fs
        self.root = {}
        self.url = url
        self.public = public
        self.parallel = parallel
        self.aws_region = aws_region
        self.endpoint_url = endpoint_url
        self.bucket = url.split("/")[2]
        self.path = "/".join(url.split("/")[3:])
        if self.bucket == "s3:":
            # FIXME for some reason this is wasabi case here, probably url is something like wasabi://s3://...
            self.bucket = url.split("/")[4]
            self.path = "/".join(url.split("/")[5:])
        self.bucketpath = posixpath.join(self.bucket, self.path)
        self.protocol = "object"

        self.client_config = botocore.config.Config(
            max_pool_connections=parallel,
        )

        self.client = boto3.client(
            "s3",
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            aws_session_token=aws_session_token,
            config=self.client_config,
            endpoint_url=self.endpoint_url,
            region_name=self.aws_region,
        )

    def __setitem__(self, path, content):
        try:
            path = posixpath.join(self.path, path)
            content = bytearray(memoryview(content))
            attrs = {
                "Bucket": self.bucket,
                "Body": content,
                "Key": path,
                "ContentType": ("application/octet-stream"),
            }
            self.client.put_object(**attrs)
        except Exception:
            raise

    def __getitem__(self, path):
        try:
            path = posixpath.join(self.path, path)
            resp = self.client.get_object(
                Bucket=self.bucket,
                Key=path,
            )
            return resp["Body"].read()
        except Exception:
            raise

    def __delitem__(self, path):
        try:
            path = posixpath.join(self.bucketpath, path)
            self.s3fs.rm(path, recursive=True)
        except Exception:
            raise

    def __len__(self):
        return len(self.s3fs.ls(self.bucketpath, detail=False, refresh=True))

    def __iter__(self):
        items = self.s3fs.ls(self.bucketpath, detail=False, refresh=True)
        yield from [item[len(self.bucketpath) + 1 :] for item in items]
