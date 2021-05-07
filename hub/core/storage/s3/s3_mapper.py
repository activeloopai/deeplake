from collections.abc import MutableMapping
import posixpath
import boto3
import botocore
from botocore.exceptions import ClientError
from typing import Optional


class S3Mapper(MutableMapping):
    def __init__(
        self,
        url: str,
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
        aws_session_token: Optional[str] = None,
        endpoint_url: Optional[str] = None,
        aws_region: Optional[str] = None,
        parallel: Optional[int] = 25,
    ):
        self.url = url
        self.aws_region = aws_region
        self.endpoint_url = endpoint_url

        # url should be "bucket_name/xyz/abc/..."
        self.bucket = url.split("/")[0]
        self.path = "/".join(url.split("/")[1:])
        self.client_config = botocore.config.Config(max_pool_connections=parallel,)

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
        except Exception:  # TODO better exceptions
            raise

    def __getitem__(self, path):
        try:
            path = posixpath.join(self.path, path)
            resp = self.client.get_object(Bucket=self.bucket, Key=path,)
            return resp["Body"].read()
        except ClientError as err:
            if err.response["Error"]["Code"] == "NoSuchKey":
                raise KeyError(err)
            raise
        except Exception as err:  # TODO better exceptions
            raise

    def __delitem__(self, path):
        try:
            path = posixpath.join(self.path, path)
            self.client.delete_object(Bucket=self.bucket, Key=path)
        except Exception:  # TODO better exceptions
            raise

    def _list_objects(self):
        # TODO boto3 list_objects only returns first 1000 objects
        items = self.client.list_objects_v2(Bucket=self.bucket, Prefix=self.path)
        items = items["Contents"]
        names = [item["Key"] for item in items]
        # removing the prefix from the names
        len_path = len(self.path.split("/"))
        names = ["/".join(name.split("/")[len_path:]) for name in names]
        return names

    def __len__(self):
        names = self._list_objects()
        return len(names)

    def __iter__(self):
        names = self._list_objects()
        yield from names

