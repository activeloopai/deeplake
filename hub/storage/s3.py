import boto3
import botocore

from .base import Base


class S3(Base):
    def __init__(self, bucket: str, aws_access_key_id: str, aws_secret_access_key: str):

        super().__init__()
        self._bucket_name = bucket
        if aws_access_key_id is None and aws_secret_access_key is None:
            self._client = boto3.client(
                "s3", config=botocore.config.Config(max_pool_connections=128),
            )
            self._resource = boto3.resource("s3")
        else:
            self._client = boto3.client(
                "s3",
                aws_access_key_id=aws_access_key_id,
                aws_secret_access_key=aws_secret_access_key,
                config=botocore.config.Config(max_pool_connections=128),
            )
            self._resource = boto3.resource(
                "s3",
                aws_access_key_id=aws_access_key_id,
                aws_secret_access_key=aws_secret_access_key,
            )
        self._bucket = self._resource.Bucket(bucket)

    def get(self, path: str) -> bytes:
        return self._bucket.Object(path).get()["Body"].read()

    def put(self, path: str, content: bytes) -> None:
        attrs = {
            "Bucket": self._bucket_name,
            "Body": content,
            "Key": path,
            "ContentType": ("application/octet-stream"),
        }
        self._client.put_object(**attrs)

    def exists(self, path: str) -> bool:
        return self.get_or_none(path) is not None

    def delete(self, path: str):
        self._bucket.objects.filter(Prefix=path).delete()

    def get_or_none(self, path):
        try:
            resp = self._client.get_object(Bucket=self._bucket_name, Key=path,)
            return resp["Body"].read()
        except botocore.exceptions.ClientError as err:
            if err.response["Error"]["Code"] == "NoSuchKey":
                return None
            else:
                raise err
