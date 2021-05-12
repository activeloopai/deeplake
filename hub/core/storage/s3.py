import boto3
import botocore  # type: ignore
import posixpath
from typing import Optional
from hub.core.storage.provider import StorageProvider
from hub.util.exceptions import S3GetError, S3SetError, S3DeletionError, S3ListError


class S3Provider(StorageProvider):
    """Provider class for using S3 storage."""

    def __init__(
        self,
        root: str,
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
        aws_session_token: Optional[str] = None,
        endpoint_url: Optional[str] = None,
        aws_region: Optional[str] = None,
        max_pool_connections: Optional[int] = 10,
    ):
        self.aws_region = aws_region
        self.endpoint_url = endpoint_url

        self.bucket = root.split("/")[0]
        self.path = "/".join(root.split("/")[1:])

        self.client_config = botocore.config.Config(
            max_pool_connections=max_pool_connections,
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
        """Sets the object present at the path with the value

        Args:
            path (str): the path relative to the root of the mapper.
            content (bytes): the value to be assigned at the path.

        Raises:
            S3SetError: Any S3 error encountered while setting the value at the path.
        """
        try:
            path = posixpath.join(self.path, path)
            content = bytearray(memoryview(content))
            self.client.put_object(
                Bucket=self.bucket,
                Body=content,
                Key=path,
                ContentType="application/octet-stream",  # signifies binary data
            )
        except Exception as err:
            raise S3SetError(err)

    def __getitem__(self, path):
        """Gets the object present at the path.

        Args:
            path (str): the path relative to the root of the mapper.

        Returns:
            bytes: The bytes of the object present at the path.

        Raises:
            KeyError: If an object is not found at the path.
            S3GetError: Any other error other than KeyError while retrieving the object.
        """
        try:
            path = posixpath.join(self.path, path)
            resp = self.client.get_object(
                Bucket=self.bucket,
                Key=path,
            )
            return resp["Body"].read()
        except botocore.exceptions.ClientError as err:
            if err.response["Error"]["Code"] == "NoSuchKey":
                raise KeyError(err)
            raise S3GetError(err)
        except Exception as err:
            raise S3GetError(err)

    def __delitem__(self, path):
        """Delete the object present at the path.

        Args:
            path (str): the path to the object relative to the root of the mapper.

        Raises:
            S3DeletionError: Any S3 error encountered while deleting the object. Note: if the object is not found, s3 won't raise KeyError.
        """
        try:
            path = posixpath.join(self.path, path)
            self.client.delete_object(Bucket=self.bucket, Key=path)
        except Exception as err:
            raise S3DeletionError(err)

    def _list_keys(self):
        """Helper function to list all the objects present at the root of the mapper.

        Returns:
            list: list of all the objects found at the root of the mapper.

        Raises:
            S3ListError: Any S3 error encountered while listing the objects.
        """
        try:
            # TODO boto3 list_objects only returns first 1000 objects
            items = self.client.list_objects_v2(Bucket=self.bucket, Prefix=self.path)
            items = items["Contents"]
            names = [item["Key"] for item in items]
            # removing the prefix from the names
            len_path = len(self.path.split("/"))
            names = ["/".join(name.split("/")[len_path:]) for name in names]
            return names
        except Exception as err:
            raise S3ListError(err)

    def __len__(self):
        """Returns the number of files present inside the root of the mapper. This is an expensive operation.

        Returns:
            int: the number of files present inside the root.

        Raises:
            S3ListError: Any S3 error encountered while listing the objects.
        """
        return len(self._list_keys())

    def __iter__(self):
        """Generator function that iterates over the keys of the mapper.

        Yields:
            str: the name of the object that it is iterating over.
        """
        yield from self._list_keys()
