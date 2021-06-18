import time
import boto3
import botocore  # type: ignore
import posixpath
from typing import Optional
from botocore.session import ComponentLocator
from hub.client.client import HubBackendClient
from hub.core.storage.provider import StorageProvider
from hub.util.exceptions import S3DeletionError, S3GetError, S3ListError, S3SetError
import hub


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
        token: Optional[str] = None,
        max_pool_connections: int = 50,
    ):
        """Initializes the S3Provider

        Example:
            s3_provider = S3Provider("snark-test/benchmarks")

        Args:
            root (str): The root of the provider. All read/write request keys will be appended to root.
            aws_access_key_id (optional, str): Specifies the AWS access key used as part of the credentials to
                authenticate the user.
            aws_secret_access_key (optional, str): Specifies the AWS secret key used as part of the credentials to
                authenticate the user.
            aws_session_token (optional, str): Specifies an AWS session token used as part of the credentials to
                authenticate the user.
            endpoint_url (optional, str): The complete URL to use for the constructed client.
                This needs to be provided for cases in which you're interacting with MinIO, Wasabi, etc.
            aws_region (optional, str): Specifies the AWS Region to send requests to.
            token (optional, str): required for activeloop authentication
            max_pool_connections (int): The maximum number of connections to keep in a connection pool.
                If this value is not set, the default value of 10 is used.
        """
        self.root = root
        self.aws_access_key_id = aws_access_key_id
        self.aws_secret_access_key = aws_secret_access_key
        self.aws_session_token = aws_session_token
        self.aws_region: Optional[str] = aws_region
        self.endpoint_url: Optional[str] = endpoint_url
        self.max_pool_connections = max_pool_connections
        self.expiration: Optional[str] = None
        self.tag: Optional[str] = None
        self.token: Optional[str] = token

        self._initialize_s3_parameters()

    def __setitem__(self, path, content):
        """Sets the object present at the path with the value

        Args:
            path (str): the path relative to the root of the S3Provider.
            content (bytes): the value to be assigned at the path.

        Raises:
            S3SetError: Any S3 error encountered while setting the value at the path.
            ReadOnlyError: If the provider is in read-only mode.
        """
        self.check_readonly()
        self._check_update_creds()
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
            path (str): the path relative to the root of the S3Provider.

        Returns:
            bytes: The bytes of the object present at the path.

        Raises:
            KeyError: If an object is not found at the path.
            S3GetError: Any other error other than KeyError while retrieving the object.
            ReadOnlyError: If the provider is in read-only mode.
        """
        self.check_readonly()
        self._check_update_creds()
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
            path (str): the path to the object relative to the root of the S3Provider.

        Raises:
            S3DeletionError: Any S3 error encountered while deleting the object. Note: if the object is not found, s3
                won't raise KeyError.
            ReadOnlyError: If the provider is in read-only mode.
        """
        self.check_readonly()
        self._check_update_creds()
        try:
            path = posixpath.join(self.path, path)
            self.client.delete_object(Bucket=self.bucket, Key=path)
        except Exception as err:
            raise S3DeletionError(err)

    def _list_keys(self):
        """Helper function that lists all the objects present at the root of the S3Provider.

        Returns:
            list: list of all the objects found at the root of the S3Provider.

        Raises:
            S3ListError: Any S3 error encountered while listing the objects.
        """
        self._check_update_creds()
        try:
            # TODO boto3 list_objects only returns first 1000 objects
            items = self.client.list_objects_v2(Bucket=self.bucket, Prefix=self.path)
            if items["KeyCount"] <= 0:
                return []
            items = items["Contents"]
            names = [item["Key"] for item in items]
            # removing the prefix from the names
            len_path = len(self.path.split("/"))
            names = ["/".join(name.split("/")[len_path:]) for name in names]
            return names
        except Exception as err:
            raise S3ListError(err)

    def __len__(self):
        """Returns the number of files present at the root of the S3Provider. This is an expensive operation.

        Returns:
            int: the number of files present inside the root.

        Raises:
            S3ListError: Any S3 error encountered while listing the objects.
        """
        self._check_update_creds()
        return len(self._list_keys())

    def __iter__(self):
        """Generator function that iterates over the keys of the S3Provider.

        Yields:
            str: the name of the object that it is iterating over.
        """
        self._check_update_creds()
        yield from self._list_keys()

    def clear(self):
        """Deletes ALL data on the s3 bucket (under self.root). Exercise caution!"""
        self.check_readonly()
        self._check_update_creds()
        if self.resource is not None:
            bucket = self.resource.Bucket(self.bucket)
            bucket.objects.filter(Prefix=self.path).delete()
        else:
            super().clear()

    def __getstate__(self):
        state = {}
        state["root"] = self.root
        state["aws_access_key_id"] = self.aws_access_key_id
        state["aws_secret_access_key"] = self.aws_secret_access_key
        state["aws_session_token"] = self.aws_session_token
        state["aws_region"] = self.aws_region
        state["endpoint_url"] = self.endpoint_url
        state["max_pool_connections"] = self.max_pool_connections
        state["expiration"] = self.expiration
        state["tag"] = self.tag
        return state

    def __setstate__(self, state):
        self.__dict__ = state
        self._initialize_s3_parameters()

    def _set_bucket_and_path(self):
        root = self.root.replace("s3://", "")
        self.bucket = root.split("/")[0]
        self.path = "/".join(root.split("/")[1:])

    def _set_hub_creds_info(self, tag: str, expiration: str):
        """Sets the tag and expiration of the credentials. These are only relevant to datasets using Hub storage.
        This info is used to fetch new credentials when the temporary 12 hour credentials expire.

        Args:
            tag (str): The Hub tag of the dataset in the format username/dataset_name.
            expiration (str): The time at which the credentials expire.
        """
        self.tag = tag
        self.expiration = expiration

    def _initialize_s3_parameters(self):
        self._set_bucket_and_path()

        self.client_config = botocore.config.Config(
            max_pool_connections=self.max_pool_connections,
        )

        if self.aws_access_key_id is None and self.aws_secret_access_key is None:
            self._locate_and_load_creds()

        self._set_s3_client_and_resource()

    def _check_update_creds(self):
        """If the client has an expiration time, check if creds are expired and fetch new ones.
        This would only happen for datasets stored on Hub storage for which temporary 12 hour credentials are generated.
        """
        if self.expiration and float(self.expiration) < time.time():
            client = HubBackendClient(self.token)
            org_id, ds_name = self.tag.split("/")

            if hasattr(self, "read_only") and self.read_only:
                mode = "r"
            else:
                mode = "a"
            url, creds, mode, expiration = client.get_dataset_credentials(
                org_id, ds_name, mode
            )
            self.expiration = expiration
            self._set_s3_client_and_resource(
                creds.get("aws_access_key_id"),
                creds.get("aws_secret_access_key"),
                creds.get("aws_session_token"),
            )

    def _locate_and_load_creds(self):
        session = boto3._get_default_session()._session
        component_locator = ComponentLocator()
        component_locator.lazy_register_component(
            "credential_provider", session._create_credential_resolver
        )
        credentials = component_locator.get_component(
            "credential_provider"
        ).load_credentials()
        if credentials is not None:
            self.aws_access_key_id = credentials.access_key
            self.aws_secret_access_key = credentials.secret_key
            self.aws_session_token = credentials.token
            self.aws_region = session._resolve_region_name(
                self.aws_region, self.client_config
            )

    def _set_s3_client_and_resource(
        self, aws_access_key_id=None, aws_secret_access_key=None, aws_session_token=None
    ):
        key = aws_access_key_id or self.aws_access_key_id
        secret = aws_secret_access_key or self.aws_secret_access_key
        token = aws_session_token or self.aws_session_token

        self.client = boto3.client(
            "s3",
            aws_access_key_id=key,
            aws_secret_access_key=secret,
            aws_session_token=token,
            config=self.client_config,
            endpoint_url=self.endpoint_url,
            region_name=self.aws_region,
        )

        self.resource = boto3.resource(
            "s3",
            aws_access_key_id=key,
            aws_secret_access_key=secret,
            aws_session_token=token,
            config=self.client_config,
            endpoint_url=self.endpoint_url,
            region_name=self.aws_region,
        )
