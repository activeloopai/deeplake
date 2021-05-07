from typing import Optional
from hub.core.storage.provider_mapper import ProviderMapper
from hub.core.storage.s3.s3_mapper import S3Mapper


class S3Provider(ProviderMapper):
    """
    Provider class for using S3 storage.
    """

    def __init__(
        self,
        root: str,
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
        aws_session_token: Optional[str] = None,
        endpoint_url: Optional[str] = None,
        aws_region: Optional[str] = None,
        max_pool_connections: Optional[int] = 25,
    ):
        """
        Initializes the S3Provider

        Example:
            s3_provider = S3Provider("snark-test/benchmarks")

        Args:
            root (str): The root of the provider. All read/write request keys will be appended to root.
            aws_access_key_id (optional, str): Specifies the AWS access key used as part of the credentials to authenticate the user.
            aws_secret_access_key (optional, str): Specifies the AWS secret key used as part of the credentials to authenticate the user.
            aws_session_token (optional, str): Specifies an AWS session token used as part of the credentials to authenticate the user.
            endpoint_url (optional, str): The complete URL to use for the constructed client.
                This needs to be provided for cases in which you're interacting with MinIO, Wasabi, etc.
            aws_region (optional, str): Specifies the AWS Region to send requests to.
            max_pool_connections (optional, int): The maximum number of connections to keep in a connection pool.
                If this value is not set, the default value of 10 is used.

        Returns:
            None

        Raises:
            None
        """
        # passing no creds, would cause boto to read credentials
        self.mapper = S3Mapper(
            root,
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            aws_session_token=aws_session_token,
            aws_region=aws_region,
            endpoint_url=endpoint_url,
            max_pool_connections=max_pool_connections,
        )
