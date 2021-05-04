from hub.core.storage.provider import Provider
from hub.core.storage.s3.s3_boto_storage import S3BotoStorage


class S3BotoProvider(Provider):
    def __init__(
        self,
        path,
        aws_access_key_id=None,
        aws_secret_access_key=None,
        aws_session_token=None,
        endpoint_url=None,
        aws_region=None,
    ):
        root = "s3://" + path
        self.d = S3BotoStorage(
            self,
            root,
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            aws_session_token=aws_session_token,
            aws_region=aws_region,
            endpoint_url=endpoint_url,
        )

    def __getitem__(self, path):
        return self.d[path]

    def __setitem__(self, path, value):
        self.d[path] = value

    def __iter__(self):
        yield from self.d.items()

    def __delitem__(self, path):
        del self.d[path]

    def __len__(self):
        return len(self.d.keys)


def __init__(self, *args, expiration=None, **kwargs):
    super().__init__(*args, **kwargs)
    self._args = args
    self._kwargs = kwargs
    self.expiration = expiration


def get_mapper(self, root: str, check=False, create=False):
    root = "s3://" + root
    client_kwargs = self._kwargs.get("client_kwargs")
    endpoint_url = client_kwargs and client_kwargs.get("endpoint_url") or None
    aws_region = client_kwargs and client_kwargs.get("region_name") or None
    return S3BotoStorage(
        self,
        root,
        aws_access_key_id=self._kwargs.get("key"),
        aws_secret_access_key=self._kwargs.get("secret"),
        aws_session_token=self._kwargs.get("token"),
        aws_region=aws_region,
        endpoint_url=endpoint_url,
    )
