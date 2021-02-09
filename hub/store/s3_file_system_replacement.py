"""
License:
This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""

from s3fs import S3FileSystem

from hub.store.s3_storage import S3Storage
from zarr import MemoryStore


class S3FileSystemReplacement(S3FileSystem):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._args = args
        self._kwargs = kwargs

    def get_mapper(self, root: str, check=False, create=False):
        root = "s3://" + root
        client_kwargs = self._kwargs.get("client_kwargs")
        endpoint_url = client_kwargs and client_kwargs.get("endpoint_url") or None
        return S3Storage(
            self,
            root,
            aws_access_key_id=self._kwargs.get("key"),
            aws_secret_access_key=self._kwargs.get("secret"),
            aws_session_token=self._kwargs.get("token"),
            endpoint_url=endpoint_url,
        )
