"""
License:
This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""

from collections.abc import MutableMapping
import posixpath

import boto3
import botocore
from botocore.exceptions import ClientError
from s3fs import S3FileSystem

from hub_v1.exceptions import S3Exception
from hub_v1.log import logger
from hub_v1.client.hub_control import HubControlClient
import time


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
        aws_region=None,
        expiration=None,
    ):
        self.s3fs = s3fs
        self.root = {}
        self.url = url
        self.public = public
        self.parallel = parallel
        self.aws_region = aws_region
        self.endpoint_url = endpoint_url
        self.expiration = expiration
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

        self.resource = boto3.resource(
            "s3",
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            aws_session_token=aws_session_token,
            config=self.client_config,
            endpoint_url=self.endpoint_url,
            region_name=self.aws_region,
        )

    def check_update_creds(self):
        if self.expiration and float(self.expiration) < time.time():
            details = HubControlClient().get_credentials()
            self.expiration = details["expiration"]
            self.client = boto3.client(
                "s3",
                aws_access_key_id=details["access_key"],
                aws_secret_access_key=details["secret_key"],
                aws_session_token=details["session_token"],
                config=self.client_config,
                endpoint_url=self.endpoint_url,
                region_name=self.aws_region,
            )
            self.resource = boto3.resource(
                "s3",
                aws_access_key_id=details["access_key"],
                aws_secret_access_key=details["secret_key"],
                aws_session_token=details["session_token"],
                config=self.client_config,
                endpoint_url=self.endpoint_url,
                region_name=self.aws_region,
            )

    def __setitem__(self, path, content):
        self.check_update_creds()
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
        except Exception as err:
            logger.error(err)
            raise S3Exception(err)

    def __getitem__(self, path):
        self.check_update_creds()
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
        self.check_update_creds()
        try:
            path = posixpath.join(self.bucketpath, path)
            self.s3fs.rm(path, recursive=True)
        except Exception as err:
            logger.error(err)
            raise S3Exception(err)

    def __len__(self):
        self.check_update_creds()
        return len(self.s3fs.ls(self.bucketpath, detail=False, refresh=True))

    def __iter__(self):
        self.check_update_creds()
        items = self.s3fs.ls(self.bucketpath, detail=False, refresh=True)
        yield from [item[len(self.bucketpath) + 1 :] for item in items]
