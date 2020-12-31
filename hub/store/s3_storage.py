from collections.abc import MutableMapping
from hub.store.tests.test_s3_storage import test_s3_storage
import posixpath

import boto3
import botocore
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
    def __init__(self, s3fs: S3FileSystem, url: str=None, public=False, aws_access_key_id=None,
                 aws_secret_access_key=None, aws_session_token=None, parallel=25, endpoint_url = None):
        self.s3fs = s3fs
        self.url = url
        self.endpoint_url = endpoint_url
        self.bucket = url.split('/')[1]
        self.path = '/'.join(url.split('/')[2:])
        self.protocol = 'object'

        print(self.bucket, self.path)

        client_config = botocore.config.Config(
            max_pool_connections=parallel,
        )

        self.client = boto3.client(
            's3',
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
                'Bucket': self.bucket,
                'Body': content,
                'Key': path,
                'ContentType': ('application/octet-stream'),
            }

            self.client.put_object(**attrs)
        except Exception as err:
            logger.error(err)
            raise S3Exception(err)

    @retry
    def __getitem__(self, path):
        path = posixpath.join(self.path, path)
        try:
            resp = self.client.get_object(
                Bucket=self.bucket,
                Key=path,
            )
            x = resp['Body'].read()
            return x
        except botocore.exceptions.ClientError as err:
            if err.response['Error']['Code'] == 'NoSuchKey':
                return None
            else:
                logger.error(err)
                raise S3Exception(err)

    @retry
    def __delitem__(self, path):
        path = posixpath.join(self.path, path)
        try:
            self.client.delete_object(self.bucket, path)
        except botocore.exceptions.ClientError as err:
            if err.response['Error']['Code'] == 'NoSuchKey':
                return None
            else:
                logger.error(err)
                raise S3Exception(err)

    @retry
    def __len__(self):
        return len(self.s3fs.listdir(self.path, detail=False))

    @retry
    def __iter__(self):
        yield from self.s3fs.listdir(self.path, detail=False)
