import boto3
from botocore.exceptions import ClientError
import random
import string
from hub.log import logger


class Verify(object):
    def __init__(self, access_key, secret_key):
        self.creds = {
            'AWS_ACCESS_KEY_ID': access_key,
            'AWS_SECRET_ACCESS_KEY': secret_key,
            'BUCKET': ''
        }

        self.client = client = boto3.client(
            's3',
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            region_name='us-east-1'
        )
        self.prefix = 'snark-hub'

    def randomString(self, stringLength=6):
        """Generate a random string of fixed length """
        letters = string.ascii_lowercase
        return ''.join(random.choice(letters) for i in range(stringLength))

    def verify_aws(self, bucket):
        # check if AWS credentials have access to any snark-hub-... bucket
        if not bucket:
            bucket = self.lookup_hub_bucket()
            bucket_name = bucket.replace(self.prefix, '')

        # if still no bucket, then try to create one
        if not bucket:
            bucket = '{}'.format(self.randomString())

        bucket_name = '{}-{}'.format(self.prefix, bucket)

        # Create bucket or verify if it exists and have access
        success = self.create_bucket(bucket_name)
        if success:
            self.creds['BUCKET'] = bucket_name
            return True, self.creds
        return False, None

    def lookup_hub_bucket(self):
        try:
            response = self.client.list_buckets()

            for bucket in response['Buckets']:
                if 'snark-hub-' in bucket["Name"]:
                    return bucket["Name"]
        except ClientError as e:
            logger.error(e)

        return None

    def exist_bucket(self, bucket):
        try:
            response = self.client.list_buckets()
        except ClientError as e:
            logger.error(e)
            return

        for bucket in response['Buckets']:
            if bucket["Name"] == bucket:
                return bucket
        return ''

    def create_bucket(self, bucket_name, region=None):
        """Create an S3 bucket in a specified region

        If a region is not specified, the bucket is created in the S3 default
        region (us-east-1).

        :param bucket_name: Bucket to create
        :param region: String region to create bucket in, e.g., 'us-west-2'
        :return: True if bucket created, else False
        """

        # Create bucket
        try:
            if region is None:
                self.client.create_bucket(Bucket=bucket_name)
            else:
                location = {'LocationConstraint': region}
                self.client.create_bucket(Bucket=bucket_name,
                                          CreateBucketConfiguration=location)
        except ClientError as e:
            logger.error(e)
            return False
        return True
