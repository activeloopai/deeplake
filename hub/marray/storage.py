import boto3
import botocore
import tenacity
from hub.log import logger
from hub.utils.store_control import StoreControlClient

retry = tenacity.retry(
    reraise=True, 
    stop=tenacity.stop_after_attempt(7), 
    wait=tenacity.wait_random_exponential(0.5, 60.0),
)
  
class Storage(object):
    def __init__(self):
        return
    
    def get(self, path):
        raise NotImplementedError

    def put(self, path, file):
        raise NotImplementedError


class S3(Storage):
    def __init__(self, bucket, public=False):
        super(Storage, self).__init__()
        #TODO add pools
        self.bucket = StoreControlClient.get_config(public)['BUCKET']
        self.client = boto3.client(
            's3',
            aws_access_key_id=StoreControlClient.get_config(public)['AWS_ACCESS_KEY_ID'],
            aws_secret_access_key=StoreControlClient.get_config(public)['AWS_SECRET_ACCESS_KEY'],
            #endpoint_url=URL,
        )
    
    @retry
    def put(self, path, content, content_type, compress=False, cache_control=False):
        try:
            attrs = {
                'Bucket': self.bucket,
                'Body': content,
                'Key': path.replace('s3://{}/'.format(self.bucket), ''),
                'ContentType': (content_type or 'application/octet-stream'),
            }

            if compress:
                attrs['ContentEncoding'] = 'gzip'
            if cache_control:
                attrs['CacheControl'] = cache_control
            self.client.put_object(**attrs)
        except Exception as err:
            logger.error(err)
            raise
        
    @retry
    def get(self, path):
        try:
            resp = self.client.get_object(
                Bucket=self.bucket,
                Key=path.replace('s3://{}/'.format(self.bucket), ''),
            )
            return resp['Body'].read()
        except botocore.exceptions.ClientError as err:
            if err.response['Error']['Code'] == 'NoSuchKey':
                return None
            else:
                logger.error(err)
                raise
