import logging
import boto3
import cloudvolume
import pprint
#logger = logging.getLogger()
#logger.addHandler(logging.StreamHandler()) # Writes to console
#logger.setLevel(logging.DEBUG)
debug = False
from meta.utils.store_control import StoreControlClient

if not debug:
  for log in ['boto3', 'botocore', 's3transfer', 'urllib3', 'python_jsonschema_objects']:
    logging.getLogger(log).setLevel(logging.CRITICAL)

creds = StoreControlClient().get_config()
ACCESS_KEY = creds['AWS_ACCESS_KEY_ID']
SECRET_KEY = creds['AWS_SECRET_ACCESS_KEY']
#URL = 'http://localhost:8000'

client = boto3.client(
  's3',
  aws_access_key_id=ACCESS_KEY,
  aws_secret_access_key=SECRET_KEY,
  #endpoint_url=URL,
)

import tenacity
retry = tenacity.retry(
  reraise=True, 
  stop=tenacity.stop_after_attempt(7), 
  wait=tenacity.wait_random_exponential(0.5, 60.0),
)


class S3ConnectionPool(cloudvolume.ConnectionPool):
  def __init__(self, service, bucket):
    self.service = service
    self.bucket = bucket
    #self.credentials = aws_credentials(bucket, service)
    super(S3ConnectionPool, self).__init__()

  @retry
  def _create_connection(self):
    if self.service in ('aws', 's3'):
      return client
    else:
      raise UnsupportedProtocolError("{} unknown. Choose from 's3' or 'matrix'.", self.service)
      
  def close(self, conn):
    try:
      return conn.close()
    except AttributeError:
      pass # AttributeError: 'S3' object has no attribute 'close' on shutdown



class S3ConnectionPool2(cloudvolume.ConnectionPool):
  def __init__(self, service, bucket):
    self.service = service
    self.bucket = bucket
    #self.credentials = aws_credentials(bucket, service)
    super(S3ConnectionPool2, self).__init__()

  def _create_connection(self):
    if self.service in ('aws', 's3'):
      return client
    else:
      raise ServiceUnknownException("{} unknown. Choose from 's3'.")

  def close(self, conn):
    try:
      return conn.close()
    except AttributeError:
      pass # AttributeError: 'S3' object has no attribute 'close' on shutdown

from cloudvolume import CloudVolume
from cloudvolume.storage.storage_interfaces import keydefaultdict
#cloudvolume.storage.S3_POOL = keydefaultdict(lambda service: keydefaultdict(lambda bucket_name: S3ConnectionPool(service, bucket_name)))
