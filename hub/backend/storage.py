from google.cloud import storage
from google.oauth2 import service_account
import boto3
import os
import botocore
import tenacity
from hub.log import logger
from hub.utils.store_control import StoreControlClient
from hub.exceptions import FileSystemException, S3Exception
from hub.config import CACHE_FILE_PATH
import gzip
retry = tenacity.retry(
    reraise=True,
    stop=tenacity.stop_after_attempt(6),
    wait=tenacity.wait_random_exponential(0.4, 50.0),
)


class Storage(object):
    def __init__(self):
        self.protocol = ''

    def get(self, path):
        raise NotImplementedError

    def put(self, path, file):
        raise NotImplementedError

    def delete(self, path):
        raise NotImplementedError

    def exist(self, path):
        raise NotImplementedError


class StorageFactory(Storage):
    def __init__(self, protocols, caching=False):
        super(Storage, self).__init__()
        if isinstance(protocols, str):
            protocols = [protocols]

        self.storages = list(map(self.map, protocols))
        self.protocol = 'factory_{}'.format(
            '_'.join(map(lambda x: x.protocol, self.storages)))
        self.caching = caching

    def __map(self, name):
        # S3 object storage
        if isinstance(name, str) and name == 's3':
            return S3()
        # GS object storage
        elif isinstance(name, str) and name == 'gs':
            return GS()
        # FileSystem object Storage
        elif isinstance(name, str) and name == 'fs':
            return FS()

        elif isinstance(name, Storage):
            return name

        raise Exception('Backend not found {}'.format(name))

    def map(self, name):
        storage = self.__map(name)
        return storage
        #return GZipStorage(storage)

    def get(self, path):
        # Return first object found
        for store in self.storages:

            obj = store.get(path)
            if obj is not None:

                # caching
                if self.caching:
                    ith = self.storages.index(store)
                    for sub_store in self.storages[:ith]:
                        sub_store.put(path, obj)

                return obj

    def put(self, path, file):
        # Store in each object storage
        for store in self.storages:
            store.put(path, file)


class FS(Storage):
    def __init__(self, bucket=None):
        super(Storage, self).__init__()
        if bucket is None:
            bucket = CACHE_FILE_PATH
        self.bucket = bucket
        self.protocol = 'file'

    def get(self, path):
        path = os.path.join(self.bucket, path)
        try:
            with open(path, 'rb') as f:
                data = f.read()
            return data
        except IOError as err:
            logger.debug(err)
        return None

    def put(self, path, content):
        path = os.path.join(self.bucket, path)
        try:
            folder = '/'.join(path.split('/')[:-1])
            if not os.path.isdir(folder):
                os.makedirs(folder)

            with open(path, 'wb') as f:
                f.write(content)
        except IOError as err:
            logger.debug(err)
            raise FileSystemException


class S3(Storage):
    def __init__(self, bucket=None, public=False, aws_access_key_id=None, aws_secret_access_key=None, parallel=25):
        super(Storage, self).__init__()

        if bucket is None:
            self.bucket = bucket=StoreControlClient.get_config()['BUCKET']
        self.bucket = bucket
        self.protocol = 'object'

        client_config = botocore.config.Config(
            max_pool_connections=parallel,
        )

        if aws_access_key_id is None:
            aws_access_key_id = StoreControlClient.get_config(public)['AWS_ACCESS_KEY_ID']
        if aws_secret_access_key is None:
            aws_secret_access_key = StoreControlClient.get_config(public)['AWS_SECRET_ACCESS_KEY']

        self.client = boto3.client(
            's3',
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            config=client_config
            # endpoint_url=URL,
        )

    @retry
    def put(self, path, content):
        path = path.replace('s3://{}/'.format(self.bucket), '')
        try:
            attrs = {
                'Bucket': self.bucket,
                'Body': content,
                'Key': path,
                'ContentType': ('application/octet-stream'),
            }

            # if compress:
            #    attrs['ContentEncoding'] = 'gzip'
            # if cache_control:
            #    attrs['CacheControl'] = cache_control

            self.client.put_object(**attrs)
        except Exception as err:
            logger.error(err)
            raise S3Exception(err)

    @retry
    def get(self, path):
        path = path.replace('s3://{}/'.format(self.bucket), '')
        try:
            resp = self.client.get_object(
                Bucket=self.bucket,
                Key=path,
            )
            return resp['Body'].read()
        except botocore.exceptions.ClientError as err:
            if err.response['Error']['Code'] == 'NoSuchKey':
                return None
            else:
                logger.error(err)
                raise S3Exception(err)

class GS(Storage):
    def __init__(self):
        service_account.Credentials.from_service_account_info()
        self.protocol = 'gs'
        self.__bucket_name = 'snark_waymo_open_dataset'
        self.__bucket = storage.Client().get_bucket(self.__bucket_name)

    @retry
    def get(self, path):
        try:
            # print(path)
            # path = path.replace('gs://{}/'.format(self.__bucket), '')
            blob = self.__bucket.blob(path)
            return blob.download_as_string()
        except Exception as ex:
            return None
            # raise S3Exception(ex)

    @retry
    def put(self, path, content):
        try:
            # path = path.replace('gs://{}/'.format(self.__bucket), '')
            blob = self.__bucket.blob(path)
            blob.upload_from_string(content)
        except Exception as ex:
            raise S3Exception(ex)

    @retry
    def delete(self, path):
        try:
            # path = path.replace('gs://{}/'.format(self.__bucket), '')
            self.__bucket.delete_blob(path)
        except:
            pass

    @retry
    def exist(self, path):
        try:
            # path = path.replace('gs://{}/'.format(self.__bucket), '')
            blob = self.__bucket.blob(path)
            return blob.exists()
        except: 
            pass
        
class GZipStorage(Storage):
    def __init__(self, internal_storage):
        self.__internar_storage = internal_storage
        self.protocol = internal_storage.protocol
    
    def get(self, path):
        content = self.__internar_storage.get(path);
        data = gzip.decompress(content)
        return data

    def put(self, path, content):
        data = gzip.compress(content, compresslevel=1)
        self.__internar_storage.put(path, data)

    def delete(self, path):
        self.__internar_storage.delete(path)
    
    def exist(self, path):
        return self.__internar_storage.exist(path)
