import boto3
import botocore

from .storage import Storage

class AmazonS3(Storage):
    def __init__(self, bucket: str, aws_access_key_id: str, aws_secret_access_key: str):
        super().__init__()
        self._bucket_name = bucket
        self._client = boto3.client('s3', aws_access_key_id = aws_access_key_id, aws_secret_access_key = aws_secret_access_key)
        self._resource = boto3.resource('s3', aws_access_key_id = aws_access_key_id, aws_secret_access_key = aws_secret_access_key)
        self._bucket = self._resource.Bucket(bucket)

    def get(self, path: str) -> bytes:
        return self._bucket.Object(path).get()['Body'].read()

    def put(self, path: str, content: bytes) -> None:
        attrs = {
            'Bucket': self._bucket_name,
            'Body': content,
            'Key': path,
            'ContentType': ('application/octet-stream'),
        }
        self._client.put_object(**attrs)
        # self._bucket.Object(path).put({'Body': content, 'ContentType': ('application/octet-stream')})
    
    def exists(self, path: str) -> bool:
        try:
            self._resource.Object(self._bucket_name, path).get()
        except botocore.exceptions.ClientError as ex:
            if ex.response['Error']['Code'] in ['NoSuchKey', '404']:
                return False
            else:
                raise
        return True
    
    def delete(self, path: str):
        self._client.delete_object(Bucket=self._bucket_name, Key=path)