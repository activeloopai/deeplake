
from .storage import Storage
from .filesystem_storage import FileSystemStorage
from .amazon_storage import AmazonStorage

class StorageFactory():
    @staticmethod
    def filesystem(dir: str) -> Storage:
        return FileSystemStorage(dir)
    
    @staticmethod
    def amazon(bucket: str, aws_access_key_id: str, aws_secret_access_key: str) -> Storage:
        return AmazonStorage(bucket, aws_access_key_id, aws_secret_access_key)
    