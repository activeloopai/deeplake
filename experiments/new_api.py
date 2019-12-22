class Hub:
    def connect(self, backend: str, creds: str = None) -> Bucket:
        # *.connect('s3:bucketname')
        pass

class Bucket:
    def array(self, name: str, shape: tuple, chunk: tuple, dtype: str, order: str = 'C') -> Array:
        pass

class TransferProgress:
    @property 
    def progress(self) -> float:
        pass

class Array:
    def copy_to_bucket(self, bucket: Bucket) -> TransferProgress:
        pass