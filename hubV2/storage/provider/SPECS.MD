#Storage Providers

Each storage provider must add these API/methods:



```python


class StorageProvider():
    def __init__(
        self,
        ...
    ):
        
    def __setitem__(self, path, content):
        
    def __getitem__(self, path):
        
    def __delitem__(self, path):

    def __iter__(self):

    def __len__(self):
```

