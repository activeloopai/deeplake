from typing import Dict, List
from hub.core.storage.provider import StorageProvider
from hub.core.meta.meta import Meta
from hub.util.keys import get_dataset_meta_key

class HashlistMeta(Meta):

    def __init__(self):
        
        self.hashlist = []
        
        super().__init__()
    
    def append(self, hash):
        self.hashlist.append(hash)

