import hub
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
    
class TorchDataset(data.Dataset):
    """ Dataset
    Args:
        dataset (hub.Dataset): hub dataset object 
    """

    def __init__(self, dataset, transforms=None):
        self.size = dataset.shape[0]
        self.key = dataset.key
        self.dataset = None
        self.batch = None
        self.indexbeta = 0
        self.batch_size = dataset.chunk_shape[0]

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (dataset,...) 
        """
        # Load dataset for each worker
        if not self.dataset:
            self.dataset = hub.Dataset(key=self.key)

        # if self.batch is None or self.indexbeta == self.batch_size:
        #     self.indexbeta = 0
        #     self.batch = self.dataset[:self.batch_size]

        # output = list(map(lambda x:x[self.indexbeta], self.batch))
        # self.indexbeta += 1

        output = list(self.dataset[index])
        
        return (*output, )

    def __len__(self):
        return self.size

class TorchIterableDataset(data.IterableDataset):
    def __init__(self, dataset, transform=None):
        self.size = dataset.shape[0]
        self.path = dataset._path
        self.storage = dataset._storage 
        self.dataset = None
        self.transform = transform
        
    def _get_common_chunk(self, dataset):
        batch_sizes = map(lambda x: x[0], dataset.chunks.values())
        return min(batch_sizes)

    def _enumerate(self, dataset):
        """
        Seperate for each worker the id such that there is no conflict
        """
        worker_info = torch.utils.data.get_worker_info()

        if worker_info is None:
            for i, x in enumerate(dataset):
                yield x
                
        if worker_info is None:
            id = 1
            workers = 1 
        else:        
            id = worker_info.id
            workers = worker_info.num_workers
        batch = self._get_common_chunk(dataset)

        # Be careful might contain bugs
        for i in range(dataset.shape[0]):
            if (i//batch)%workers == id:
                yield dataset[i]
        
    def __iter__(self):
        
        if self.dataset is None:
            self.dataset = hub.Dataset(self.path, self.storage)
            
        for x in self._enumerate(self.dataset):
            if self.transform:
                x = self.transform(x)
            yield x
        
            
    def __len__(self):
        return self.size