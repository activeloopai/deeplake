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
    def __init__(self, dataset, transforms=None):
        self.size = dataset.shape[0]
        self.path = dataset._path
        self.storage = dataset._storage 
        self.dataset = None
        self.transform = transforms
        
    def __iter__(self):
        if self.dataset is None:
            self.dataset = hub.Dataset(self.path, self.storage)
            
        for i in self.dataset:
            i = self.transform(i)
            yield (*list(i),)