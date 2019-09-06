import hub
import torch
import torch.utils.data as data
    
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

        if self.batch is None or self.indexbeta == self.batch_size:
            self.indexbeta = 0
            self.batch = self.dataset[:self.batch_size]

        output = list(map(lambda x:x[self.indexbeta], self.batch))
        self.indexbeta += 1
        
        return (*output, )

    def __len__(self):
        return self.size