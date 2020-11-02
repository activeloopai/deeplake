import os
from typing import Dict, Tuple
from functools import partial
import multiprocessing

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST
from torchvision import transforms
import pytorch_lightning as pl
from pytorch_lightning.metrics.functional import accuracy
from pytorch_lightning.callbacks import Callback
import ray

from hub.training import logs
from hub.api.dataset import Dataset
from hub.training.model import Model



class LitMNIST(pl.LightningModule):    
    def __init__(self, **kwargs,
    #  logs_dataset: Dataset=None
     ): 
     
        super().__init__()        
        # self.log_tracker = logs.Track(logs=logs_dataset)        
        self.optimizer = kwargs.get('optimizer', torch.optim.Adam)
        self.learning_rate = kwargs.get('lr', 0.01)

        self.hidden_size = 32        
        self.dropout_rate = 0.5

        self.num_classes = 10
        self.dims = (1, 28, 28)
        channels, width, height = self.dims
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels * width * height, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.hidden_size, self.num_classes)
        )

    def forward(self, x):
        x = self.model(x)
        return F.log_softmax(x, dim=1)
   

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y)     

        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc', acc, prog_bar=True)   
        return {'loss': loss, 'train_acc': acc}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y)

        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)        
        return {'val_acc': acc, 'val_loss': loss}

    def validation_epoch_end(self, outputs):   
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_acc = torch.stack([x['val_acc'] for x in outputs]).mean()        
        # self.log_tracker.track(logs.Track().scalar, 'val', {
        #     'loss': avg_loss, 
        #     'acc' : avg_acc
        # })

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        avg_acc = torch.stack([x['train_acc'] for x in outputs]).mean()
        # self.log_tracker.track(logs.Track().scalar, 'train', {
        #     'loss': avg_loss, 
        #     'acc' : avg_acc
        # }).iterate() 

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def configure_optimizers(self):
        optimizer = self.optimizer(self.parameters(), lr=self.learning_rate)
        return optimizer


@ray.remote(num_gpus=1, num_cpus=4, max_calls=1)
def train_model_tune(model: pl.LightningModule, dataloaders: Tuple, num_epochs: int = 3,
                     logs_dir: str = './models/logs', model_output_dir: str = './models/', **kwargs):
    """Train Pytorch Lightning model using Ray

    Arguments:
    model: Pytorch Lightning module which should be trained
    dataloaders: Tuple of data loaders in the following order (train, validation, test).
                 If test data loader is not provided, test step will be skipped.
    num_epochs: Number of epochs to be run
    logs_dir: Directory to which training metrics will be stored
    model_output_dir: Directory to which model state_dict will be saved
    **kwargs: Arbitrary keyword arguments for model initialization. 
    """
    # log_dataset = Dataset(dtype={"train_acc": float, "train_loss": float, "val_acc": float, "val_loss": float},
    #                       shape=(num_epochs,),
    #                       url = logs_dir,
    #                       mode='w')    
    model = model(**kwargs)
    model_obj = Model(model)
    trainer = pl.Trainer(max_epochs=num_epochs, 
                         progress_bar_refresh_rate=20, num_sanity_val_steps=0, 
        )
    trainer.fit(model, dataloaders[0], val_dataloaders=dataloaders[1])

    if len(dataloaders) == 3:
        trainer.test(verbose=False, test_dataloaders=dataloaders[2])
    # model.log_tracker.logs.commit()
    model_obj.store(model_output_dir)


def fit(model: pl.LightningModule, dataloaders: Tuple, num_epochs: int = 3, 
        num_gpus: int = 1, num_cpus: int = multiprocessing.cpu_count()):
    """Fit dataloaders into model using remote Ray function

    Arguments:
     model: Pytorch Lightning module which should be trained
    dataloaders: Tuple of data loaders in the following order (train, validation, test).
                 If test data loader is not provided, test step will be skipped.
    num_epochs: Number of epochs to be run
    num_gpus: Number of gpus to be used during the training. Default: 1
    num_cpus: Number of cpus to be used during the training. Default: number of cores in machine
    """
    ray.init()

    g = train_model_tune.options(num_gpus=num_gpus, num_cpus=num_cpus)
    ray.get(g.remote(model=model, dataloaders=dataloaders))


if __name__ == "__main__":
    transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])

    data_dir='./'
    mnist_full = MNIST(data_dir, train=True, transform=transform)
    mnist_train, mnist_val = random_split(mnist_full, [55000, 5000])
    mnist_test = MNIST(data_dir, train=False, transform=transform)

    train_dataloader = DataLoader(mnist_train, batch_size=32)
    val_dataloader = DataLoader(mnist_val, batch_size=32)
    test_dataloader = DataLoader(mnist_test, batch_size=32)

    fit(LitMNIST, dataloaders=(train_dataloader, val_dataloader, test_dataloader))