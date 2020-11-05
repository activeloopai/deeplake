import os
from typing import Dict, Tuple
import multiprocessing

from hub.training import logs
from hub.api.dataset import Dataset

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST
from torchvision import transforms
import pytorch_lightning as pl
from pytorch_lightning.metrics.functional import accuracy
from pytorch_lightning.callbacks import Callback


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
