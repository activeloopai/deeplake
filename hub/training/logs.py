import importlib
import shutil
import torch
import time
import os
import sys
import hub
# from utils.creds import Creds
import uuid
import json
from hub.api.dataset import Dataset

tensorboardX_spec = importlib.util.find_spec("tensorboardX")
if tensorboardX_spec is not None:    
    from tensorboardX import SummaryWriter

class _SingletonWrapper:
    """
    A singleton wrapper class. Its instances would be created
    for each decorated class. 
    """

    def __init__(self, cls):
        self.__wrapped__ = cls
        self._instance = None

    def __call__(self, *args, **kwargs):
        """Returns a single instance of decorated class"""
        if self._instance is None:
            self._instance = self.__wrapped__(*args, **kwargs)
        return self._instance

def singleton(cls):
    """
    A singleton decorator. Returns a wrapper objects. A call on that object
    returns a single instance object of decorated class. Use the __wrapped__
    attribute to access decorated class directly in unit tests
    """
    return _SingletonWrapper(cls)


@singleton
class Track(object):
    """
    Manage model tracking logic
    """
    # TODO save hyper parameters
   
    scalar = 'scalar'
    counter = 0
    
    def __init__(self, logs: Dataset=None, dir='./data/logs', id=None): 
        super().__init__()
        if id == None:
            id = str(int(time.time()))
        self.logs = logs
        self.writer = None
        if "tensorboardX" in sys.modules:            
            self.path = "{}/{}".format(dir, id)
            self.writer = SummaryWriter(logdir=self.path)
        self.step = 0
        self.timer = {}
        self.meters = {}       


    def add_meter(self, label, scalar, fmt=':f'):
        if label not in self.meters:
            meter = AverageMeter(label, ':6.3f')
            self.meters[label] = meter

        self.meters[label].update(scalar)
        return self.meters[label]
    
    def display(self, iter=None, num_batches=0, cls=None):
        if iter == None:
            iter = self.step
        to_display = []
        for key, value in self.meters.items():
            if cls and cls not in key:
                continue
            to_display.append(value)
        ProgressMeter(num_batches, to_display).display(iter)
        return self
    
    def time(self, label: str, reset: bool = False):
        now = time.time()
        if label not in self.timer:
            self.timer[label] = {
                'diff': 0,
                'last': now,
                'meter': self.add_meter(label, 0, fmt='6.3ft')
            }
        if reset:
            self.timer[label]['diff'] = 0
            self.timer[label]['last'] = now
        else:
            self.timer[label]['diff'] = now - self.timer[label]['last']
            self.timer[label]['last'] = now
            self.timer[label]['meter'].update(self.timer[label]['diff'])
        return self
    
    def add_scalar(self, tag: str, el, frequency_upload_to_s3=1):
        self.counter += 1 
        if isinstance(el,dict):
            if self.writer:
                self.writer.add_scalars(tag, el, global_step=self.step)
            for key, val in el.items():
                self.add_meter("{}_{}".format(tag, key), val)
        else:
            if self.writer:
                self.writer.add_scalar(tag, el, global_step=self.step)
            self.add_meter(tag, el)        
        if self.counter > 0 and self.counter % frequency_upload_to_s3 == 0:
            self.add_to_logs()
    
    def add_to_logs(self):   
        for key, value in self.meters.items():
            self.logs[key][self.step] = value.avg
    
    def track(self, label: str, tag: str, el):  
        if label == self.scalar:
            self.add_scalar(tag, el)
        else:
            #TODO implement other 
            raise NotImplementedError
        return self
    
    def iterate(self):
        self.step += 1
        return self
      
class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'
        