import hub
import tensorflow
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_spec

class TensorflowDataset(tensorflow.data.Dataset):
    def __init__(self):
        pass

    def _inputs(self):
        pass

    @property
    def element_spec(self):
        pass
    
    def __iter__(self):
        raise NotImplementedError()

    def __len__(self):
        raise NotImplementedError()

    def apply(self, transformation_func):
        return ApplyTensorflowDataset(self, transformation_func)

    def batch(self, batch_size, drop_remainder=False):
        return BatchTensorflowDataset(self, batch_size, drop_remainder=drop_remainder)

    def cache(self, filename=''):
        pass

    def concatenate(self, dataset):
        pass

    def enumerate(self, start = 0):
        i = 0
        for item in self:
            if i >= start:
                yield (i, item)
            i += 1

    def filter(self, predicate):
        pass
    
    def flat_map(self, map_func):
        pass

    # def interleave(self, map_func, cycle_length=tensorflow.data.AUTOTUNE, block_length=1, num_parallel_calls=None):
    #     pass

    def map(self, map_func, num_parallel_calls=None):
        pass
    
    def options(self):
        pass

    def padded_batch(self, batch_size, padded_shapes, padding_values=None, drop_remainder=False):
        pass

    def prefetch(self, buffer_size):
        pass

    def reduce(self, initial_state, reduce_func):
        pass

    def repeat(self, count=None):
        pass

    def shard(self, num_shards, index):
        pass

    def shuffle(self, buffer_size, seed=None, reshuffle_each_iteration=None):
        pass

    def skip(self, count):
        pass

    def take(self, count):
        pass

    def window(self, size, shift=None, stride=1,drop_remainder=False):
        pass

    def with_options(options):
        pass

class HubTensorflowDataset(TensorflowDataset):
    def __init__(self, hub_dataset):
        self.__hub_dataset = hub_dataset

    def __len__(self):
        return self.__hub_dataset.shape[0]

    def __iter__(self):
        # if self.__dataset is None:
        #     self.__dataset = hub.Dataset(key = self._hub_dataset.key)
        
        for i in self.__hub_dataset:
            print(i)
            yield (*list(i),)

class ApplyTensorflowDataset(TensorflowDataset):
    def __init__(self, wrappee, transformation_func):
        self.__wrappee = wrappee
        self.__transformation_func = transformation_func
    
    def __iter__(self):
        pass
    

class BatchTensorflowDataset(TensorflowDataset):
    def __init__(self, wrappee, batch_size, drop_remainder=False):
        self.__wrappee = wrappee
        self.__batch_size = batch_size
        self.__drop_remainder = drop_remainder
    
    def __iter__(self):
        res = None
        cnt = 0
        for item in self.__wrappee:
            if res is None:
                res = [[]] * len(item)
            
            if cnt > 0 and cnt % self.__batch_size == 0:
                yield (*res,)
                res = [[]] * len(res)
            
            cnt += 1
            for j in range(0, len(item)):
                res[j].append(item[j])
        
        if not self.__drop_remainder and res != None and len(res[0]) > 0:
            yield (*res,)

    def __len__(self):
        ans = len(self.__wrappee) / self.__batch_size
        if not self.__drop_remainder and len(self.__wrappee) % self.__batch_size > 0:
            return ans + 1
        else:
            return ans
    