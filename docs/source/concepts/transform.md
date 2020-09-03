# Data Pipelines

Data pipelines are usually a series of data transformations on datasets. User needs to implement the transformation in the dataset generator form. 

## Transform
Hub Transform are user-defined classes that implement Hub Transform interface. You can think of them as user-defined data transformations that stand as nodes from which the data pipelines are constructed.

Transform interface looks like this.
```python
class Transform:

    def forward(self, input):
        raise NotImplementedError()
​
    def meta(self):
        raise NotImplementedError()
```

then you can apply the function on a list or a `hub.dataset` object. `.generate()` function returns a dataset object. Note that all computations are done in lazy mode, and in order to get the final dataset we need to call the compute method. 

```python
from hub import dataset

ids = [1,2,3] 
croped_images = dataset.generate(Transform(), ids)
croped_images.compute()
```

You can stack multiple transformations together before calling compute function.

```python
from hub import dataset

ids = [1,2,3] 
croped_images = dataset.generate(Transform1(), croped_images)
flipped_images = dataset.generate(Transform2(), ids)
flipped_images.compute()
```

To make it easier to comprehend, let's discuss an example.

## Example

Let's say you have a set of images and want to crop the center and then flip them. You also want to execute this data pipeline in parallel on all samples of your dataset.

1. Implement `Crop(Transform)` class that describes how to crop one image.

   We assume we want to crop 256 * 256 rectangle. Then meta should indicate that in output we are going to have one 2 dimensional array with 256 * 256 shape. The call function should implement the actual crop functionality.

   ```python
   from hub import Transform

   class Crop(Transform):
      def forward(self, input):
         return {"image": input[0:1, :256, :256]}

      def meta(self):
         return {"image": {"shape": (1, 256, 256), "dtype": "uint8"}}
   ```

2. Implement `Flip(Transform)` class that describes how to flip one image.
   ```python
   class Flip(Transform):
      def forward(self, input):
         img = np.expand_dims(input["image"], axis=0)
         img = np.flip(img, axis=(1, 2))
         return {"image": img}

      def meta(self):
         return {"image": {"shape": (1, 256, 256), "dtype": "uint8"}}
   ```
3. Apply those transformations on the dataset. 
   ```python
   from hub import dataset

   images = [np.ones((1, 512, 512), dtype="uint8") for i in range(20)]
   ds = dataset.generate(Crop(), images)
   ds = dataset.generate(Flip(), ds)
   ds.store("/tmp/cropflip")
   ```

Special care need to be taken for `meta` information and output dimensions of each sample in `forward` pass. We are planning to simplify this process. Any recommendation as Git issue would be greatly appreciated. 

## API
```eval_rst
.. autoclass:: hub.Transform
   :members:
```