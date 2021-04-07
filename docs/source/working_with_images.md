# Working with Images on Hub
Hey! Welcome to this example notebook to help you understand how to work with Image datasets on Hub. Let's get started with a simple dataset: Dogs vs Cats. This training archive consists of 25,000 images (Split into half for Dogs and Cats). The model's performance is tested is tested on test1.zip (1 = dog, 0 = cat). 
Dataset is avaiable at [Kaggle](https://www.kaggle.com/c/dogs-vs-cats/overview)

If you want to run this code yourself, check out the notebook [here!](https://colab.research.google.com/drive/1hG9sjdgnpqQhXWAFqApXXRx5tqm-UuYY?usp=sharing)

### Imports
First, let's import all the necessary packages and libraries:
```py
import os
import pandas as pd
import numpy as np
from numpy import asarray
from tqdm import tqdm
import hub
from hub.schema import Image, ClassLabel
from PIL import Image
from skimage.transform import resize
from skimage import img_as_ubyte
from hub import Dataset, transform, schema
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
%matplotlib inline
```

### Reading Sample Images
```py
from matplotlib import pyplot
from matplotlib.image import imread

folder = 'train/'
for i in range(9):
	pyplot.subplot(330 + 1 + i)
    # plot dog photos from the dogs vs cats dataset
	filename = folder + 'dog.' + str(i) + '.jpg'
	image = imread(filename)
	pyplot.imshow(image)
pyplot.show()
```
![](/img/working_with_images1.png)


Now, let's just collect all the filenames corresponding to cat images and append them to one common DataFrame.

*Note: There are various ways of uploading datasets to Hub. However, we will use a DataFrame for this example.*

```py
images_df = pd.DataFrame()
root_dir = '/content/train/'
count = 0
for i in range(12500):
  with Image.open(root_dir+'cat.'+str(i) + ' resized' +'.jpg', 'r') as fin:
    images_df = images_df.append({'Image' : asarray(fin) , 'Label': 0}, ignore_index = True)
    count += 1
fin.close()
```
Let's now do the same for the dog images.
```py
for i in range(12500):
  with Image.open(root_dir+'dog.'+str(i) + ' resized' +'.jpg', 'r') as fin:
    images_df = images_df.append({'Image' : asarray(fin) , 'Label': 1}, ignore_index = True)
    count += 1
fin.close()
```


### Uploading the dataset to Hub
```py
# Set the value of url to <your_username>/<dataset_name>
url = "Eshan/dogsvscats"

# Define the schema for our dataset, out dataset will include an image with a corresponding label
my_schema={
        "image": schema.Image(shape=(None,None , 3), max_shape=(150,150,3), dtype="uint8"),
        "label": ClassLabel(num_classes=2)}

ds = hub.Dataset(url, shape=(25000,), schema=my_schema)
for i in tqdm(range(len(ds))):
    ds["image", i] = images_df["Image"][i]
    ds["label", i] = images_df["Label"][i]
# Saving the dataset to the cloud:
ds.flush()
```
Please note that there is no need to run this piece of code multiple times. Once you've run this cell once, you can find your dataset at [https://app.activeloop.ai](https://app.activeloop.ai), and you can call that dataset at anytime, simply by running the line below.
```py
ds = hub.Dataset(url) # Where url is the same as the code above
```
Upload your dataset once, then just keep calling it for your use case! In the cases of many popular datasets such as the one we're using right now, you don't even need to download the dataset yourself. Simply run the code above with the correct `url`, and you're good to go!

### Pre-processing Data using Hub Transform
Our dataset would require pre-processing before we train on it. Let's use Hub's transform method to quickly resize all images to 150x150x3 and store it in a new dataset **resized_dogsvscats**

Schema for new dataset
```py
new_schema = {
    "resized_image": schema.Image(shape=(150, 150, 3), dtype="uint8"),
    "label": ClassLabel(num_classes=2)
}
```

Hub transform method to resize images
```py
@hub.transform(schema=new_schema)
def resize_transform(index):
    image = resize(ds['image', index].compute(), (150, 150, 3), anti_aliasing=True)
    image = img_as_ubyte(image)  # recast from float to uint8
    label = int(ds['label', index].compute())
    return {
        "resized_image": image,
        "label": label
    }
```

Transform object and store in our Resized dataset
```py
ds2 = resize_transform(range(25000))

url = "Eshan/resized_dogsvscats"
ds3 = ds2.store(url)
```

### Hub in Action
Let's try to see Hub in action. We'll train a binary classification model, streaming the data from Hub. 

```py
# Training dataset
train_dataset = np.array([item["resized_image"].compute() for item in ds3])
X = np.reshape(train_dataset, (train_dataset.shape[0], -1))
# Training Labels
y = ds["label"].compute() 
```
```py
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegressionCV
clf = LogisticRegressionCV()

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1, test_size=0.2, shuffle=True)
clf.fit(X_train, y_train)
```
Let's make some predictions based on our model:
```py
classes = {0: 'cats',
           1: 'dogs'}

def show_image_prediction(X, idx, model) :
  image = X[idx].reshape(1,-1)
  image_class = classes[model.predict(image).item()]
  image = image.reshape((150, 150, 3))
  plt.figure(figsize = (4,2))
  plt.imshow(image)
  plt.title("Test {} : Image prediction: {}".format(idx, image_class))
  plt.show()

for i in np.random.randint(0, len(X_test), 3) :
  show_image_prediction(X_test, i, clf)
```

![](/img/working_with_images2.png)

### Conclusion
Hopefully, this example as given you a good idea on the powerful capabilities of Hub and the use cases it has. Again, the notebook is available on Google Colaboratory for your reference [here.](https://colab.research.google.com/drive/1hG9sjdgnpqQhXWAFqApXXRx5tqm-UuYY?usp=sharing)