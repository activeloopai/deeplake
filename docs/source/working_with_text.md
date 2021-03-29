# Working with Text on Hub
Hey! Welcome to this example notebook to help you understand how to work with Text datasets on Hub. Let's get started with a simple dataset: The IMDB Reviews Dataset. This dataset consists of 50,000 reviews (Split into 25000 reviews each for training and testing), each with a particular label: Positive or Negative Review. 

If you want to run this code yourself, check out the notebook [here!](https://colab.research.google.com/drive/1Od6mV6FJXUkq_emsi1rPI4HaPBk7hRzL?usp=sharing#scrollTo=BApS3hkD9hkY)

### Imports
First, let's import all the necessary packages and libraries:
```py
import os
import pandas as pd
from tqdm import tqdm
import hub
from hub.schema import Text, ClassLabel
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfTransformer
```
For reference, you can download this dataset from the link provided [here.](https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz)

### Reading a Sample Review

For now, let's begin with reading one sample review.
```py
filename = 'aclImdb/train/pos/0_9.txt'
with open (filename, 'r') as fin:
    line = fin.readline()
fin.close()
```
```py
line
```
```py
'Bromwell High is a cartoon comedy. It ran at the same time as some other programs about school life, such as "Teachers". My 35 years in the teaching profession lead me to believe that Bromwell High\'s satire is much closer to reality than is "Teachers". The scramble to survive financially, the insightful students who can see right through their pathetic teachers\' pomp, the pettiness of the whole situation, all remind me of the schools I knew and their students. When I saw the episode in which a student repeatedly tried to burn down the school, I immediately recalled ......... at .......... High. A classic line: INSPECTOR: I\'m here to sack one of your teachers. STUDENT: Welcome to Bromwell High. I expect that many adults of my age think that Bromwell High is far fetched. What a pity that it isn\'t!'
```

Now, let's just collect all the filenames corresponding to positive reviews and append them to one common DataFrame.
```py
reviews_df = pd.DataFrame()
root_dir = 'aclImdb/train/pos/'
count = 0
for i in file_names[0]:
    with open(root_dir+i, 'r') as fin:
        reviews_df = reviews_df.append({'Review' : fin.readline(), 'Label': 1}, ignore_index = True)
        count += 1
fin.close()
```
Let's now do the same for the negative reviews.
```py
root_dir = 'aclImdb/train/neg/'
count = 0
for i in file_names[0]:
    with open(root_dir+i, 'r') as fin:
        reviews_df = reviews_df.append({'Review' : fin.readline(), 'Label': 0}, ignore_index = True)
        count += 1
fin.close()
```
Saving this file for later (just in case we need it):
```py
reviews_df.to_csv('IMDBDataset.csv', index=False)
```
Now, let's find the length of the largest review present in the DataFrame.
```py
max_length = 0
for i in reviews_df['Review']:
    if(len(i) > max_length):
        max_length = len(i)
```
After all this pre-processing, now to the good stuff. Let's upload this dataset into Hub!

### Uploading the dataset to Hub
```py
# Set the value of url to <your_username>/<dataset_name>
url = "dhiganthrao/IMDB-MovieReviews"
max_length = 13704 # Obtained from the code above
# We need it to define the shape of the biggest tensor possible in this dataset.
my_schema = {"Review": Text(shape=(None, ), max_shape=(max_length, )),
             "Label": ClassLabel(num_classes=2)}

ds = hub.Dataset(url, shape=(25000,), schema=my_schema)
for i in tqdm(range(len(ds))):
    ds["Review", i] = reviews_df["Review"][i]
    ds["Label", i] = labels[i]
# Saving the dataset to the cloud:
ds.flush()
```
Please note that there is no need to run this piece of code multiple times. Once you've run this cell once, you can find your dataset at [https://app.activeloop.ai](https://app.activeloop.ai), and you can call that dataset at anytime, simply by running the line below.
```py
ds = hub.Dataset(url) # Where url is the same as the code above
```
See? It's that easy! Upload your dataset once, then just keep calling it for your use case! In the cases of many popular datasets such as the one we're using right now, you don't even need to download the dataset yourself. Simply run the code above with the correct `url`, and you're good to go!

Now, let's just verify that `ds` contains the values we want it to have:
```py
print(type(ds))
print(ds.schema)

print(ds["text", 4].compute())
print(ds["label", 4].compute())
```
All is well. Your dataset is safe in the cloud!

### Hub in Action
Let's try to see Hub in action. We'll train a binary classification model, streaming the data from Hub. 
Some basic preprocessing functions are provided below.
```py
import re
def preprocessor(text):
  text =re.sub("<[^>]*>", "", text)
  emoticons = re.findall("(?::|;|=)(?:-)?(?:\)|\(|D|P)", text)
  text = re.sub("[\W]+", " ", text.lower()) + " ".join(emoticons).replace("-", "")
  return text
preprocessor("This is a :) test :-( !")
```
```py
from nltk.stem.porter import PorterStemmer
porter = PorterStemmer()
def tokenizer(text):
  return text.split()
tokenizer("I find it fun to use Hub")
```
```py
def tokenizer_stemmer(text):
  return[porter.stem(word) for word in text.split()]
tokenizer_stemmer("Hub is extremely easy and efficient to use")
```
Now, let's instantiate the model.
```py
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(strip_accents=None,lowercase=True,preprocessor=preprocessor,
                        tokenizer=tokenizer_stemmer,use_idf=True,norm="l2",smooth_idf=True)
X = tfidf.fit_transform([item["Review"].compute() for item in ds]) # Our training dataset
y = ds["Label"].compute() # Training Labels
```
```py
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegressionCV
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1, test_size=0.5, shuffle=True)
clf = LogisticRegressionCV(cv=5,scoring="accuracy",random_state=0,n_jobs=-1,verbose=3,max_iter=300).fit(X_train, y_train)
```
Let's check the accuracy of the model we've just created:
```py
print(f"Accuracy: {clf.score(X_test, y_test)}")
```
Not bad! We've obtained an accuracy of about 88% on the training data! You could do the same yourself; on the test data now, and see the score of this model. 

### Conclusion
Hopefully, this example as given you a good idea on the powerful capabilities of Hub and the use cases it has. Again, the notebook is available on Google Colaboratory for your reference [here.](https://colab.research.google.com/drive/1Od6mV6FJXUkq_emsi1rPI4HaPBk7hRzL?usp=sharing#scrollTo=BApS3hkD9hkY)

