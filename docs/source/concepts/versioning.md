# Dataset Version Control

Hub can also be used to create different versions of datasets in a manner similar to git versioning.
These versions are not full copies but rather they keep track of differences between versions and are thus stored very efficiently. Unlike git, this isn’t a CLI tool but rather a python API.

## How it works
Hub datasets are always stored in a chunk wise manner. This allows us to store and load the data optimally. While versioning, Hub only creates copies of those chunks that are modified after the previous commit, the rest of the chunks are fetched from previous commits.

### What can I do with Hub versioning currently?
* Modify dataset elements across different versions
* Seamlessly switch between versions

### Features coming in the future
* Modify schema across versions (add or remove Tensors)
* Track versions across transforms
* Delete branches
* (More suggestions welcome!)

## Core Concepts

### Saving the state of a dataset
The current state of a dataset can be saved using commit which is very similar to git commit. commit takes a message as a parameter, saves the current state of the dataset and returns the commit id generated. You can use this commit id to come back to this commit at any time. You don’t necessarily need to store this commit id, it can be accessed later using log (similar to git log).
Example:-
```python
ds = hub.Dataset("path/to/dataset")
for i in range(10):
    ds["img", i] = i * np.ones((100, 100, 3))  # assuming the dataset has a tensor called img
a = ds.commit("my first commit") # the commit message is optional, but recommended
```

### Changing branches
Similar to git, you start out on the master branch by default. There are 2 ways to switch to another branch:-
* Using checkout and passing the branch name as the address
* Making changes while not on the head of a branch i.e. in a state similar to the “Detached head" state in git. For instance, if you have made 5 commits on master and go back to the second commit (again using checkout, explained in detail in the next section) and try to make some change there, you will automatically get checked out to a new branch with a name similar to “auto:ec33fced9d75021a32ae28ff”. Alternatively you could also manually checkout to a branch from this state yourself and prevent auto checkout.

```python
ds = hub.Dataset("path/to/dataset")
ds.checkout("alternate", create=True)  # creates a new branch

ds.checkout("master")
for i in range(10):
    ds["img", i] = 2 * i * np.ones((100, 100, 3))
a = ds.commit("first commit")
for i in range(10):
    ds["img", i] = 3 * ds["img", i].compute()
b = ds.commit("second commit")
ds.checkout(a)
ds["img", 6] = 4 * ds["img", i].compute()  # auto checks out to new branch
c = ds.commit("third commit")
```


### Switching between different versions of the dataset
Checkout is used to switch between different versions of the dataset. It takes a commit id or a branch name as a parameter and changes the dataset state to that. It can also be used to create a new branch from the current version by passing the create parameter as True.
When checking out from the head of a branch, the current uncommitted changes (if any) are also brought over to the new branch.
Example:-
```python
ds = hub.Dataset("path/to/dataset")
ds.checkout("alternate", create=True)  # creates a new branch

# checkout to a commit id of the dataset (this is an example id for illustration)
ds.checkout("ec33fced9d75021a32ae28ff")  # commit id can be obtained from log or by storing result of commit/checkout
```

### Looking at the past commits
Using log you can get a log of all commits made in the past that lead up to the current commit similar to git log. The log provides the commit ids and commit messages. This can be useful for figuring out which commit to go back to.

```python
ds = hub.Dataset("path/to/dataset")
ds.log()
```

### Getting a list of all the branches
The branches property of dataset can be used to get a list of all the branches present.

```python
ds = hub.Dataset("path/to/dataset")
print(ds.branches)
```

### Wrapping up a session
Hub uses a memory cache to ensure fast data storage and retrieval. This cache is automatically flushed to the storage whenever it gets full. Every time you are done making changes to the dataset, you should make a call to commit to ensure that the data is properly flushed. In case you don’t want to create a new commit at the moment, save or flush can be used to save the dataset state so you can continue from where you left off next time.

```python
ds = hub.Dataset("path/to/dataset")
for i in range(10):
    ds["img", i] = i * np.ones((100, 100, 3))

ds.commit("modifications made") # creates a new commit

# OR just save without creating a new commit
ds.flush() # same as ds.save()

```

## API
```eval_rst
.. autofunction:: hub.api.dataset.Dataset.commit
.. autofunction:: hub.api.dataset.Dataset.checkout
.. autofunction:: hub.api.dataset.Dataset.log
.. autofunction:: hub.api.dataset.Dataset.branches
.. autofunction:: hub.api.dataset.Dataset.save
.. autofunction:: hub.api.dataset.Dataset.flush
```