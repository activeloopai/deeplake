# Temporary document, will be added to ReadTheDocs later

In order to add a new storage provider (say XYZ):
- Create a file XYZ.py inside hub/core/storage containing a class XYZProvider
- XYZProvider needs to inherit from hub.core.storage.provider.StorageProvider and needs to implement the following methods:-
    - \_\_init__(self, **args) :- Initialize the object, assign credentials if required.
    - \_\_getitem__(self, path) :- Gets the object present at the path. 
    - \_\_setitem__(self, path, value) :- Sets the object present at the path with the value
    - \_\_delitem__(self, path) :- Delete the object present at the path.
    - \_\_iter__(self):- Generator function that iterates over the keys of the mapper
    - \_\_len__(self) :- Returns the number of files present in the directory at the root of the mapper

