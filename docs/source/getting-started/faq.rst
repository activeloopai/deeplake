FAQ
#####################

Q: Will my code stuck on internet download when I run `hub.load` to load the dataset?
A: No. When `hub.load` is a lazy-load meaning it's only downloading the meta data such as shape and data types.
The actual data IO happens when you try to access any slice/chunk of the array.