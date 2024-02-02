.. _deep_memory:

Deep Memory API
===============
.. currentmodule:: deeplake.core.vectorstore.deep_memory.deep_memory

Creating a Deep Memory
~~~~~~~~~~~~~~~~~~~~~~

If Deep Memory is available on your plan, it will be automatically initialized when you create a Vector Store.

.. autosummary::
    :toctree:
    :nosignatures:

    DeepMemory.__init__

Deep Memory Operations
~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
    :toctree:
    :nosignatures:

    DeepMemory.train
    DeepMemory.cancel
    DeepMemory.delete

Deep Memory Properties
~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
    :toctree:
    :nosignatures:

    DeepMemory.status
    DeepMemory.list_jobs
    

Syntax
~~~~~~
.. role:: sql(code)
    :language: sql

This page describes  :meth:`ds.query <deeplake.core.vectostore.deep_memory>`. DeepMemory is a deep learning model that is trained on the dataset 
to improve the search results, by aligning queries with the corpus dataset. It gives up to +22% of recall improvement on an eval dataset. 
To use deep_memory, please subscribe to our waitlist.

Training
--------

To start training you should first create a vectostore object, and then preprocess the data and use deep memory with it:

>>> from deeplake import VectorStore
>>> db = VectorStore(
...     path="hub://{$ORG_ID}/{$DATASET_ID}",
...     token=token, # or you can be signed in with CLI
...     runtime={"tensor_db": True},
...     embedding_function=embedding_function, # function that takes converts texts into embeddings, it is optional and can be provided later
... )

To train a deepmemory model you need to preprocess the dataset so that, ``corpus``, will become a list of list of tuples, where outer 
list corresponds to the query and inner list to the relevant documents. Each tuple should contain the document id (``id`` tensor from the corpus dataset) 
and the relevance score (range is 0-1, where 0 represents unrelated document and 1 related). ``queries`` should be a list of strings.

>>> job_id = db.deep_memory.train(
...     corpus: List[List[Tuple[str, float]]] = corpus,
...     queries: List[str] = queries,
...     embedding_function = embedding_function, # function that takes converts texts into embeddings, it is optional and can be skipped if provided during initialization
... )

Tracking the training progress
------------------------------
``job_id`` is string, which can be used to track the training progress. You can use ``db.deep_memory.status(job_id)`` to get the status of the job. 

when the model is still in pending state (not started yet) you will see the following:
>>> db.deep_memory.status(job_id)
--------------------------------------------------------------
|                  6508464cd80cab681bfcfff3                  |
--------------------------------------------------------------
| status                     | pending                       |
--------------------------------------------------------------
| progress                   | None                          |
--------------------------------------------------------------
| results                    | not available yet             |
--------------------------------------------------------------

After some time the model will start training and you will see the following:

>>> db.deep_memory.status(job_id)
--------------------------------------------------------------
|                  6508464cd80cab681bfcfff3                  |
--------------------------------------------------------------
| status                     | training                      |
--------------------------------------------------------------
| progress                   | eta: 2.5 seconds              |
|                            | recall@10: 0.62% (+0.62%)     |
--------------------------------------------------------------
| results                    | not available yet             |
--------------------------------------------------------------

If you want to get all training jobs you can use ``db.deep_memory.list_jobs()`` which will show all jobs that happened on this dataset.


>>> db.deep_memory.list_jobs()
ID                          STATUS     RESULTS                      PROGRESS       
65198efcd28df3238c49a849    completed  recall@10: 0.62% (+0.62%)    eta: 2.5 seconds
                                                                    recall@10: 0.62% (+0.62%)                                                                                         
651a4d41d05a21a5a6a15f67    completed  recall@10: 0.62% (+0.62%)    eta: 2.5 seconds
                                                                    recall@10: 0.62% (+0.62%)  

Deep Memory Evaluation
----------------------
Once the training is completed, you can use ``db.deep_memory.evaluate`` to evaluate the model performance on the custom dataset.
Once again you would need to preprocess the dataset so that, ``corpus``, will become a list of list of tuples, where outer 
list corresponds to the query and inner list to the relevant documents. Each tuple should contain the document id (``id`` tensor from the corpus dataset) 
and the relevance score (range is 0-1, where 0 represents unrelated document and 1 related). ``queries`` should be a list of strings.

>>> recalls = db.deep_memory.evaluate(
...     corpus: List[List[Tuple[str, float]]] = corpus,
...     queries: List[str] = queries,
...     embedding_function = embedding_function, # function that takes converts texts into embeddings, it is optional and can be skipped if provided during initialization
...     qvs_params = {"enbabled": True}
... )

``recalls`` is a dictionary with the following keys:
``with_model`` contains a dictionary with recall metrics for the naive vector search on the custom dataset for different k values
``without_model`` contains a dictionary with recall metrics for the naive vector search on the custom dataset for different k values
``qvs_params`` when specified creates a separate vectorstore that tracks all evaluation queries and documents, so that you can use it to compare the performance of 
deep_memory to naive vector search. By default, it is turned off. If enabled the dataset will be created at ``hub://{$ORG_ID}/{$DATASET_ID}_eval_queries``

Deep Memory Search
------------------
After the model is trained you also can search using it:

>>> results = db.search(
...     embedding_data: List[str] = queries,
...     embedding_function = embedding_function, # function that takes converts texts into embeddings, it is optional and can be skipped if provided during initialization
...     k = 4, # number of results to return
...     deep_memory = True, # use deep memory model
... )
