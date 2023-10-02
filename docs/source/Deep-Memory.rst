.. _tql:

Deep Memory
=====================

.. role:: sql(code)
    :language: sql

This page describes  :meth:`ds.query <deeplake.core.vectostore.deep_memory>`. To use deep_memory, please subscribe to our wait list.

Syntax
~~~~~~~~

Training
------

To start training you should first create a vectostore object, and then preprocess the data and use deep memory with it:

.. code-block:: vectorstore initialization
    from deeplake import VectorStore
   
    
    db = VectorStore(
        path="hub://{$ORG_ID}/{$DATASET_ID}",
        token=token, # or you can be signed in with CLI
        runtime={"tensor_db": True},
        embedding_function=embedding_function, # function that takes converts texts into embeddings, it is optional and can be provided later
    )

To train a deepmemory model you need to preprocess the dataset so that, ``corpus``, will become a list of list of tuples, where outer 
list corresponds to the query and inner list to the relevent documents. Each tuple should contain the document id (``id`` tensor from the corpus dataset) 
and the relevence score (range is 0-1, where 0 represents unrelated document and 1 related). ``queries`` should be a list of strings.

.. code-block:: vectorstore preprocessing
    job_id = db.train(
        corpus: List[List[Tuple[str, float]]] = corpus,
        queries: List[str] = queries,
        embedding_function = embedding_function, # function that takes converts texts into embeddings, it is optional and can be skipped if provided during initialization
    )

``job_id`` is string, which can be used to track the training progress. You can use ``db.deep_memory.status(job_id)`` to get the status of the job. 
If you want to get all training jobs you can use ``db.deep_memory.list_jobs()`` which will show all jobs that happened on this dataset.

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

>>> db.deep_memory.list_jobs()
ID                        DATASET ID                                    ORGANIZATION ID  STATUS     RESULTS            PROGRESS       
65198efcd28df3238c49a849  testingacc2/deepmemory_test_corpus_managed_2  testingacc2      completed  Congratulations!   eta: 2.5 seconds
                                                                                                    Your model has     dataset: query 
                                                                                                    achieved a         recall@10: 0.62% (+0.62%)
                                                                                                    recall@10 of                      
                                                                                                    0.62% which is                    
                                                                                                    an improvement                    
                                                                                                    of 0.62% on                       
                                                                                                    the validation                    
                                                                                                    set compared                      
                                                                                                    to naive                          
                                                                                                    vector search.                    
                                                                                                                                      
651a4d41d05a21a5a6a15f67  testingacc2/deepmemory_test_corpus_managed_2  testingacc2      completed  Congratulations!   eta: 2.5 seconds
                                                                                                    Your model has     dataset: query 
                                                                                                    achieved a         recall@10: 0.62% (+0.62%)
                                                                                                    recall@10 of                      
                                                                                                    0.62% which is                    
                                                                                                    an improvement                    
                                                                                                    of 0.62% on                       
                                                                                                    the validation                    
                                                                                                    set compared                      
                                                                                                    to naive                          
                                                                                                    vector search.    


Once the training is completed, you can use ``db.deep_memory.evaluate`` to evaluate the model performance on the custom dataset.
Once again you would need to preprocess the dataset so that, ``corpus``, will become a list of list of tuples, where outer 
list corresponds to the query and inner list to the relevent documents. Each tuple should contain the document id (``id`` tensor from the corpus dataset) 
and the relevence score (range is 0-1, where 0 represents unrelated document and 1 related). ``queries`` should be a list of strings.

.. code-block:: deep_memory evaluation
    recalls = db.deep_memory.evaluate(
        corpus: List[List[Tuple[str, float]]] = corpus,
        queries: List[str] = queries,
        embedding_function = embedding_function, # function that takes converts texts into embeddings, it is optional and can be skipped if provided during initialization
    )

``recalls`` is a dictionary with the following keys:
``with_model`` contains a dictionary with recall metrics for the naive vector search on the custom dataset for different k values
``without_model`` contains a dictionary with recall metrics for the naive vector search on the custom dataset for different k values

After the model is trained you also can search using it:

.. code-block:: deep_memory search:
    results = db.search(
        embedding_data: List[str] = queries,
        embedding_function = embedding_function, # function that takes converts texts into embeddings, it is optional and can be skipped if provided during initialization
        k = 4, # number of results to return
    )
