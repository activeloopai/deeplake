---
seo_title: "Deep Lake RAG | RAG Application using Deeplake Multimodal vector search"
description: "Up to 10x More Efficient Data Retrieval with TQL, learn how to build RAG applications with deeplake."
---

# Using Deep Lake as a Vector Store in LangChain

## How to Use Deep Lake as a Vector Store in LangChain
Deep Lake can be used as a VectorStore in [LangChain](https://github.com/langchain-ai/langchain) for building Apps that require filtering and vector search. In this tutorial we will show how to create a Deep Lake Vector Store in LangChain and use it to build a Q&A App about the [Twitter OSS recommendation algorithm](https://github.com/twitter/the-algorithm). This tutorial requires installation of:

Install the main libraries:

```bash
pip install --upgrade --quiet  langchain-openai langchain-deeplake tiktoken
```
## Downloading and Preprocessing the Data
First, let's import necessary packages and make sure the Activeloop and OpenAI keys are in the environmental variables `ACTIVELOOP_TOKEN`, `OPENAI_API_KEY`.




```python
import os
import getpass
from langchain_openai import OpenAIEmbeddings
from langchain_deeplake.vectorstores import DeeplakeVectorStore
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
```

Next, we set up environmental variables
```python
if "OPENAI_API_KEY" not in os.environ:
    os.environ["OPENAI_API_KEY"] = getpass.getpass("OpenAI API Key:")

if "ACTIVELOOP_TOKEN" not in os.environ:
    os.environ["ACTIVELOOP_TOKEN"] = getpass.getpass("activeloop token:")
```

Next, let's clone the Twitter OSS recommendation algorithm:

```bash
!git clone https://github.com/twitter/the-algorithm
```

Next, let's load all the files from the repo into a list:


```python
repo_path = '/the-algorithm'

docs = []
for dirpath, dirnames, filenames in os.walk(repo_path):
    for file in filenames:
        try:
            loader = TextLoader(os.path.join(dirpath, file), encoding='utf-8')
            docs.extend(loader.load_and_split())
        except Exception as e:
            print(e)
            pass
```

## A note on chunking text files

Text files are typically split into chunks before creating embeddings. In general, more chunks increases the relevancy of data that is fed into the language model, since granular data can be selected with higher precision. However, since an embedding will be created for each chunk, more chunks increase the computational complexity.

```python
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(docs)
```

## Creating the Deep Lake Vector Store

First, we specify a path for storing the Deep Lake dataset containing the embeddings and their metadata.

```python
dataset_path = 'al://<org-id>/twitter_algorithm'
```

Next, we specify an OpenAI algorithm for creating the embeddings, and create the VectorStore. This process creates an embedding for each element in the texts lists and stores it in Deep Lake format at the specified path. 

```python
embeddings = OpenAIEmbeddings()
```


```python
db = DeeplakeVectorStore.from_documents(dataset_path=dataset_path, embedding=embeddings, documents=texts, overwrite=True)
```

The Deep Lake Vector Store has 4 columns including the `texts`, `embeddings`, `ids`, and `metadata`.

```python
ds.dataset.summary()
```

```bash
Dataset length: 31305
Columns:
  documents : text
  embeddings: embedding(1536, clustered)
  ids       : text
  metadata  : dict
```

## Use the Vector Store in a Q&A App

We can now use the VectorStore in Q&A app, where the embeddings will be used to filter relevant documents (`texts`) that are fed into an LLM in order to answer a question.

If we were on another machine, we would load the existing Vector Store without recalculating the embeddings:

```python
db = DeeplakeVectorStore(dataset_path=dataset_path, read_only=True, embedding_function=embeddings)

```

We have to create a `retriever` object and specify the search parameters.

```python
retriever = db.as_retriever()
retriever.search_kwargs['distance_metric'] = 'cos'
retriever.search_kwargs['k'] = 20
```

Finally, let's create an `RetrievalQA` chain in LangChain and run it:

```python
model = ChatOpenAI(model='gpt-3.5-turbo')
qa = RetrievalQA.from_llm(model, retriever=retriever)
```

```python
qa.run('What programming language is most of the SimClusters written in?')
```

This returns:
```
Most of the SimClusters code is written in Scala as indicated by the packages such as `com.twitter.simclustersann.modules`, `com.twitter.simclusters_v2.scio.common`, `com.twitter.simclusters_v2.summingbird.storm`, and references to Scala-based GCP jobs.
```


## Accessing the Low Level Deep Lake API (Advanced)
When using a Deep Lake Vector Store in LangChain, the underlying Vector Store and its low-level Deep Lake dataset can be accessed via:

```python
# LangChain Vector Store
db = DeeplakeVectorStore(dataset_path=dataset_path)

# Deep Lake Dataset object
ds = db.dataset
```

## SelfQueryRetriever with Deep Lake

Deep Lake supports the [SelfQueryRetriever](https://api.python.langchain.com/en/latest/retrievers/langchain.retrievers.self_query.base.SelfQueryRetriever.html) implementation in LangChain, which translates a user prompt into a metadata filters.


>This section of the tutorial requires installation of additional packages:
>   `pip install deeplake lark`

First let's create a Deep Lake Vector Store with relevant data using the documents below.

```python
from langchain_core.documents import Document

docs = [
    Document(
        page_content="A bunch of scientists bring back dinosaurs and mayhem breaks loose",
        metadata={"year": 1993, "rating": 7.7, "genre": "science fiction"},
    ),
    Document(
        page_content="Leo DiCaprio gets lost in a dream within a dream within a dream within a ...",
        metadata={"year": 2010, "director": "Christopher Nolan", "rating": 8.2},
    ),
    Document(
        page_content="A psychologist / detective gets lost in a series of dreams within dreams within dreams and Inception reused the idea",
        metadata={"year": 2006, "director": "Satoshi Kon", "rating": 8.6},
    ),
    Document(
        page_content="A bunch of normal-sized women are supremely wholesome and some men pine after them",
        metadata={"year": 2019, "director": "Greta Gerwig", "rating": 8.3},
    ),
    Document(
        page_content="Toys come alive and have a blast doing so",
        metadata={"year": 1995, "genre": "animated"},
    ),
    Document(
        page_content="Three men walk into the Zone, three men walk out of the Zone",
        metadata={
            "year": 1979,
            "rating": 9.9,
            "director": "Andrei Tarkovsky",
            "genre": "science fiction",
            "rating": 9.9,
        },
    ),
]
```

Since this feature uses Deep Lake's [Tensor Query Language](https://docs.deeplake.ai/latest/advanced/tql/) under the hood, the Vector Store must be stored in or connected to Deep Lake, which requires [registration with Activeloop](https://app.activeloop.ai/levongh/home):

```python
org_id = <YOUR_ORG_ID>
dataset_path = f"al://{org_id}/self_query"

vectorstore = DeeplakeVectorStore.from_documents(
    docs, embeddings, dataset_path = dataset_path, overwrite = True,
)
```

Next, let's instantiate our retriever by providing information about the metadata fields that our documents support and a short description of the document contents.

```python
from langchain.llms import OpenAI
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.base import AttributeInfo

metadata_field_info = [
    AttributeInfo(
        name="genre",
        description="The genre of the movie",
        type="string or list[string]",
    ),
    AttributeInfo(
        name="year",
        description="The year the movie was released",
        type="integer",
    ),
    AttributeInfo(
        name="director",
        description="The name of the movie director",
        type="string",
    ),
    AttributeInfo(
        name="rating", description="A 1-10 rating for the movie", type="float"
    ),
]

document_content_description = "Brief summary of a movie"
llm = OpenAI(temperature=0)

retriever = SelfQueryRetriever.from_llm(
    llm, vectorstore, document_content_description, metadata_field_info, verbose=True
)
```

And now we can try actually using our retriever!

```python
# This example only specifies a relevant query
retriever.get_relevant_documents("What are some movies about dinosaurs")
```

Output:
Output:

```
[Document(metadata={'genre': 'science fiction', 'rating': array(7.7), 'year': array(1993)}, page_content='A bunch of scientists bring back dinosaurs and mayhem breaks loose'),
 Document(metadata={'genre': 'science fiction', 'rating': array(7.7), 'year': array(1993)}, page_content='A bunch of scientists bring back dinosaurs and mayhem breaks loose'),
 Document(metadata={'genre': 'animated', 'year': array(1995)}, page_content='Toys come alive and have a blast doing so'),
 Document(metadata={'genre': 'animated', 'year': array(1995)}, page_content='Toys come alive and have a blast doing so')]
```

Now we can run a query to find movies that are above a certain ranking:

```python
# This example only specifies a filter
retriever.get_relevant_documents("I want to watch a movie rated higher than 8.5")
```

Output:
```
[Document(metadata={'director': 'Satoshi Kon', 'rating': array(8.6), 'year': array(2006)}, page_content='A psychologist / detective gets lost in a series of dreams within dreams within dreams and Inception reused the idea'),
 Document(metadata={'director': 'Andrei Tarkovsky', 'genre': 'science fiction', 'rating': array(9.9), 'year': array(1979)}, page_content='Three men walk into the Zone, three men walk out of the Zone'),
 Document(metadata={'director': 'Satoshi Kon', 'rating': array(8.6), 'year': array(2006)}, page_content='A psychologist / detective gets lost in a series of dreams within dreams within dreams and Inception reused the idea'),
 Document(metadata={'director': 'Andrei Tarkovsky', 'genre': 'science fiction', 'rating': array(9.9), 'year': array(1979)}, page_content='Three men walk into the Zone, three men walk out of the Zone'), Document(metadata={'director': 'Satoshi Kon', 'rating': array(8.6), 'year': array(2006)}, page_content='A psychologist / detective gets lost in a series of dreams within dreams within dreams and Inception reused the idea'), Document(metadata={'director': 'Andrei Tarkovsky', 'genre': 'science fiction', 'rating': array(9.9), 'year': array(1979)}, page_content='Three men walk into the Zone, three men walk out of the Zone')]
```


Congrats! You just used the Deep Lake Vector Store in LangChain to create a Q&A App! 🎉

