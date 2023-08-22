from dotenv import load_dotenv
from langchain.schema import Document
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import DeepLake
from langchain.llms import OpenAI
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.base import AttributeInfo


load_dotenv()


def test():
    embeddings = OpenAIEmbeddings()
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
    username_or_org = "testingacc2"
    vectorstore = DeepLake.from_documents(
        docs,
        embeddings,
        dataset_path=f"hub://{username_or_org}/self_queery",
        overwrite=True,
    )

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
        llm,
        vectorstore,
        document_content_description,
        metadata_field_info,
        verbose=True,
    )

    retriever.get_relevant_documents("What are some movies about dinosaurs")

    retriever.get_relevant_documents("I want to watch a movie rated higher than 8.5")

    # This example specifies a query and a filter
    retriever.get_relevant_documents("Has Greta Gerwig directed any movies about women")

    # This example specifies a composite filter
    retriever.get_relevant_documents(
        "What's a highly rated (above 8.5) science fiction film?"
    )

    # This example specifies a query and composite filter
    retriever.get_relevant_documents(
        "What's a movie after 1990 but before 2005 that's all about toys, and preferably is animated"
    )

    retriever = SelfQueryRetriever.from_llm(
        llm,
        vectorstore,
        document_content_description,
        metadata_field_info,
        enable_limit=True,
        verbose=True,
    )

    # This example only specifies a relevant query
    retriever.get_relevant_documents("what are two movies about dinosaurs")
