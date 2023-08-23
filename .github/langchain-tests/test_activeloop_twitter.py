from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import DeepLake
from langchain.document_loaders import TextLoader
from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter
import os
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain


load_dotenv()


def test():
    root_dir = "./the-algorithm"
    docs = []
    idx = 0
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for file in filenames:
            try:
                loader = TextLoader(os.path.join(dirpath, file), encoding="utf-8")
                docs.extend(loader.load_and_split())
                idx += 1
            except Exception as e:
                pass
        if idx > 10:
            break

    embeddings = OpenAIEmbeddings(disallowed_special=())

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(docs)

    username = "testingacc2"  # replace with your username from app.activeloop.ai
    db = DeepLake(
        dataset_path=f"hub://{username}/twitter-algorithm",
        embedding=embeddings,
        overwrite=True,
    )
    db.add_documents(texts)

    db = DeepLake(
        dataset_path=f"hub://{username}/twitter-algorithm",
        read_only=True,
        embedding=embeddings,
    )

    retriever = db.as_retriever()
    retriever.search_kwargs["distance_metric"] = "cos"
    retriever.search_kwargs["fetch_k"] = 100
    retriever.search_kwargs["maximal_marginal_relevance"] = True
    retriever.search_kwargs["k"] = 10

    def filter(x):
        # filter based on source code
        if "com.google" in x["text"].data()["value"]:
            return False

        # filter based on path e.g. extension
        metadata = x["metadata"].data()["value"]
        return "scala" in metadata["source"] or "py" in metadata["source"]

    ### turn on below for custom filtering
    retriever.search_kwargs["filter"] = filter

    model = ChatOpenAI(model_name="gpt-3.5-turbo-0613")  # switch to 'gpt-4'
    qa = ConversationalRetrievalChain.from_llm(model, retriever=retriever)

    questions = [
        "What does favCountParams do?",
    ]
    chat_history = []

    for question in questions:
        result = qa({"question": question, "chat_history": chat_history})
        chat_history.append((question, result["answer"]))
        print(f"-> **Question**: {question} \n")
        print(f"**Answer**: {result['answer']} \n")
