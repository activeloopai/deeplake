import os
from langchain.document_loaders import TextLoader
from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import DeepLake
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain


load_dotenv()


def test():
    root_dir = "libs"

    docs = []
    idx = 0
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for file in filenames:
            if file.endswith(".py") and "*venv/" not in dirpath:
                try:
                    loader = TextLoader(os.path.join(dirpath, file), encoding="utf-8")
                    docs.extend(loader.load_and_split())
                    idx += 1
                except Exception as e:
                    pass

            if idx == 10:
                break

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(docs)
    embeddings = OpenAIEmbeddings()

    db = DeepLake.from_documents(
        texts,
        embeddings,
        dataset_path=f"hub://testingacc2/langchain-code",
        overwrite=True,
    )

    db = DeepLake(
        dataset_path=f"hub://testingacc2/langchain-code",
        read_only=True,
        embedding=embeddings,
    )

    retriever = db.as_retriever()
    retriever.search_kwargs["distance_metric"] = "cos"
    retriever.search_kwargs["fetch_k"] = 20
    retriever.search_kwargs["maximal_marginal_relevance"] = True
    retriever.search_kwargs["k"] = 20

    def filter(x):
        # filter based on source code
        if "something" in x["text"].data()["value"]:
            return False

        # filter based on path e.g. extension
        metadata = x["metadata"].data()["value"]
        return "only_this" in metadata["source"] or "also_that" in metadata["source"]

    model = ChatOpenAI(
        model_name="gpt-3.5-turbo-0613"
    )  # 'ada' 'gpt-3.5-turbo-0613' 'gpt-4',
    qa = ConversationalRetrievalChain.from_llm(model, retriever=retriever)

    questions = [
        "What is the class hierarchy?",
    ]
    chat_history = []
    qa_dict = {}

    for question in questions:
        result = qa({"question": question, "chat_history": chat_history})
        chat_history.append((question, result["answer"]))
        qa_dict[question] = result["answer"]
        print(f"-> **Question**: {question} \n")
        print(f"**Answer**: {result['answer']} \n")
