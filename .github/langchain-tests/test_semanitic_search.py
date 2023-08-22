text = """

Participants:

Jerry: Loves movies and is a bit of a klutz.
Samantha: Enthusiastic about food and always trying new restaurants.
Barry: A nature lover, but always manages to get lost.
Jerry: Hey, guys! You won't believe what happened to me at the Times Square AMC theater. I tripped over my own feet and spilled popcorn everywhere! 🍿💥

Samantha: LOL, that's so you, Jerry! Was the floor buttery enough for you to ice skate on after that? 😂

Barry: Sounds like a regular Tuesday for you, Jerry. Meanwhile, I tried to find that new hiking trail in Central Park. You know, the one that's supposed to be impossible to get lost on? Well, guess what...

Jerry: You found a hidden treasure?

Barry: No, I got lost. AGAIN. 🧭🙄

Samantha: Barry, you'd get lost in your own backyard! But speaking of treasures, I found this new sushi place in Little Tokyo. "Samantha's Sushi Symphony" it's called. Coincidence? I think not!

Jerry: Maybe they named it after your ability to eat your body weight in sushi. 🍣

Barry: How do you even FIND all these places, Samantha?

Samantha: Simple, I don't rely on Barry's navigation skills. 😉 But seriously, the wasabi there was hotter than Jerry's love for Marvel movies!

Jerry: Hey, nothing wrong with a little superhero action. By the way, did you guys see the new "Captain Crunch: Breakfast Avenger" trailer?

Samantha: Captain Crunch? Are you sure you didn't get that from one of your Saturday morning cereal binges?

Barry: Yeah, and did he defeat his arch-enemy, General Mills? 😆

Jerry: Ha-ha, very funny. Anyway, that sushi place sounds awesome, Samantha. Next time, let's go together, and maybe Barry can guide us... if we want a city-wide tour first.

Barry: As long as we're not hiking, I'll get us there... eventually. 😅

Samantha: It's a date! But Jerry, you're banned from carrying any food items.

Jerry: Deal! Just promise me no wasabi challenges. I don't want to end up like the time I tried Sriracha ice cream.

Barry: Wait, what happened with Sriracha ice cream?

Jerry: Let's just say it was a hot situation. Literally. 🔥

Samantha: 🤣 I still have the video!

Jerry: Samantha, if you value our friendship, that video will never see the light of day.

Samantha: No promises, Jerry. No promises. 🤐😈

Barry: I foresee a fun weekend ahead! 🎉

"""

from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
)
from langchain.vectorstores import DeepLake
from langchain.chains import ConversationalRetrievalChain, RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from dotenv import load_dotenv


load_dotenv()


def test():
    embeddings = OpenAIEmbeddings()

    dataset_path = "hub://testingacc2/data"

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    pages = text_splitter.split_text(dataset_path)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = text_splitter.create_documents(pages)

    embeddings = OpenAIEmbeddings()
    # db = DeepLake.from_documents(
    #     texts, embeddings, dataset_path=dataset_path, overwrite=True
    # )

    db = DeepLake.from_documents(
        texts,
        embeddings,
        dataset_path=dataset_path,
        overwrite=True,
        runtime={"tensor_db": True},
    )

    assert db.vectorstore.exec_option == "tensor_db"

    db = DeepLake(dataset_path=dataset_path, read_only=True, embedding=embeddings)

    retriever = db.as_retriever()
    retriever.search_kwargs["distance_metric"] = "cos"
    retriever.search_kwargs["k"] = 4

    qa = RetrievalQA.from_chain_type(
        llm=OpenAI(),
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=False,
    )

    # What was the restaurant the group was talking about called?
    query = "what did Jerry do?"

    # The Hungry Lobster
    ans = qa({"query": query})
