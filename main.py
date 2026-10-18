from langchain_community.document_loaders import YoutubeLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS 
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_core.runnables import RunnableParallel,RunnablePassthrough,RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
load_dotenv()
video_url = "https://www.youtube.com/watch?v=J5_-l7WIO_w&list=PLKnIA16_RmvaTbihpo4MtzVm4XOQa0ER0&index=17"
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
model = ChatGoogleGenerativeAI(model = 'gemini-3.7-flash')
loader = YoutubeLoader.from_youtube_url(
    video_url,
    add_video_info=False,
    language=["hi"],
)

docs = loader.load()
transcript = " ".join(doc.page_content for doc in docs)
print("Transcript loaded successfully!")
print(transcript)

splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 200,
)
chunk = splitter.create_documents([transcript])

vector_store = FAISS.from_documents(
    documents=chunk,
    embedding=embeddings
)
print(vector_store.index_to_docstore_id)

retreiver = vector_store.as_retriever(search_type = 'similarity',search_kwargs = {"k":4})
question = "iss vedio ka main purpose kya hai?"
text_retriever=retreiver.invoke(question)
prompt = ChatPromptTemplate.from_messages([
    ("system", """
    Answer only from the provided transcript context in Roman Urdu.
    If the context is insufficient, say "Mujhe nahi pata".

    Context:
    {context}
    """),
    ("human", "{question}")
])

# Chat loop
while True:
    question = input("\nApna question likho (exit to quit): ")
    if question.lower().strip() == "exit":
        break
    retrieved_docs = text_retriever.invoke(question)

    context_text = "\n\n".join(
        doc.page_content for doc in retrieved_docs
    )

    final_prompt = prompt.invoke({
        "context": context_text,
        "question": question
    })

    result = model.invoke(final_prompt)

    print("\nAnswer:", result.content)
