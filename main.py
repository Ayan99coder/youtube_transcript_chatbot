
from langchain_community.document_loaders import YoutubeLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_core.runnables import (
    RunnableParallel,
    RunnablePassthrough,
    RunnableLambda
)
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

# 1. YouTube video URL
video_url = "https://www.youtube.com/watch?v=J5_-l7WIO_w&list=PLKnIA16_RmvaTbihpo4MtzVm4XOQa0ER0&index=17"

# 2. Embedding model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# 3. Gemini LLM
model = ChatGoogleGenerativeAI(
    model="gemini-3.7-flash"
)

# 4. Load YouTube transcript
loader = YoutubeLoader.from_youtube_url(
    video_url,
    add_video_info=False,
    language=["hi"],
)

docs = loader.load()

# 5. Combine transcript text
transcript = " ".join(
    doc.page_content for doc in docs
)

# 6. Split transcript into chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
)

chunks = splitter.create_documents([transcript])

# 7. Create FAISS vector store
vector_store = FAISS.from_documents(
    documents=chunks,
    embedding=embeddings
)

# 8. Create retriever
retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 4}
)

# 9. Prompt template
prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
        Answer only from the provided transcript context.
        Respond in Roman Urdu.

        If the context is insufficient, say:
        "Mujhe nahi pata".

        Context:
        {context}
        """
    ),
    ("human", "{question}")
])

# 10. Format retrieved documents
def format_docs(docs):
    return "\n\n".join(
        doc.page_content for doc in docs
    )

# 11. Parallel chain
parallel_chain = RunnableParallel({
    "context": retriever | RunnableLambda(format_docs),
    "question": RunnablePassthrough()
})

# 12. Complete RAG chain
result_chain = (
    parallel_chain
    | prompt
    | model
    | StrOutputParser()
)



# 14. Ask questions continuously
while True:
    question = input(
        "\nApna question likho (exit to quit): "
    ).strip()

    if question.lower() == "exit":
        break

    if not question:
        continue

    # Retrieve context, generate answer, parse output
    answer = result_chain.invoke(question)

    print("\nAnswer:", answer)

  
