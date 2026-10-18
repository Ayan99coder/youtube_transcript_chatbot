from langchain_community.document_loaders import YoutubeLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS 
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
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

template = PromptTemplate(
    template="""
You are a helpful assistant.
Answere ONLY from the provided transcript context in roman urdu 
if the context is insufficient just say i don't know
{context}
Question : {question}
""",
input_variables=["question","context"]
)
context_text = "\n\n".join(doc.page_content for doc in text_retriever)
final_prompt = template.invoke({"context":context_text,"question":question})
result = model.invoke(final_prompt)
print(result.content)

