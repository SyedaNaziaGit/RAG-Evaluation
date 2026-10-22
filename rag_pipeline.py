from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI,GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate

PDF_PATH = "document/sustainable_development.pdf"
CHROMA_DIR = "data/chroma_db"
EMBED_MODEL = "gemini-embedding-001"
LLM_MODEL = "gemini-2.5-flash"

#building RAG pipeline
def buildRAGChain():
    docs = PyPDFLoader(PDF_PATH).load()
    chunks = RecursiveCharacterTextSplitter(chunk_size = 800, chunk_overlap = 150).split_documents(docs)
    embeddings = GoogleGenerativeAIEmbeddings(model= EMBED_MODEL,task_type="retrieval_document")
    vector_store = Chroma.from_documents(chunks, embeddings ,persist_directory = CHROMA_DIR)
    retriever = vector_store.as_retriever(search_kwargs = {"k":4})
    llm = ChatGoogleGenerativeAI(model = LLM_MODEL,temperature = 0)
    prompt = ChatPromptTemplate.from_template(
        "Use only the context below to answer the question. "
        "If the context does not contain the answer, say you don't know.\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}"
    )
    chain = prompt | llm
    return chain,retriever


    