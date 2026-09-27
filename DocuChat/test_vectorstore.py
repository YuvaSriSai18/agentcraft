from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma


# 1. Load PDF
loader = PyPDFLoader("DocuChat/data/yuvasrisai.pdf")
documents = loader.load()

print(f"Loaded {len(documents)} document(s)")


# 2. Split into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100,
)

chunks = text_splitter.split_documents(documents)

print(f"Created {len(chunks)} chunks")


# 3. Create embedding model
embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)


# 4. Create vector database
vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    collection_name="resume_collection",
    persist_directory="DocuChat/chroma_db",
)

print("Vector store created successfully!")