from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma


# 1. Load the same embedding model
embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)


# 2. Connect to our existing Chroma database
vector_store = Chroma(
    collection_name="resume_collection",
    persist_directory="DocuChat/chroma_db",
    embedding_function=embeddings,
)


# 3. Create a retriever
retriever = vector_store.as_retriever(
    search_kwargs={"k": 3}
)


# 4. Ask a question
question = input("Ask a question about the resume: ")


# 5. Retrieve relevant chunks
results = retriever.invoke(question)


# 6. Display the results
print(f"\nRetrieved {len(results)} chunks:\n")

for i, document in enumerate(results, start=1):

    print("=" * 70)
    print(f"RESULT {i}")
    print("=" * 70)

    print(document.page_content)

    print("\nMetadata:")
    print(document.metadata)