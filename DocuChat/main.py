from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma


# -------------------------
# 1. Embedding model
# -------------------------

embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)


# -------------------------
# 2. Load vector database
# -------------------------

vector_store = Chroma(
    collection_name="resume_collection",
    persist_directory="DocuChat/chroma_db",
    embedding_function=embeddings,
)


# -------------------------
# 3. Create retriever
# -------------------------

retriever = vector_store.as_retriever(
    search_kwargs={"k": 3}
)


# -------------------------
# 4. Load LLM
# -------------------------

llm = ChatOllama(
    model="qwen2.5-coder:7b",
    temperature=0,
)


# -------------------------
# 5. Ask question
# -------------------------

question = input("Ask a question: ")


# -------------------------
# 6. Retrieve documents
# -------------------------

documents = retriever.invoke(question)


# -------------------------
# 7. Build context
# -------------------------

context = "\n\n".join(
    document.page_content
    for document in documents
)


# -------------------------
# 8. Build prompt
# -------------------------

prompt = f"""
You are answering questions about a person's resume.

Use ONLY the information provided in the context.

If the answer cannot be found in the context,
say that you don't know.

Context:
----------------
{context}
----------------

Question:
{question}

Answer:
"""


# -------------------------
# 9. Generate answer
# -------------------------

response = llm.invoke(prompt)


print("\nAnswer:")
print(response.content)


# -------------------------
# 10. Show sources
# -------------------------

print("\nSources:")

for document in documents:
    print(document.metadata)