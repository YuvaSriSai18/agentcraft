from langchain_ollama import OllamaEmbeddings


embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)


text = "LangChain helps developers build applications with language models."

vector = embeddings.embed_query(text)


print(f"Vector dimensions: {len(vector)}")
print(f"First 10 values: {vector[:10]}")