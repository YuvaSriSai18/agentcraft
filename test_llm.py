from langchain_ollama import ChatOllama


llm = ChatOllama(
    model="qwen2.5-coder:7b",
    temperature=0,
)


response = llm.invoke(
    "Explain Retrieval Augmented Generation in 3 simple sentences."
)


print(response.content)