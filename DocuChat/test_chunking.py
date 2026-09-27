from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


# 1. Load PDF
loader = PyPDFLoader("DocuChat/data/yuvasrisai.pdf")
documents = loader.load()

print(f"Original documents: {len(documents)}")


# 2. Create text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100,
)


# 3. Split documents
chunks = text_splitter.split_documents(documents)

print(f"Total chunks: {len(chunks)}")


# 4. Inspect chunks
for i, chunk in enumerate(chunks):
    print("\n" + "=" * 60)
    print(f"CHUNK {i + 1}")
    print("=" * 60)

    print(chunk.page_content)

    print("\nMetadata:")
    print(chunk.metadata)