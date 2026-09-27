from langchain_community.document_loaders import PyPDFLoader


loader = PyPDFLoader("DocuChat/data/yuvasrisai.pdf")

documents = loader.load()

print(f"Number of pages: {len(documents)}")

for document in documents[:2]:
    print("\n--- PAGE ---")
    print(document.page_content[:500])
    print("\nMetadata:")
    print(document.metadata)