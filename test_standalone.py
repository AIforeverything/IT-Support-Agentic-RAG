from app.rag.vectorstore import (
    get_vectorstore,
    get_retriever,
    check_pinecone_stats
)

# 1. Check Pinecone statistics
check_pinecone_stats()

# 2. Check vector store
vectorstore = get_vectorstore()
print("Vector store:", vectorstore)

# 3. Test retrieval without reranking
retriever = get_retriever()

query = "How do I reset my password?"

docs = retriever.invoke(query)

print("Retrieved documents:", len(docs))

for i, doc in enumerate(docs, 1):

    print(f"\nDocument {i}")
    print("Content:", doc.page_content[:500])
    print("Metadata:", doc.metadata)
