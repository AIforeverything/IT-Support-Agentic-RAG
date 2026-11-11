from app.rag.vectorstore import (
    get_embedding_model,
    get_vectorstore,
    get_reranker_model,
    get_embedding_model_dimension,
    rerank_documents,
)

# Test 1: Embedding model
emb1 = get_embedding_model()
emb2 = get_embedding_model()

print(emb1 is emb2)  # True


# Test 2: Vector store
vs1 = get_vectorstore()
vs2 = get_vectorstore()

print(vs1 is vs2)  # True


# Test 3: Reranker
r1 = get_reranker_model()
r2 = get_reranker_model()

print(r1 is r2)  # True


# Test 4: Embedding dimension
print(get_embedding_model_dimension())


# Test 5: Retrieve and rerank
query = "what is AI Agent??"

docs = rerank_documents(query, top_n=3)

for i, doc in enumerate(docs, 1):
    print(f"\nDocument {i}:")
    print(doc.page_content)
