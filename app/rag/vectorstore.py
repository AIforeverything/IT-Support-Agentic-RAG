import time

from pinecone import Pinecone, ServerlessSpec
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from sentence_transformers import CrossEncoder  # reranker

from app.core.config import get_settings

settings = get_settings()

# global objects are initialized only once and used every time without initializing again and again
_embeddings = None
_vectorstore = None
_reranker = None
_pinecone_client = None
_EMBEDDING_MODEL_NAME = settings.EMBEDDING_MODEL_NAME

# Step-1 : Initializing embedding model using HuggingFaceEmbeddings


def get_embedding_model():
    global _embeddings
    if _embeddings is None:
        print("Loading embedding model...")
        _embeddings = HuggingFaceEmbeddings(
            model_name=_EMBEDDING_MODEL_NAME,
            model_kwargs={"token": settings.HUGGINGFACE_API_KEY},
            encode_kwargs={"normalize_embeddings": True},
        )
    return _embeddings


# Step-2 : finding the embedding model dimension


def get_embedding_model_dimension():
    embedding_model = get_embedding_model()
    sample_vector = embedding_model.embed_query("What is AI in 5 words?")
    return len(sample_vector)


# Step-3: Initialize pinecone client
def get_pinecone_client():
    global _pinecone_client

    if _pinecone_client is None:
        if not settings.PINECONE_API_KEY:
            raise RuntimeError("Pinecone API key is missing.")

        _pinecone_client = Pinecone(api_key=settings.PINECONE_API_KEY)

    return _pinecone_client


# Step-4: checking whether Pinecone Index exists. if the index is not in pinecone, creating new.


def ensure_index():

    pc = get_pinecone_client()
    index_name = settings.PINECONE_INDEX_NAME
    desired_dimension = get_embedding_model_dimension()

    existing_indexes_pinecone = [x["name"] for x in pc.list_indexes()]

    if index_name in existing_indexes_pinecone:
        index_info = pc.describe_index(name=index_name)
        current_dimension = getattr(index_info, "dimension", None)

        if current_dimension != desired_dimension:
            raise ValueError(
                f"Index dimension mismatch"
                f"Existing: {current_dimension}"
                f"Required: {desired_dimension}"
                "create a new index with the same dimension"
                "or migrate the existing vectors"
            )
    else:
        pc.create_index(
            name=settings.PINECONE_INDEX_NAME,
            dimension=desired_dimension,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
    # waiting until index is ready
    while True:
        index_info = pc.describe_index(name=index_name)
        if index_info.status["ready"]:
            break
        time.sleep(1)
    return pc.Index(name=index_name)


# Step-5 : Initializing Pinecone  vector store to send document chunks


def get_vectorstore():
    global _vectorstore

    if _vectorstore is None:
        print("Loading CrossEncoder model...")
        embedding_model = get_embedding_model()
        index = ensure_index()
        _vectorstore = PineconeVectorStore(
            index=index,
            embedding=_embeddings,
        )
    return _vectorstore


# Step-6 : adding documents to vector store


def add_documents(chunks):
    vectorstore = get_vectorstore()
    return vectorstore.add_documents(chunks)


# Step-7 : making retriever


def get_retriever():
    vectorstore = get_vectorstore()
    return vectorstore.as_retriever(
        search_kwargs={
            "k": settings.TOP_K,
            "namespace": settings.PINECONE_INDEX_NAMESPACE,
        }
    )


# Step-8: Initializing CrossEncoder Reranker


def get_reranker_model():
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder(
            model_name_or_path=settings.RERANKING_MODEL_NAME,
            model_kwargs={"token": settings.HUGGINGFACE_API_KEY},
        )
    return _reranker


# Step-8: Retrieve and rerank documents
def rerank_documents(query: str, top_n: int = 3):
    # step-1: retrieve relevant documents using vector store
    retriever = get_retriever()
    kb_docs = retriever.invoke(query)

    if not kb_docs:
        return []

    # Step 2: Remove duplicate documents
    unique_docs = []
    seen = set()

    for doc in kb_docs:
        # Normalize content before checking duplicates
        content = " ".join(doc.page_content.lower().split())

        if content not in seen:
            seen.add(content)
            unique_docs.append(doc)

    # Step 3: Get reranker model
    reranker = get_reranker_model()

    # Step 4: Create query-document pairs
    pairs = [[query, doc.page_content] for doc in unique_docs]

    # Step 5: Calculate relevance scores
    scores = reranker.predict(pairs)

    # Step 6: Combine documents and scores
    scored_docs = list(zip(unique_docs, scores))

    # Step 7: Sort by relevance score
    scored_docs.sort(key=lambda x: x[1], reverse=True)

    # Step 8: Return top N unique documents
    return [doc for doc, score in scored_docs[:top_n]]
