import time

from pinecone import Pinecone, ServerlessSpec
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from sentence_transformers import CrossEncoder  # reranker

from app.core.config import get_settings

settings = get_settings()

_embeddings = None
_vectorstore = None
_EMBEDDING_MODEL_NAME = settings.EMBEDDING_MODEL_NAME

# Step-1 : creating embedding model using HuggingFaceEmbeddings


def get_embedding_model():
    global _embeddings
    _embeddings = HuggingFaceEmbeddings(
        model_name=_EMBEDDING_MODEL_NAME,
        encode_kwargs={"normalize_embeddings": True}
    )
    return _embeddings

# Step-2 : finding the embedding model dimension


def embedding_model_dimension():
    embedding_model = get_embedding_model()
    sample_vector = embedding_model.embed_query("What is AI?")
    dimension = len(sample_vector)
    return dimension

# Step-3 : checking the index in pinecone. if the index is not in pinecone, creating new.


def ensure_index():
    if not settings.PINECONE_API_KEY:
        raise RuntimeError("Pinecone API Key is missing.")

    desired_dimension = embedding_model_dimension()
    pc = Pinecone(api_key=settings.PINECONE_API_KEY)

    index_names = [x["name"] for x in pc.list_indexes()]

    if settings.PINECONE_INDEX_NAME in index_names:
        index_info = pc.describe_index(name=settings.PINECONE_INDEX_NAME)
        current_dimension = getattr(index_info, "dimension", None)
        print(current_dimension)

        if current_dimension is None and isinstance(index_info, dict):
            current_dimension = index_info.get("dimension")
            print(current_dimension)

        if current_dimension is not None and current_dimension != desired_dimension:
            pc.delete_index(name=settings.PINECONE_INDEX_NAME)
            while settings.PINECONE_INDEX_NAME in [x["name"] for x in pc.list_indexes()]:
                time.sleep(1)

    if settings.PINECONE_INDEX_NAME not in [x["name"] for x in pc.list_indexes()]:
        pc.create_index(
            name=settings.PINECONE_INDEX_NAME,
            dimension=desired_dimension,
            metric='cosine',
            spec=ServerlessSpec(
                cloud="AWS",
                region="us-east-1"
            ),
        )
        while not pc.describe_index(settings.PINECONE_INDEX_NAME).status["ready"]:
            time.sleep(1)

    return pc.Index(name=settings.PINECONE_INDEX_NAME)

# Step-4 : making vector store to send document chunks


def get_vectorstore():
    global _vectorstore
    if _vectorstore is None:
        index = ensure_index()
        _vectorstore = PineconeVectorStore(
            index=index,
            embedding=_embeddings,
            namespace=settings.PINECONE_INDEX_NAMESPACE
        )
    return _vectorstore

# Step-5 : adding documents to vector store


def add_documents(chunks):
    vectorstore = get_vectorstore()
    return vectorstore.add_documents(chunks)

# Step-6 : making retriever


def get_retriever():
    return get_vectorstore().as_retriever(
        search_kwargs={"k": settings.TOP_K, 'namespace': settings.PINECONE_INDEX_NAMESPACE})

# Step-7: reranker


def get_reranker(query: str, kb_docs):
    # step-1: making reranker
    reranker= CrossEncoder(model_name_or_path=settings.RERANKING_MODEL_NAME)

    # step-2: create query document pairs
    pairs= [[query,doc.page_content] for doc in kb_docs]

    # Step-3: calculating reranking scores
    scores= reranker.predict(pairs)

    # step-4: attach scores to documents
    
