from pathlib import Path
from app.core.config import get_settings
from app.core.logging import configure_logging
from app.services.ingestion import load_file, chunk_documents
from app.rag.vectorstore import add_documents

logger = configure_logging()

folder = Path(get_settings().SAMPLE_KB_PATH)
# print(folder)
files = [f for f in folder.iterdir() if f.is_file()]
# print(files)

all_docs = []
# loading files
for f in files:
    all_docs.extend(load_file(f))

# chunking files
doc_chunks = chunk_documents(all_docs)
# print(len(doc_chunks))
# print(doc_chunks)

# sending chunks to vector database
index_content = add_documents(doc_chunks)

print(
    f"Indexed {len(files)} files -> {len(doc_chunks)} chunks -> {len(index_content)} Pinecone vectors"
)
