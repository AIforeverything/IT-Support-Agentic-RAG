from pathlib import Path
from typing import Iterable
from app.core.config import get_settings

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from docx import Document as Docxloader

SUPPORTED = {".pdf", ".txt", ".md", ".docx"}
settings = get_settings()


def load_file(path: Path) -> list[Document]:
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        return PyPDFLoader(str(path)).load()
    if suffix in {".txt", ".md"}:
        return TextLoader(str(path), encoding="utf-8").load()
    if suffix == ".docx":
        doc = Docxloader(str(path))
        text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
        return [Document(page_content=text, metadata={"source": str(path)})]
    raise ValueError(f"Unsupported file type: {suffix}")


def chunk_documents(docs: Iterable[Document]) -> list[Document]:
    # Optimized for all-MiniLM-L6-v2 (256 token limit)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
        add_start_index=True,
        length_function=len,
        separators=[
            "\n\n",
            "\n",
            " ",
            "",
        ],  # Splits naturally at paragraphs, then lines, then words
    )
    return splitter.split_documents(list(docs))
