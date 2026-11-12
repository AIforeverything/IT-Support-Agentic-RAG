from pathlib import Path
from fastapi import APIRouter, UploadFile, File, Header, HTTPException
from pydantic import BaseModel, Field
from app.core.config import get_settings
from app.rag.workflow import ask
from app.rag.vectorstore import add_documents
from app.services.ingestion import load_file, chunk_documents, SUPPORTED
from app.services.audit import write_audit

router = APIRouter(prefix="/api")
settings = get_settings()


class ChatRequest(BaseModel):
    question: str = Field(min_length=2, max_length=1000)


@router.get("/health")
def health():
    return {"status": "ok", "services": settings.APP_NAME}


@router.post("/chat")
def chat(payload: ChatRequest):
    try:
        result = ask(payload.question)
        write_audit(payload.question,
                    result["source_used"], result.get("trace", []))

        return {
            "answer": result["answer"],
            "source_used": result["source_used"],
            "trace": result.get("trace", []),
            "citations": result.get("citations", []),
            "rewritten_query": result.get("current_query", payload.question),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/ingest")
async def ingest(file: UploadFile = File(...), admin_key: str = Header(default="")):
    if admin_key != settings.ADMIN_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid admin key.")
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in SUPPORTED:
        raise HTTPException(
            status_code=400, detail=f"supported {', '.join(sorted(SUPPORTED))}"
        )

    upload_dir = Path(settings.UPLOAD_DIR)
    upload_dir.mkdir(parents=True, exist_ok=True)
    destination = upload_dir / Path(file.filename).name
    destination.write_bytes(await file.read())
    docs = load_file(destination)
    chunks = chunk_documents(docs)
    print("Generated chunks:", len(chunks))
    ids = add_documents(chunks)
    print("Inserted vectors:", len(ids))
    return {
        "message": "Document indexed",
        "file": destination.name,
        "chunks": len(chunks),
        "ids created": len(ids),
    }
