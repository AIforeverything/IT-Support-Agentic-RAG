# Load all `.env` configurations into the application using a centralized settings class .
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path

BASE_PATH = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    APP_NAME: str = "Enterprise IT Support Agentic RAG"
    APP_ENV: str = "Development"
    HUGGINGFACE_API_KEY: str = ""  # not mandatory, because str = ''
    GEMINI_API_KEY: str = ""
    GROQ_API_KEY: str
    GROQ_MODEL: str  # required, otherwise pydantic throws an error
    LANGSMITH_API_KEY: str = ""
    TAVILY_API_KEY: str  # required
    TEMPERATURE: float = 0

    EMBEDDING_MODEL_NAME: str  # required
    RERANKING_MODEL_NAME: str  # required
    PINECONE_API_KEY: str  # required
    PINECONE_INDEX_NAME: str  # required
    PINECONE_INDEX_NAMESPACE: str  # required

    MAX_RETRIES: int = 1
    TOP_K: int = 10
    TOP_N_RANKED: int = 3
    ADMIN_API_KEY: str = "load"
    AUDIT_DB_PATH: str = str(BASE_PATH / "data" / "audit.db")
    UPLOAD_DIR: str = str(BASE_PATH / "uploads")
    SAMPLE_KB_PATH: str = str(BASE_PATH / "data" / "sample_kb")

    model_config = SettingsConfigDict(
        title="Enterprise IT Support Agentic RAG",
        env_file=str(BASE_PATH / ".env"),
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
