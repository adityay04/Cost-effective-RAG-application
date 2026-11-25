
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    groq_api_key: str
    llm_model: str = "openai/gpt-oss-120b"
    judge_model: str = "openai/gpt-oss-120b"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    chunk_size: int = 200
    chunk_overlap: int = 40
    top_k: int = 5

    chroma_path: str = "./chroma_db"
    documents_path: str = "./data/documents"

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore"
    )


settings = Settings()
