"""
Application settings, loaded from environment variables / .env file.
"""
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # --- App ---
    APP_NAME: str = "RAG E-Commerce Support Assistant"
    ENV: str = "development"

    # --- CORS ---
    FRONTEND_ORIGIN: str = "http://localhost:8501"

    # --- Vector store ---
    VECTOR_STORE_DIR: str = "data/vector_store"
    COLLECTION_NAME: str = "ecommerce_support"
    EMBEDDING_MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"

    # --- Retrieval ---
    TOP_K: int = 4

    # --- LLM (Ollama) ---
    OLLAMA_MODEL: str = "llama3.2"
    OLLAMA_HOST: str = "http://localhost:11434"

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()
