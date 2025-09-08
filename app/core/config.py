"""
Application configuration settings
"""

import os
from datetime import datetime
from typing import List

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings"""

    # API Configuration
    HOST: str = Field(default="0.0.0.0", description="API host")
    PORT: int = Field(default=8000, description="API port")
    DEBUG: bool = Field(default=True, description="Debug mode")

    # CORS Configuration
    ALLOWED_ORIGINS: List[str] = Field(
        default=["http://localhost:3000", "http://localhost:8080", "http://127.0.0.1:3000"],
        description="Allowed CORS origins",
    )

    # File Upload Configuration
    MAX_FILE_SIZE: int = Field(
        default=50 * 1024 * 1024, description="Max file size in bytes (50MB)"
    )
    UPLOAD_DIR: str = Field(default="uploads", description="Upload directory")
    ALLOWED_EXTENSIONS: List[str] = Field(
        default=[".pdf", ".docx", ".txt", ".doc"], description="Allowed file extensions"
    )

    # NLP Model Configuration
    SUMMARIZATION_MODEL: str = Field(
        default="facebook/bart-large-cnn", description="Hugging Face model for summarization"
    )
    QA_MODEL: str = Field(
        default="deepset/roberta-base-squad2", description="Hugging Face model for Q&A"
    )
    NER_MODEL: str = Field(
        default="dbmdz/bert-large-cased-finetuned-conll03-english",
        description="Hugging Face model for Named Entity Recognition",
    )
    EMBEDDING_MODEL: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="Sentence transformer model for embeddings",
    )

    # Vector Database Configuration
    VECTOR_DB_PATH: str = Field(default="data/vector_db", description="Vector database path")
    FAISS_INDEX_PATH: str = Field(default="data/faiss_index", description="FAISS index path")
    DUCKDB_PATH: str = Field(default="data/documents.db", description="DuckDB database path")

    # Processing Configuration
    MAX_WORKERS: int = Field(default=4, description="Maximum number of parallel workers")
    CHUNK_SIZE: int = Field(default=512, description="Text chunk size for processing")
    CHUNK_OVERLAP: int = Field(default=50, description="Overlap between text chunks")

    # Logging Configuration
    LOG_LEVEL: str = Field(default="INFO", description="Logging level")
    LOG_FILE: str = Field(default="logs/app.log", description="Log file path")
    LOG_ROTATION: str = Field(default="10 MB", description="Log rotation size")
    LOG_RETENTION: str = Field(default="30 days", description="Log retention period")

    class Config:
        """Pydantic configuration class"""

        env_file = ".env"
        case_sensitive = True

    def get_current_timestamp(self) -> str:
        """Get current timestamp"""
        return datetime.utcnow().isoformat()

    def ensure_directories(self):
        """Ensure required directories exist"""
        directories = [
            self.UPLOAD_DIR,
            os.path.dirname(self.VECTOR_DB_PATH),
            os.path.dirname(self.FAISS_INDEX_PATH),
            os.path.dirname(self.DUCKDB_PATH),
            os.path.dirname(self.LOG_FILE),
        ]

        for directory in directories:
            if directory:
                os.makedirs(directory, exist_ok=True)


settings = Settings()
settings.ensure_directories()
