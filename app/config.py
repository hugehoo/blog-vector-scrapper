import os
from typing import Optional

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Settings:
    """Application settings loaded from environment variables"""

    # MongoDB settings
    MONGODB_URI: str = os.getenv("MONGODB_URI", "")
    MONGODB_DB_NAME: str = os.getenv("MONGODB_DB_NAME", "test")

    # Hugging Face Embedding settings
    HF_EMBEDDING_MODEL: str = os.getenv(
        "HF_EMBEDDING_MODEL", "BAAI/bge-m3"
    )
    HF_EMBEDDING_DEVICE: str = os.getenv("HF_EMBEDDING_DEVICE", "cpu")
    HF_EMBEDDING_NORMALIZE: bool = os.getenv("HF_EMBEDDING_NORMALIZE", "true").lower() not in {
        "0",
        "false",
        "no",
    }

    # Milvus settings
    MILVUS_DB_PATH: str = os.getenv("MILVUS_DB_PATH", "./milvus_blog.db")

    @classmethod
    def validate(cls) -> None:
        """Validate required settings"""
        if not cls.MONGODB_URI:
            raise ValueError("MONGODB_URI environment variable is required")


# Create singleton instance
settings = Settings()
