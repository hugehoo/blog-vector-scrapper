from typing import Iterable, Optional

from sentence_transformers import SentenceTransformer

from app.config import settings


class HuggingFaceEmbedding:
    """Local text embedding service backed by Hugging Face models"""

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        normalize_embeddings: Optional[bool] = None,
    ):
        self.model_name = model_name or settings.HF_EMBEDDING_MODEL
        self.device = device or settings.HF_EMBEDDING_DEVICE
        self.normalize_embeddings = (
            normalize_embeddings
            if normalize_embeddings is not None
            else settings.HF_EMBEDDING_NORMALIZE
        )

        try:
            # SentenceTransformer automatically handles caching on disk for reuse
            self.model = SentenceTransformer(self.model_name, device=self.device)
            self.embedding_dimension = self.model.get_sentence_embedding_dimension()
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load Hugging Face embedding model '{self.model_name}': {exc}"
            )

    def _validate_text(self, text: str) -> str:
        """Validate and clean input text"""
        if not isinstance(text, str):
            raise TypeError("Text to embed must be a string")

        cleaned = text.strip()
        if not cleaned:
            raise ValueError("Text to embed must not be empty")

        return cleaned

    def generate_embedding(self, text: str) -> list[float]:
        """Generate a dense embedding vector for the provided text"""
        return self.generate_embeddings([text])[0]

    def generate_embeddings(self, texts: Iterable[str]) -> list[list[float]]:
        """Generate embeddings for multiple text chunks in a single batch"""
        text_list = [self._validate_text(text) for text in texts if text is not None]
        if not text_list:
            raise ValueError("At least one non-empty text chunk is required")

        try:
            embeddings = self.model.encode(
                text_list,
                convert_to_numpy=True,
                normalize_embeddings=self.normalize_embeddings,
                show_progress_bar=False,
            )
            return [vector.tolist() for vector in embeddings]
        except Exception as exc:
            raise RuntimeError(f"Failed to generate embeddings: {exc}")

    def generate_query_embedding(self, query: str) -> list[float]:
        """Generate embeddings for queries using the same model"""
        return self.generate_embedding(query)

    def get_dimension(self) -> int:
        """Return embedding vector dimension for downstream systems"""
        return self.embedding_dimension


# Singleton instance
_embedding_service: Optional[HuggingFaceEmbedding] = None


def get_embedding_service() -> HuggingFaceEmbedding:
    """Get or create singleton embedding service instance"""
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = HuggingFaceEmbedding()
    return _embedding_service
