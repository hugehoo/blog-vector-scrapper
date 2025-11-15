from pydantic import BaseModel, Field


class EmbeddingRequest(BaseModel):
    """Request model for text embedding"""

    text: str = Field(..., description="Text to embed")


class SearchRequest(BaseModel):
    """Request model for semantic search"""

    query: str = Field(..., description="Search query")
    limit: int = Field(default=5, ge=1, le=50, description="Number of results to return")
