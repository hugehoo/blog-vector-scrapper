from typing import Optional, Sequence

from pymilvus import MilvusClient, DataType

from app.config import settings


class MilvusVectorStore:
    """Milvus client for blog post embeddings"""

    def __init__(
        self,
        db_path: Optional[str] = None,
        vector_dim: Optional[int] = None,
    ):
        """
        Initialize Milvus client with Milvus Lite (local file-based storage).

        Args:
            db_path: Path to the Milvus Lite database file
            vector_dim: Dimension of the embedding vectors
        """
        db_path = db_path or settings.MILVUS_DB_PATH
        self.client = MilvusClient(db_path)
        self.collection_name = "blog_posts"

        # Auto-detect vector dimension from embedding service if not provided
        if vector_dim is None:
            from app.features.embedding import get_embedding_service

            embedding_service = get_embedding_service()
            vector_dim = embedding_service.get_dimension()

        self.vector_dim = vector_dim
        self._ensure_collection()

    def _ensure_collection(self):
        """Ensure collection exists with correct schema"""
        if self.client.has_collection(collection_name=self.collection_name):
            schema = self.client.describe_collection(collection_name=self.collection_name)
            raw_fields = schema.get("fields") or schema.get("schema", {}).get("fields", [])
            field_names = {field["name"] for field in raw_fields}
            required_fields = {"id", "vector", "doc_id", "chunk_index"}
            if not required_fields.issubset(field_names):
                raise RuntimeError(
                    "Existing Milvus collection schema is outdated. "
                    "Please drop the 'blog_posts' collection to migrate to paragraph-level embeddings."
                )
            return
        self.create_collection()

    def create_collection(self):
        """Create collection for blog post embeddings if it doesn't exist"""
        if self.client.has_collection(collection_name=self.collection_name):
            print(f"Collection '{self.collection_name}' already exists")
            return

        # Create schema
        schema = MilvusClient.create_schema(
            auto_id=False,
            enable_dynamic_field=False,
        )

        # Add fields
        schema.add_field(
            field_name="id",
            datatype=DataType.VARCHAR,
            is_primary=True,
            max_length=150,
        )
        schema.add_field(
            field_name="doc_id",
            datatype=DataType.VARCHAR,
            is_primary=False,
            max_length=100,
        )
        schema.add_field(
            field_name="chunk_index",
            datatype=DataType.INT64,
            is_primary=False,
        )
        schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=self.vector_dim)

        # Create index for vector field
        index_params = self.client.prepare_index_params()
        index_params.add_index(
            field_name="vector",
            index_type="FLAT",
            metric_type="COSINE",
        )

        # Create collection
        self.client.create_collection(
            collection_name=self.collection_name,
            schema=schema,
            index_params=index_params,
        )
        print(f"Collection '{self.collection_name}' created successfully")

    def insert_document_chunks(self, document_id: str, chunk_vectors: Sequence[list[float]]):
        """Insert paragraph-level embeddings for a document"""
        if not chunk_vectors:
            return

        # Remove existing vectors for this document to avoid duplicates
        try:
            self.delete_by_id(document_id)
        except Exception:
            pass

        data = []
        for index, vector in enumerate(chunk_vectors):
            data.append(
                {
                    "id": f"{document_id}:{index}",
                    "doc_id": document_id,
                    "chunk_index": index,
                    "vector": vector,
                }
            )

        self.client.insert(collection_name=self.collection_name, data=data)

    def search(self, query_vector: list[float], limit: int = 5) -> list[dict]:
        """
        Search for similar embeddings

        Args:
            query_vector: Query embedding vector
            limit: Number of results to return

        Returns:
            List of search results with id and distance
        """
        results = self.client.search(
            collection_name=self.collection_name,
            data=[query_vector],
            limit=limit,
            output_fields=["doc_id", "chunk_index"],
        )

        # Format results
        formatted_results = []
        for hits in results:
            for hit in hits:
                formatted_results.append(
                    {
                        "chunk_id": hit["id"],
                        "document_id": hit["doc_id"],
                        "chunk_index": int(hit["chunk_index"]),
                        "distance": hit["distance"],
                        "similarity_score": round(1 - hit["distance"], 4),
                    }
                )

        return formatted_results

    def get_stats(self) -> dict:
        """Get collection statistics"""
        stats = self.client.get_collection_stats(collection_name=self.collection_name)
        return {"collection_name": self.collection_name, "row_count": stats["row_count"]}

    def delete_by_id(self, document_id: str):
        """Delete embedding by document ID"""
        self.client.delete(collection_name=self.collection_name, filter=f'doc_id == "{document_id}"')

    def drop_collection(self):
        """Drop the entire collection"""
        self.client.drop_collection(collection_name=self.collection_name)
        print(f"Collection '{self.collection_name}' dropped")


# Singleton instance
_milvus_client: Optional[MilvusVectorStore] = None


def get_milvus_client() -> MilvusVectorStore:
    """Get or create singleton Milvus client instance"""
    global _milvus_client
    if _milvus_client is None:
        _milvus_client = MilvusVectorStore()
    return _milvus_client
