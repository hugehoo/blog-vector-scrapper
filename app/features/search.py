import time

from bson import ObjectId
from fastapi import APIRouter

from app.database.mongodb import get_blog_posts_collection
from app.database.milvus import get_milvus_client
from app.features.embedding import get_embedding_service
from app.models import SearchRequest

router = APIRouter(prefix="/search", tags=["search"])


@router.post("")
async def search_posts(request: SearchRequest):
    """
    Search for similar blog posts using semantic search.

    Steps:
    1. Generate embedding for query
    2. Search in Milvus
    3. Fetch full post data from MongoDB
    """
    start_time = time.time()

    try:
        # Step 1: Generate query embedding
        embedding_service = get_embedding_service()
        query_vector = embedding_service.generate_query_embedding(request.query)

        # Step 2: Search in Milvus (fetch extra hits to account for deduplication)
        milvus_client = get_milvus_client()
        candidate_limit = max(request.limit * 5, request.limit)
        raw_results = milvus_client.search(query_vector, limit=candidate_limit)

        # Deduplicate results by document ID
        deduped_results = []
        seen_docs: set[str] = set()
        for result in raw_results:
            doc_id = result["document_id"]
            if doc_id in seen_docs:
                continue
            seen_docs.add(doc_id)
            deduped_results.append(result)
            if len(deduped_results) >= request.limit:
                break

        if not deduped_results:
            return {
                "success": True,
                "query": request.query,
                "results": [],
                "elapsed_time_seconds": round(time.time() - start_time, 2),
            }

        # Step 3: Fetch full post data from MongoDB
        collection = get_blog_posts_collection()
        document_ids = [result["document_id"] for result in deduped_results]

        # Convert string IDs to ObjectId
        object_ids = [ObjectId(doc_id) for doc_id in document_ids]

        # Fetch posts from MongoDB
        cursor = collection.find({"_id": {"$in": object_ids}})
        posts = await cursor.to_list(length=len(object_ids))

        # Create a mapping of document_id to post
        posts_map = {str(post["_id"]): post for post in posts}

        # Combine Milvus results with MongoDB data
        combined_results = []
        for result in deduped_results:
            doc_id = result["document_id"]
            post = posts_map.get(doc_id)

            if post:
                combined_results.append(
                    {
                        "document_id": doc_id,
                        "title": post.get("title"),
                        "url": post.get("url"),
                        "summary": post.get("summary"),
                        "chunk_index": result.get("chunk_index"),
                        "similarity_score": result["similarity_score"],
                        "distance": result["distance"],
                    }
                )

        elapsed_time = time.time() - start_time

        return {
            "success": True,
            "query": request.query,
            "total_results": len(combined_results),
            "results": combined_results,
            "elapsed_time_seconds": round(elapsed_time, 2),
        }

    except Exception as e:
        return {
            "success": False,
            "query": request.query,
            "error": str(e),
            "elapsed_time_seconds": round(time.time() - start_time, 2),
        }


@router.get("/stats")
async def get_milvus_stats():
    """Get Milvus collection statistics"""
    try:
        milvus_client = get_milvus_client()
        stats = milvus_client.get_stats()
        return {"success": True, "stats": stats}
    except Exception as e:
        return {"success": False, "error": str(e)}
