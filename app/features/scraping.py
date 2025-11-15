import asyncio
import re
import time

import httpx
from crawl4ai import AsyncWebCrawler
from fastapi import APIRouter, Query

from app.database.mongodb import get_blog_posts_collection
from app.database.milvus import get_milvus_client
from app.features.embedding import get_embedding_service

router = APIRouter(prefix="/posts", tags=["scraping"])


@router.get("/count")
async def get_posts_count():
    """Get total count of blog posts in the database"""
    collection = get_blog_posts_collection()
    total_count = await collection.count_documents({})
    return {"total_count": total_count}


@router.get("")
async def get_posts(
    skip: int = Query(0, ge=0, description="Number of posts to skip"),
    limit: int = Query(50, ge=1, le=500, description="Number of posts to retrieve"),
):
    """Get blog posts with pagination"""
    collection = get_blog_posts_collection()

    # Get total count
    total_count = await collection.count_documents({})

    # Get posts with pagination
    cursor = collection.find({}).skip(skip).limit(limit)
    posts = await cursor.to_list(length=limit)

    # Convert ObjectId to string for JSON serialization
    for post in posts:
        if "_id" in post:
            post["_id"] = str(post["_id"])

    return {
        "total_count": total_count,
        "skip": skip,
        "limit": limit,
        "count": len(posts),
        "posts": posts,
    }


@router.post("/scrape-batch")
async def scrape_batch(
    skip: int = Query(0, ge=0, description="Number of posts to skip"),
    limit: int = Query(50, ge=1, le=500, description="Number of posts to process"),
):
    """
    Fetch a batch of blog posts and send HTTP requests to each URL.
    This is a placeholder for future scraping functionality.
    """
    collection = get_blog_posts_collection()
    start_time = time.time()

    # Get posts with pagination
    cursor = collection.find({}).skip(skip).limit(limit)
    posts = await cursor.to_list(length=limit)

    if not posts:
        return {
            "message": "No posts found",
            "skip": skip,
            "limit": limit,
            "processed": 0,
        }

    # Extract URLs from posts
    urls = []
    for post in posts:
        if "url" in post:
            urls.append(post["url"])

    # Send HTTP requests to each URL concurrently
    success_urls = []
    failed_urls = []

    async with httpx.AsyncClient(timeout=10.0) as client:
        tasks = []
        for url in urls:
            tasks.append(fetch_url(client, url))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        for url, result in zip(urls, results):
            if isinstance(result, Exception):
                failed_urls.append({"url": url, "error": str(result)})
            elif result:
                success_urls.append(url)
            else:
                failed_urls.append({"url": url, "error": "Unknown error"})

    elapsed_time = time.time() - start_time

    return {
        "message": "Batch processing completed",
        "skip": skip,
        "limit": limit,
        "processed": len(posts),
        "total_urls": len(urls),
        "successful": len(success_urls),
        "failed": len(failed_urls),
        "success_urls": success_urls,
        "failed_urls": failed_urls,
        "elapsed_time_seconds": round(elapsed_time, 2),
    }


@router.post("/scrape-and-embed")
async def scrape_and_embed_post(
    skip: int = Query(0, ge=0, description="Index of the post to process"),
):
    """
    Scrape a blog post, generate embedding, and store in Milvus.

    Steps:
    1. Fetch post from MongoDB
    2. Scrape content with crawl4ai
    3. Generate embedding with local model
    4. Store embedding in Milvus
    """
    collection = get_blog_posts_collection()
    start_time = time.time()

    # Get one post from MongoDB
    cursor = collection.find({}).skip(skip).limit(1)
    posts = await cursor.to_list(length=1)

    if not posts:
        return {
            "success": False,
            "message": "No post found at this index",
            "skip": skip,
        }

    post = posts[0]
    document_id = str(post.get("_id"))
    url = post.get("url")

    if not url:
        return {
            "success": False,
            "message": "Post has no URL",
            "document_id": document_id,
        }

    try:
        # Step 1: Scrape content
        scraped_data = await extract_main_content_with_crawl4ai(url)

        if not scraped_data.get("success", False):
            return {
                "success": False,
                "message": f"Scraping failed: {scraped_data.get('error', 'Unknown error')}",
                "document_id": document_id,
                "url": url,
            }

        # Step 2: Clean and chunk content, then generate embeddings
        markdown_content = scraped_data.get("markdown", "")
        if not markdown_content:
            return {
                "success": False,
                "message": "No content to embed",
                "document_id": document_id,
            }

        # Clean text: remove code blocks, HTML tags, and extra whitespace
        cleaned_content = clean_text(markdown_content)
        if not cleaned_content:
            return {
                "success": False,
                "message": "No content after cleaning",
                "document_id": document_id,
            }

        chunks = split_markdown_into_chunks(cleaned_content)
        if not chunks:
            return {
                "success": False,
                "message": "Failed to produce paragraph chunks",
                "document_id": document_id,
            }

        embedding_service = get_embedding_service()
        embedding_vectors = embedding_service.generate_embeddings(chunks)

        # Step 3: Store in Milvus
        milvus_client = get_milvus_client()
        milvus_client.insert_document_chunks(document_id, embedding_vectors)

        elapsed_time = time.time() - start_time

        return {
            "success": True,
            "message": "Post scraped, embedded, and stored successfully",
            "document_id": document_id,
            "url": url,
            "original_title": post.get("title"),
            "content_length": scraped_data.get("full_content_length", 0),
            "embedding_dimension": embedding_service.get_dimension(),
            "chunks_indexed": len(embedding_vectors),
            "elapsed_time_seconds": round(elapsed_time, 2),
        }

    except Exception as e:
        elapsed_time = time.time() - start_time
        return {
            "success": False,
            "message": f"Processing failed: {str(e)}",
            "document_id": document_id,
            "url": url,
            "error": str(e),
            "elapsed_time_seconds": round(elapsed_time, 2),
        }


# Helper functions


async def fetch_url(client: httpx.AsyncClient, url: str) -> bool:
    """
    Fetch a URL and return success status.
    This is a placeholder function that will be expanded for actual scraping.
    """
    try:
        response = await client.get(url)
        return response.status_code == 200
    except Exception as e:
        print(f"Error fetching {url}: {e}")
        raise


async def extract_main_content_with_crawl4ai(url: str) -> dict:
    """
    Extract main content from URL using crawl4ai.
    Returns a dictionary with extracted content.
    """
    try:
        async with AsyncWebCrawler(verbose=False) as crawler:
            result = await crawler.arun(url=url)

            if not result.success:
                return {
                    "url": url,
                    "success": False,
                    "error": "Failed to crawl the page",
                    "markdown": "",
                    "content_length": 0,
                }

            # Get markdown content (crawl4ai automatically extracts clean content)
            markdown_content = result.markdown or ""

            # Split into lines for paragraph counting
            lines = [line.strip() for line in markdown_content.split("\n") if line.strip()]

            return {
                "url": url,
                "success": True,
                "markdown": markdown_content[:5000],  # Preview first 5000 chars
                "full_content_length": len(markdown_content),
                "lines_count": len(lines),
                "preview_lines": lines[:10],  # First 10 lines as preview
                "metadata": {
                    "title": result.metadata.get("title", "") if result.metadata else "",
                    "description": (
                        result.metadata.get("description", "") if result.metadata else ""
                    ),
                },
            }
    except Exception as e:
        return {
            "url": url,
            "success": False,
            "error": str(e),
            "markdown": "",
            "content_length": 0,
        }


def clean_text(text: str) -> str:
    """
    Clean and normalize text by removing code blocks, HTML tags, and extra whitespace.

    Args:
        text: Raw text to clean

    Returns:
        Cleaned text
    """
    # 코드 블록 제거
    text = re.sub(r"```.*?```", "", text, flags=re.DOTALL)
    # HTML 태그 제거
    text = re.sub(r"<.*?>", "", text)
    # 불필요한 공백 제거
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def split_markdown_into_chunks(
    markdown: str,
    max_chars: int = 900,
    min_chars: int = 250,
    overlap_sentences: int = 1,
) -> list[str]:
    """Split markdown into overlapping, sentence-aware chunks"""
    if not markdown:
        return []

    # Normalize spacing and break into paragraphs first to drop navigation noise
    paragraphs = [para.strip() for para in re.split(r"\n\s*\n", markdown) if para.strip()]
    if not paragraphs:
        return []

    # Sentence splitting heuristic that works for mixed English/Korean content
    sentence_splitter = re.compile(r"(?<=[.!?。！？])\s+|\n")
    sentences: list[str] = []
    for paragraph in paragraphs:
        for sentence in sentence_splitter.split(paragraph):
            cleaned = sentence.strip()
            if cleaned:
                sentences.append(cleaned)

    if not sentences:
        return paragraphs

    chunks: list[str] = []
    current_sentences: list[str] = []
    current_len = 0

    for sentence in sentences:
        sent_len = len(sentence)
        projected = current_len + sent_len + (1 if current_sentences else 0)
        if current_sentences and projected > max_chars:
            chunk_text = " ".join(current_sentences).strip()
            if len(chunk_text) < min_chars and chunks:
                # append to previous chunk to avoid tiny fragments
                chunks[-1] = f"{chunks[-1]} \n\n{chunk_text}".strip()
            elif chunk_text:
                chunks.append(chunk_text)

            if overlap_sentences > 0:
                current_sentences = current_sentences[-overlap_sentences:]
                current_len = sum(len(s) for s in current_sentences) + max(
                    len(current_sentences) - 1, 0
                )
            else:
                current_sentences = []
                current_len = 0

        current_sentences.append(sentence)
        current_len += sent_len + (1 if current_sentences else 0)

    if current_sentences:
        chunk_text = " ".join(current_sentences).strip()
        if chunk_text:
            if len(chunk_text) < min_chars and chunks:
                chunks[-1] = f"{chunks[-1]} \n\n{chunk_text}".strip()
            else:
                chunks.append(chunk_text)

    return [chunk for chunk in chunks if chunk]
