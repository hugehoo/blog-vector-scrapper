from fastapi import FastAPI

from app.config import settings
from app.database.mongodb import MongoDBClient
from app.features import scraping, search

# Validate settings on startup
settings.validate()

# Initialize FastAPI app
app = FastAPI(
    title="Blog Scraper RAG API",
    version="2.0.0",
    description="A blog scraper with RAG (Retrieval Augmented Generation) capabilities",
)


@app.on_event("startup")
async def startup_event():
    """Initialize connections on startup"""
    MongoDBClient.get_database()
    print("✓ MongoDB connection established")
    print(f"✓ Embedding model: {settings.HF_EMBEDDING_MODEL}")
    print(f"✓ Milvus DB: {settings.MILVUS_DB_PATH}")


@app.on_event("shutdown")
async def shutdown_event():
    """Close connections on shutdown"""
    await MongoDBClient.close_connection()
    print("✓ MongoDB connection closed")


@app.get("/")
async def root():
    """API root endpoint"""
    return {
        "message": "Blog Scraper RAG API",
        "version": "2.0.0",
        "docs": "/docs",
    }


# Register routers
app.include_router(scraping.router)
app.include_router(search.router)
