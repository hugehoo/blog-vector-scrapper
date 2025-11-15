import certifi
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase

from app.config import settings


class MongoDBClient:
    """MongoDB connection management"""

    _client: AsyncIOMotorClient | None = None
    _database: AsyncIOMotorDatabase | None = None

    @classmethod
    def get_client(cls) -> AsyncIOMotorClient:
        """Get MongoDB client singleton instance"""
        if cls._client is None:
            cls._client = AsyncIOMotorClient(
                settings.MONGODB_URI, tlsCAFile=certifi.where()
            )
        return cls._client

    @classmethod
    def get_database(cls) -> AsyncIOMotorDatabase:
        """Get MongoDB database singleton instance"""
        if cls._database is None:
            client = cls.get_client()
            cls._database = client[settings.MONGODB_DB_NAME]
        return cls._database

    @classmethod
    async def close_connection(cls):
        """Close MongoDB connection"""
        if cls._client:
            cls._client.close()
            cls._client = None
            cls._database = None


def get_blog_posts_collection():
    """Get blog_posts collection from MongoDB"""
    db = MongoDBClient.get_database()
    return db.blog_posts
