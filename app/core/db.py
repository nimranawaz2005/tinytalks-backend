import os
from motor.motor_asyncio import AsyncIOMotorClient
from dotenv import load_dotenv

load_dotenv()

MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
MONGODB_DB_NAME = os.getenv("MONGODB_DB_NAME", "tinytalks")

client: AsyncIOMotorClient | None = None
db = None

async def connect_to_mongo():
    global client, db
    client = AsyncIOMotorClient(MONGODB_URI)
    db = client[MONGODB_DB_NAME]
    await client.admin.command("ping")
    print(f"[db] Connected to MongoDB database '{MONGODB_DB_NAME}'")

async def close_mongo_connection():
    global client
    if client:
        client.close()
        print("[db] MongoDB connection closed")

def get_db():
    return db  