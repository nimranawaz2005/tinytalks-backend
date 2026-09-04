from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routers import gateway, dashboard, cry_analysis, resources
from app.core.db import connect_to_mongo, close_mongo_connection

app = FastAPI(title="TinyTalks API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_event_handler("startup", connect_to_mongo)
app.add_event_handler("shutdown", close_mongo_connection)

app.include_router(gateway.router, prefix="/api/gateway", tags=["Module 1 - Gateway"])
app.include_router(dashboard.router, prefix="/api/dashboard", tags=["Module 2 - Dashboard"])
app.include_router(cry_analysis.router, prefix="/api/cry-analysis", tags=["Module 2 - Cry Analysis"])
app.include_router(resources.router, prefix="/api/resources", tags=["Module 2 - Resources"])

@app.get("/api/health")
async def health_check():
    return {"status": "ok", "service": "tinytalks-backend"}  