from datetime import datetime, timezone
from enum import Enum
from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field
from app.core.db import get_db

router = APIRouter()

class LearningTier(str, Enum):
    level_1 = "level_1"
    level_2 = "level_2"

class TierUpdate(BaseModel):
    caregiver_id: str
    tier: LearningTier

class SessionSettings(BaseModel):
    caregiver_id: str
    micro_session_minutes: int = Field(10, ge=2, le=60)
    sos_duration_seconds: int = Field(30, ge=10, le=120)

@router.put("/tier")
async def set_learning_tier(payload: TierUpdate, db=Depends(get_db)):
    await db.caregivers.update_one(
        {"_id": payload.caregiver_id},
        {"$set": {"tier": payload.tier.value, "tier_updated_at": datetime.now(timezone.utc)}},
        upsert=True
    )
    return {"caregiver_id": payload.caregiver_id, "tier": payload.tier}

@router.put("/session-settings")
async def set_session_settings(payload: SessionSettings, db=Depends(get_db)):
    await db.caregivers.update_one(
        {"_id": payload.caregiver_id},
        {"$set": {
            "session_settings": {
                "micro_session_minutes": payload.micro_session_minutes,
                "sos_duration_seconds": payload.sos_duration_seconds,
                "updated_at": datetime.now(timezone.utc)
            }
        }},
        upsert=True
    )
    return payload

@router.get("/settings/{caregiver_id}")
async def get_all_settings(caregiver_id: str, db=Depends(get_db)):
    caregiver = await db.caregivers.find_one({"_id": caregiver_id})
    if not caregiver:
        return {
            "tier": LearningTier.level_1,
            "session_settings": {"micro_session_minutes": 10, "sos_duration_seconds": 30}
        }
    return {
        "tier": caregiver.get("tier", LearningTier.level_1),
        "session_settings": caregiver.get(
            "session_settings",
            {"micro_session_minutes": 10, "sos_duration_seconds": 30}
        )
    }  