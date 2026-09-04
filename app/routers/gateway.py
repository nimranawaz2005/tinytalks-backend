from datetime import datetime, timezone
from enum import Enum
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
from app.core.db import get_db

router = APIRouter()
CURRENT_CONSENT_POLICY_VERSION = "1.0"

class Language(str, Enum):
    urdu = "ur"
    english = "en"

class LanguageSelection(BaseModel):
    caregiver_id: str
    language: Language

class ConsentPayload(BaseModel):
    caregiver_id: str
    guardian_full_name: str = Field(..., min_length=2)
    accepted_terms: bool
    accepted_biometric_clause: bool = Field(...)

@router.post("/language")
async def set_language(payload: LanguageSelection, db=Depends(get_db)):
    await db.caregivers.update_one(
        {"_id": payload.caregiver_id},
        {"$set": {"language": payload.language.value, "language_set_at": datetime.now(timezone.utc)}},
        upsert=True
    )
    return {"caregiver_id": payload.caregiver_id, "language": payload.language, "saved": True}

@router.post("/consent")
async def submit_consent(payload: ConsentPayload, db=Depends(get_db)):
    if not (payload.accepted_terms and payload.accepted_biometric_clause):
        raise HTTPException(
            status_code=400,
            detail="Terms and Privacy clause must both be accepted."
        )
    consent_record = {
        "caregiver_id": payload.caregiver_id,
        "guardian_full_name": payload.guardian_full_name,
        "policy_version": CURRENT_CONSENT_POLICY_VERSION,
        "accepted_terms": True,
        "accepted_biometric_clause": True,
        "consented_at": datetime.now(timezone.utc)
    }
    await db.caregivers.update_one(
        {"_id": payload.caregiver_id},
        {"$set": {"consent": consent_record}},
        upsert=True
    )
    return {"status": "consent_recorded", "policy_version": CURRENT_CONSENT_POLICY_VERSION}

@router.get("/consent/{caregiver_id}")
async def get_consent_status(caregiver_id: str, db=Depends(get_db)):
    caregiver = await db.caregivers.find_one({"_id": caregiver_id})
    consented = bool(
        caregiver
        and caregiver.get("consent", {}).get("accepted_terms")
        and caregiver.get("consent", {}).get("accepted_biometric_clause")
        and caregiver.get("consent", {}).get("policy_version") == CURRENT_CONSENT_POLICY_VERSION
    )
    return {"caregiver_id": caregiver_id, "consented": consented} 