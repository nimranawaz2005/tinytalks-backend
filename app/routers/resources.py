from fastapi import APIRouter

router = APIRouter()

BEHAVIORAL_RESOURCES = [
    {
        "id": "res_meltdown_101",
        "title": "Understanding Sensory Meltdowns",
        "language": "en",
        "description": "Recognizing early meltdown signs and de-escalation basics.",
        "file_url": "/static/resources/meltdown-101-en.pdf"
    },
    {
        "id": "res_meltdown_101_ur",
        "title": "محسوسیاتی برہمی کو سمجھنا",
        "language": "ur",
        "description": "Urdu translation of the sensory meltdown guide.",
        "file_url": "/static/resources/meltdown-101-ur.pdf"
    }
]

@router.get("")
async def list_resources(language: str | None = None):
    if language:
        return [r for r in BEHAVIORAL_RESOURCES if r["language"] == language]
    return BEHAVIORAL_RESOURCES  