from fastapi import APIRouter, UploadFile, File
from app.ml.cry_cnn import analyze_audio_bytes
import io

router = APIRouter(prefix="/api/cry", tags=["Cry Analysis"])

@router.post("/analyze")
async def analyze_cry_audio(file: UploadFile = File(...)):
    contents = await file.read()
    audio_stream = io.BytesIO(contents)
    result = analyze_audio_bytes(audio_stream)
    return result 