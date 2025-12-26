"""
FastAPI server with OpenAI Whisper-compatible API for Omnilingual-ASR.
"""

import logging

from fastapi import FastAPI, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse, PlainTextResponse

from app import __version__
from app.service import asr_service, lifespan
from app.config import MODEL_NAME
from app.schemas import ModelsResponse, TranscriptionResponse

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Omnilingual-ASR Server",
    description="OpenAI Whisper-compatible API for Omnilingual-ASR",
    version=__version__,
    lifespan=lifespan,
)


@app.get("/health-check")
async def health_check():
    """Health check endpoint."""
    return "ok"


@app.get("/v1/models", response_model=ModelsResponse)
async def get_models():
    """Get model information."""
    return {
        "data": [
            {
                "id": MODEL_NAME,
                "object": "model",
                "created": 0,
                "owned_by": "omnilingual-asr",
            }
        ]
    }


@app.post("/v1/audio/transcriptions")
async def transcribe(
    file: UploadFile,
    model: str = Form(default=MODEL_NAME),
    language: str | None = Form(default=None),
    prompt: str | None = Form(default=None),
    response_format: str = Form(default="json"),
    temperature: float = Form(default=0.0),
    timestamp_granularities: str | None = Form(default=None),
):
    """
    OpenAI Whisper-compatible transcription endpoint.

    Args:
        file: Audio file (wav, mp3, flac, etc.)
        model: Model identifier (informational only)
        language: Language code (ISO 639-1 or Omnilingual-ASR format)
        prompt: Optional prompt (not used)
        response_format: json, verbose_json, text, srt, or vtt
        temperature: Sampling temperature (not used)
        timestamp_granularities: Timestamp detail level (not used)
    """
    # Validate file
    if not file.filename:
        logger.warning("Transcription request rejected: no file provided")
        raise HTTPException(status_code=400, detail="No file provided")

    # Read audio bytes
    audio_bytes = await file.read()

    if len(audio_bytes) == 0:
        logger.warning("Transcription request rejected: empty file")
        raise HTTPException(status_code=400, detail="Empty file provided")

    logger.info(
        f"Transcription request: file={file.filename}, language={language}, format={response_format}"
    )

    # Run transcription
    try:
        text = await asr_service.transcribe(audio_bytes, language=language)
    except Exception:
        logger.exception(f"Transcription failed for {file.filename}")
        raise HTTPException(status_code=500, detail="Transcription failed")

    # Format response
    if response_format == "text":
        return PlainTextResponse(content=text)

    return JSONResponse(content=TranscriptionResponse(text=text).model_dump())
