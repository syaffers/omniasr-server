"""
FastAPI server with OpenAI Whisper-compatible API for Omnilingual-ASR.
"""

import logging

from fastapi import FastAPI, Form, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, PlainTextResponse

from app import __version__
from app.config import MODEL_NAME
from app.exceptions import APIError
from app.schemas import ErrorResponse, ModelsResponse, TranscriptionResponse
from app.service import asr_service, lifespan

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Omnilingual-ASR Server",
    description="OpenAI Whisper-compatible API for Omnilingual-ASR",
    version=__version__,
    lifespan=lifespan,
)

## Exception handlers


@app.exception_handler(APIError)
async def api_error_handler(request: Request, exc: APIError) -> JSONResponse:
    """Handle APIError exceptions with OpenAI-compatible error responses."""
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=ErrorResponse.ErrorInfo(
                message=exc.message,
                type=exc.error_type,
                param=exc.param,
                code=exc.code,
            )
        ).model_dump(),
    )


@app.exception_handler(RequestValidationError)
async def validation_error_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """Handle validation errors with OpenAI-compatible error responses."""
    errors = exc.errors()
    if errors:
        first_error = errors[0]
        loc = first_error.get("loc", ())
        param = ".".join(str(x) for x in loc) if loc else None
        message = first_error.get("msg", "Validation error")
    else:
        param = None
        message = "Validation error"

    return JSONResponse(
        status_code=400,
        content=ErrorResponse(
            error=ErrorResponse.ErrorInfo(
                message=message,
                type="invalid_request_error",
                param=param,
            )
        ).model_dump(),
    )


## Routes


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
        raise APIError(
            status_code=400,
            message="No file provided",
            param="file",
        )

    # Read audio bytes
    audio_bytes = await file.read()

    if len(audio_bytes) == 0:
        logger.warning("Transcription request rejected: empty file")
        raise APIError(
            status_code=400,
            message="Empty file provided",
            param="file",
        )

    logger.info(
        f"Transcription request: file={file.filename}, language={language}, "
        f"format={response_format}"
    )

    # Run transcription
    try:
        text = await asr_service.transcribe(audio_bytes, language=language)
    except RuntimeError as e:
        logger.exception(f"Transcription failed for {file.filename}")
        # Check the full exception chain for audio decoding errors
        cause = e.__cause__
        cause_msg = str(cause).lower() if cause else ""
        if "sndfile" in cause_msg or "decode" in cause_msg:
            raise APIError(
                status_code=400,
                message=(
                    f"Could not decode audio file '{file.filename}'. "
                    "The file may be corrupted or in an unsupported format."
                ),
                param="file",
                code="invalid_audio_format",
            )
        if "max audio length" in cause_msg:
            raise APIError(
                status_code=400,
                message=(
                    f"Audio file '{file.filename}' is too long. "
                    "The maximum audio length is 40 seconds."
                ),
                param="file",
                code="invalid_audio_length",
            )
        raise APIError(
            status_code=500,
            message=f"Transcription failed: {e}. Cause: {cause_msg or 'unknown'}",
            error_type="server_error",
        )
    except Exception as e:
        logger.exception(f"Transcription failed for {file.filename}")
        raise APIError(
            status_code=500,
            message=f"Transcription failed: {e}",
            error_type="server_error",
        )

    # Format response
    if response_format == "text":
        return PlainTextResponse(content=text)

    return JSONResponse(content=TranscriptionResponse(text=text).model_dump())
