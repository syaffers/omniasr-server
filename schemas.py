"""
OpenAI Whisper-compatible response schemas for Omnilingual-ASR.

These schemas match the OpenAI API specification for audio transcription
to ensure compatibility with existing clients and tools.
"""

from pydantic import BaseModel, Field


class TranscriptionResponse(BaseModel):
    """
    Standard transcription response (response_format=json).

    Matches OpenAI's CreateTranscriptionResponseJson schema.
    """

    text: str = Field(..., description="The transcribed text")


class TranscriptionWord(BaseModel):
    """
    Word-level timestamp information.

    Used when timestamp_granularities includes "word".
    """

    word: str = Field(..., description="The word text")
    start: float = Field(..., description="Start time in seconds")
    end: float = Field(..., description="End time in seconds")


class TranscriptionSegment(BaseModel):
    """
    Segment-level transcription information.

    Used in verbose_json response format.
    """

    id: int = Field(..., description="Unique segment identifier")
    seek: int = Field(..., description="Seek offset")
    start: float = Field(..., description="Start time in seconds")
    end: float = Field(..., description="End time in seconds")
    text: str = Field(..., description="Transcribed text for this segment")
    tokens: list[int] = Field(default_factory=list, description="Token IDs")
    temperature: float = Field(default=0.0, description="Sampling temperature")
    avg_logprob: float = Field(default=0.0, description="Average log probability")
    compression_ratio: float = Field(default=1.0, description="Compression ratio")
    no_speech_prob: float = Field(default=0.0, description="Probability of no speech")


class TranscriptionVerboseResponse(BaseModel):
    """
    Verbose transcription response (response_format=verbose_json).

    Matches OpenAI's CreateTranscriptionResponseVerboseJson schema.
    """

    task: str = Field(default="transcribe", description="Task type")
    language: str = Field(..., description="Detected or specified language")
    duration: float = Field(..., description="Audio duration in seconds")
    text: str = Field(..., description="Full transcribed text")
    words: list[TranscriptionWord] = Field(
        default_factory=list, description="Word-level timestamps (if requested)"
    )
    segments: list[TranscriptionSegment] = Field(
        default_factory=list, description="Segment-level transcription data"
    )


class ErrorResponse(BaseModel):
    """
    Error response format matching OpenAI's error schema.
    """

    error: dict = Field(
        ...,
        description="Error details",
        examples=[
            {
                "message": "Invalid file format",
                "type": "invalid_request_error",
                "code": "invalid_file",
            }
        ],
    )


class ModelInfo(BaseModel):
    """
    Model information for /v1/models endpoint.
    """

    id: str = Field(..., description="Model identifier")
    object: str = Field(default="model", description="Object type")
    created: int = Field(default=0, description="Creation timestamp")
    owned_by: str = Field(default="meta", description="Model owner")


class ModelList(BaseModel):
    """
    Response for /v1/models endpoint.
    """

    object: str = Field(default="list", description="Object type")
    data: list[ModelInfo] = Field(..., description="List of available models")


class HealthResponse(BaseModel):
    """
    Health check response.
    """

    status: str = Field(..., description="Health status")
    model: str | None = Field(None, description="Loaded model name")
    device: str | None = Field(None, description="Device being used")
