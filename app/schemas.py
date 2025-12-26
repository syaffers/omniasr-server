"""
OpenAI Whisper-compatible response schemas for Omnilingual-ASR.
"""

from pydantic import BaseModel, Field


class TranscriptionResponse(BaseModel):
    """Standard transcription response (response_format=json).

    See: https://platform.openai.com/docs/api-reference/audio/json-object
    """

    text: str = Field(..., description="The transcribed text")


class ErrorResponse(BaseModel):
    """Error response format matching OpenAI's error schema."""

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


class ModelsResponse(BaseModel):
    """Models response format matching OpenAI's models schema.

    See: https://platform.openai.com/docs/api-reference/models/list
    """

    class ModelInfo(BaseModel):
        id: str = Field(..., description="The model identifier")
        object: str = Field(..., description="The object type")
        created: int = Field(0, description="The creation timestamp")
        owned_by: str = Field(..., description="The owner of the model")

    data: list[ModelInfo] = Field(..., description="List of model information")
