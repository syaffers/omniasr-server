"""
Omnilingual-ASR Model Server with OpenAI Whisper-compatible API.

Uses LitServe for high-performance model serving with batching support.
"""

import io
import time
from typing import Optional

import litserve as ls
import soundfile as sf
from fastapi import UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse, PlainTextResponse

from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline

from schemas import (
    TranscriptionResponse,
    TranscriptionVerboseResponse,
    TranscriptionSegment,
)


class OmnilingualASRAPI(ls.LitAPI):
    """
    LitServe API for Omnilingual-ASR model serving.

    Supports multiple model cards:
    - omniASR_CTC_{300M,1B,3B,7B}_v2: Fast parallel CTC generation
    - omniASR_LLM_{300M,1B,3B,7B}_v2: Language-conditioned autoregressive
    - omniASR_LLM_Unlimited_{300M,1B,3B,7B}_v2: Unlimited audio length
    """

    def __init__(self, model_card: str = "omniASR_CTC_1B_v2"):
        """
        Initialize the API with a specific model card.

        Args:
            model_card: The model to load. Defaults to CTC 1B for balanced
                       performance/quality. Use LLM variants for language conditioning.
        """
        self.model_card = model_card
        self._is_llm_model = "LLM" in model_card

    def setup(self, device: str):
        """Load the model on the specified device."""
        print(f"Loading model {self.model_card} on device {device}...")

        # Device mapping for omnilingual pipeline
        # None = auto-detect, otherwise specify cuda:N
        device_arg = None if device == "cuda" else device

        self.pipeline = ASRInferencePipeline(
            model_card=self.model_card,
            device=device_arg,
        )

        print(f"Model {self.model_card} loaded successfully!")

    def decode_request(self, request: dict) -> dict:
        """
        Decode incoming request to internal format.

        Handles the OpenAI-compatible multipart form data.
        """
        return {
            "audio_bytes": request["audio_bytes"],
            "language": request.get("language"),
            "response_format": request.get("response_format", "json"),
            "timestamp_granularities": request.get("timestamp_granularities", []),
        }

    def predict(self, inputs: dict) -> dict:
        """
        Run inference on the audio input.

        Args:
            inputs: Dict containing audio_bytes, language, etc.

        Returns:
            Dict with transcription results.
        """
        audio_bytes = inputs["audio_bytes"]
        language = inputs.get("language")

        start_time = time.perf_counter()

        # Prepare language parameter for LLM models
        lang_param = None
        if language and self._is_llm_model:
            lang_param = [self._map_language_code(language)]

        # Run transcription
        if lang_param:
            transcriptions = self.pipeline.transcribe(
                [audio_bytes],
                lang=lang_param,
                batch_size=1,
            )
        else:
            transcriptions = self.pipeline.transcribe(
                [audio_bytes],
                batch_size=1,
            )

        duration = time.perf_counter() - start_time

        # Get audio duration for response
        audio_duration = self._get_audio_duration(audio_bytes)

        return {
            "text": transcriptions[0] if transcriptions else "",
            "language": language or "unknown",
            "duration": audio_duration,
            "processing_time": duration,
        }

    def encode_response(self, output: dict, request: dict) -> dict:
        """
        Encode the model output to the requested response format.

        Supports OpenAI-compatible response formats:
        - json: Standard JSON response
        - verbose_json: Detailed response with segments
        - text: Plain text response
        - srt/vtt: Subtitle formats (simplified)
        """
        response_format = request.get("response_format", "json")

        return {
            "output": output,
            "response_format": response_format,
        }

    def _map_language_code(self, language: str) -> str:
        """
        Map OpenAI language codes to Omnilingual-ASR language codes.

        OpenAI uses ISO 639-1 (e.g., 'en', 'es', 'fr')
        Omnilingual-ASR uses ISO 639-3 + script (e.g., 'eng_Latn', 'spa_Latn')
        """
        # Common language mappings
        mapping = {
            "en": "eng_Latn",
            "english": "eng_Latn",
            "es": "spa_Latn",
            "spanish": "spa_Latn",
            "fr": "fra_Latn",
            "french": "fra_Latn",
            "de": "deu_Latn",
            "german": "deu_Latn",
            "it": "ita_Latn",
            "italian": "ita_Latn",
            "pt": "por_Latn",
            "portuguese": "por_Latn",
            "ru": "rus_Cyrl",
            "russian": "rus_Cyrl",
            "zh": "zho_Hans",
            "chinese": "zho_Hans",
            "ja": "jpn_Jpan",
            "japanese": "jpn_Jpan",
            "ko": "kor_Hang",
            "korean": "kor_Hang",
            "ar": "arb_Arab",
            "arabic": "arb_Arab",
            "hi": "hin_Deva",
            "hindi": "hin_Deva",
            "nl": "nld_Latn",
            "dutch": "nld_Latn",
            "pl": "pol_Latn",
            "polish": "pol_Latn",
            "tr": "tur_Latn",
            "turkish": "tur_Latn",
            "vi": "vie_Latn",
            "vietnamese": "vie_Latn",
            "th": "tha_Thai",
            "thai": "tha_Thai",
            "id": "ind_Latn",
            "indonesian": "ind_Latn",
            "ms": "zsm_Latn",
            "malay": "zsm_Latn",
            "uk": "ukr_Cyrl",
            "ukrainian": "ukr_Cyrl",
            "cs": "ces_Latn",
            "czech": "ces_Latn",
            "sv": "swe_Latn",
            "swedish": "swe_Latn",
            "da": "dan_Latn",
            "danish": "dan_Latn",
            "fi": "fin_Latn",
            "finnish": "fin_Latn",
            "no": "nob_Latn",
            "norwegian": "nob_Latn",
            "el": "ell_Grek",
            "greek": "ell_Grek",
            "he": "heb_Hebr",
            "hebrew": "heb_Hebr",
            "hu": "hun_Latn",
            "hungarian": "hun_Latn",
            "ro": "ron_Latn",
            "romanian": "ron_Latn",
            "bg": "bul_Cyrl",
            "bulgarian": "bul_Cyrl",
            "sk": "slk_Latn",
            "slovak": "slk_Latn",
            "hr": "hrv_Latn",
            "croatian": "hrv_Latn",
            "sl": "slv_Latn",
            "slovenian": "slv_Latn",
            "sr": "srp_Cyrl",
            "serbian": "srp_Cyrl",
            "et": "est_Latn",
            "estonian": "est_Latn",
            "lv": "lvs_Latn",
            "latvian": "lvs_Latn",
            "lt": "lit_Latn",
            "lithuanian": "lit_Latn",
        }

        lang_lower = language.lower()
        if lang_lower in mapping:
            return mapping[lang_lower]

        # If already in Omnilingual-ASR format, return as-is
        if "_" in language and len(language.split("_")) == 2:
            return language

        # Default fallback
        return "eng_Latn"

    def _get_audio_duration(self, audio_bytes: bytes) -> float:
        """Get the duration of audio in seconds."""
        try:
            audio_io = io.BytesIO(audio_bytes)
            info = sf.info(audio_io)
            return info.duration
        except Exception:
            return 0.0


def create_server(
    model_card: str = "omniASR_CTC_1B_v2",
    max_batch_size: int = 4,
    batch_timeout: float = 0.05,
    workers_per_device: int = 1,
) -> ls.LitServer:
    """
    Create and configure the LitServe server.

    Args:
        model_card: Model to load (see OmnilingualASRAPI for options)
        max_batch_size: Maximum batch size for inference
        batch_timeout: Timeout in seconds to wait for batch to fill
        workers_per_device: Number of workers per GPU

    Returns:
        Configured LitServer instance
    """
    api = OmnilingualASRAPI(model_card=model_card)

    server = ls.LitServer(
        api,
        accelerator="auto",
        max_batch_size=max_batch_size,
        batch_timeout=batch_timeout,
        workers_per_device=workers_per_device,
    )

    return server


def setup_routes(server: ls.LitServer):
    """
    Add OpenAI-compatible routes to the server.

    This adds the /v1/audio/transcriptions endpoint that matches
    OpenAI's Whisper API specification.
    """
    app = server.app

    @app.post("/v1/audio/transcriptions")
    async def transcribe(
        file: UploadFile,
        model: str = Form(default="omniASR_CTC_1B_v2"),
        language: Optional[str] = Form(default=None),
        prompt: Optional[str] = Form(default=None),
        response_format: str = Form(default="json"),
        temperature: float = Form(default=0.0),
        timestamp_granularities: Optional[str] = Form(default=None),
    ):
        """
        OpenAI Whisper-compatible transcription endpoint.

        Args:
            file: Audio file (wav, mp3, flac, etc.)
            model: Model to use (mapped to Omnilingual-ASR model cards)
            language: Language code (ISO 639-1 or Omnilingual-ASR format)
            prompt: Optional prompt (not currently used)
            response_format: json, verbose_json, text, srt, or vtt
            temperature: Sampling temperature (not currently used)
            timestamp_granularities: Timestamp detail level
        """
        # Validate file
        if not file.filename:
            raise HTTPException(status_code=400, detail="No file provided")

        # Read audio bytes
        audio_bytes = await file.read()

        if len(audio_bytes) == 0:
            raise HTTPException(status_code=400, detail="Empty file provided")

        # Parse timestamp granularities
        granularities = []
        if timestamp_granularities:
            granularities = [g.strip() for g in timestamp_granularities.split(",")]

        # Prepare request for the model
        request_data = {
            "audio_bytes": audio_bytes,
            "language": language,
            "response_format": response_format,
            "timestamp_granularities": granularities,
        }

        # Get the API instance and run inference
        api: OmnilingualASRAPI = server.lit_api

        # Decode, predict, encode
        decoded = api.decode_request(request_data)
        output = api.predict(decoded)
        encoded = api.encode_response(output, request_data)

        # Format response based on requested format
        result = encoded["output"]
        fmt = encoded["response_format"]

        if fmt == "text":
            return PlainTextResponse(content=result["text"])

        elif fmt == "verbose_json":
            response = TranscriptionVerboseResponse(
                text=result["text"],
                language=result["language"],
                duration=result["duration"],
                segments=[
                    TranscriptionSegment(
                        id=0,
                        seek=0,
                        start=0.0,
                        end=result["duration"],
                        text=result["text"],
                        temperature=temperature,
                        avg_logprob=0.0,
                        compression_ratio=1.0,
                        no_speech_prob=0.0,
                    )
                ],
            )
            return JSONResponse(content=response.model_dump())

        elif fmt == "srt":
            # Simple SRT format
            duration = result["duration"]
            srt_content = (
                f"1\n00:00:00,000 --> {_format_srt_time(duration)}\n{result['text']}\n"
            )
            return PlainTextResponse(content=srt_content, media_type="text/plain")

        elif fmt == "vtt":
            # Simple WebVTT format
            duration = result["duration"]
            vtt_content = f"WEBVTT\n\n00:00:00.000 --> {_format_vtt_time(duration)}\n{result['text']}\n"
            return PlainTextResponse(content=vtt_content, media_type="text/vtt")

        else:  # json (default)
            response = TranscriptionResponse(text=result["text"])
            return JSONResponse(content=response.model_dump())

    @app.get("/v1/models")
    async def list_models():
        """List available models (OpenAI-compatible)."""
        models = [
            # CTC models (fast, parallel generation)
            {"id": "omniASR_CTC_300M_v2", "object": "model", "owned_by": "meta"},
            {"id": "omniASR_CTC_1B_v2", "object": "model", "owned_by": "meta"},
            {"id": "omniASR_CTC_3B_v2", "object": "model", "owned_by": "meta"},
            {"id": "omniASR_CTC_7B_v2", "object": "model", "owned_by": "meta"},
            # LLM models (language-conditioned)
            {"id": "omniASR_LLM_300M_v2", "object": "model", "owned_by": "meta"},
            {"id": "omniASR_LLM_1B_v2", "object": "model", "owned_by": "meta"},
            {"id": "omniASR_LLM_3B_v2", "object": "model", "owned_by": "meta"},
            {"id": "omniASR_LLM_7B_v2", "object": "model", "owned_by": "meta"},
            # Unlimited audio length models
            {
                "id": "omniASR_LLM_Unlimited_300M_v2",
                "object": "model",
                "owned_by": "meta",
            },
            {
                "id": "omniASR_LLM_Unlimited_1B_v2",
                "object": "model",
                "owned_by": "meta",
            },
            {
                "id": "omniASR_LLM_Unlimited_3B_v2",
                "object": "model",
                "owned_by": "meta",
            },
            {
                "id": "omniASR_LLM_Unlimited_7B_v2",
                "object": "model",
                "owned_by": "meta",
            },
        ]
        return JSONResponse(content={"object": "list", "data": models})

    @app.get("/health")
    async def health():
        """Health check endpoint."""
        return JSONResponse(content={"status": "healthy"})


def _format_srt_time(seconds: float) -> str:
    """Format seconds to SRT timestamp format (HH:MM:SS,mmm)."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def _format_vtt_time(seconds: float) -> str:
    """Format seconds to WebVTT timestamp format (HH:MM:SS.mmm)."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"


if __name__ == "__main__":
    import os

    # Configuration from environment variables
    model_card = os.environ.get("OMNILINGUAL_MODEL", "omniASR_CTC_1B_v2")
    max_batch_size = int(os.environ.get("OMNILINGUAL_BATCH_SIZE", "4"))
    batch_timeout = float(os.environ.get("OMNILINGUAL_BATCH_TIMEOUT", "0.05"))
    workers = int(os.environ.get("OMNILINGUAL_WORKERS", "1"))
    port = int(os.environ.get("OMNILINGUAL_PORT", "8000"))

    print(f"Starting Omnilingual-ASR server with model: {model_card}")
    print(
        f"Batch size: {max_batch_size}, Timeout: {batch_timeout}s, Workers: {workers}"
    )

    server = create_server(
        model_card=model_card,
        max_batch_size=max_batch_size,
        batch_timeout=batch_timeout,
        workers_per_device=workers,
    )

    setup_routes(server)

    server.run(port=port, host="0.0.0.0")
