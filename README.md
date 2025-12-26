# Omnilingual-ASR Model Server

High-performance ASR model server for [Omnilingual ASR](https://github.com/facebookresearch/omnilingual-asr) with an OpenAI Whisper-compatible API.

## Features

- 🚀 **High Performance**: Built on [LitServe](https://github.com/Lightning-AI/LitServe) with batching support
- 🎯 **OpenAI Compatible**: Drop-in replacement for OpenAI's Whisper API
- 🌍 **1600+ Languages**: Leverages Meta's Omnilingual ASR models
- 🐳 **GPU Ready**: Optimized Docker container with CUDA support

## Quick Start

### Using Docker (Recommended)

```bash
# Build the image
docker build -t omnilingual-asr .

# Run with GPU support
docker run --gpus all -p 8000:8000 omnilingual-asr
```

### Local Development

```bash
# Install dependencies with uv
uv sync --extra-index-url https://download.pytorch.org/whl/cu121

# Run the server
uv run python server.py
```

## API Usage

The API is compatible with OpenAI's Whisper transcription endpoint.

### Transcribe Audio

```bash
curl -X POST http://localhost:8000/v1/audio/transcriptions \
  -H "Content-Type: multipart/form-data" \
  -F "file=@audio.wav" \
  -F "model=omniASR_CTC_1B_v2"
```

### With Language Hint

```bash
curl -X POST http://localhost:8000/v1/audio/transcriptions \
  -F "file=@audio.wav" \
  -F "model=omniASR_LLM_1B_v2" \
  -F "language=en"
```

### Response Formats

**JSON (default)**
```bash
curl -X POST http://localhost:8000/v1/audio/transcriptions \
  -F "file=@audio.wav" \
  -F "response_format=json"
```
```json
{"text": "Hello, world!"}
```

**Verbose JSON**
```bash
curl -X POST http://localhost:8000/v1/audio/transcriptions \
  -F "file=@audio.wav" \
  -F "response_format=verbose_json"
```
```json
{
  "task": "transcribe",
  "language": "en",
  "duration": 2.5,
  "text": "Hello, world!",
  "segments": [...]
}
```

**Plain Text**
```bash
curl -X POST http://localhost:8000/v1/audio/transcriptions \
  -F "file=@audio.wav" \
  -F "response_format=text"
```

**SRT/VTT Subtitles**
```bash
curl -X POST http://localhost:8000/v1/audio/transcriptions \
  -F "file=@audio.wav" \
  -F "response_format=srt"
```

### Python Client

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed"  # No auth required
)

with open("audio.wav", "rb") as audio_file:
    transcription = client.audio.transcriptions.create(
        model="omniASR_CTC_1B_v2",
        file=audio_file
    )
    print(transcription.text)
```

## Available Models

| Model | Type | Description |
|-------|------|-------------|
| `omniASR_CTC_300M_v2` | CTC | Fast, small model |
| `omniASR_CTC_1B_v2` | CTC | Balanced performance (default) |
| `omniASR_CTC_3B_v2` | CTC | Higher quality |
| `omniASR_CTC_7B_v2` | CTC | Best CTC quality |
| `omniASR_LLM_*_v2` | LLM | Language-conditioned, autoregressive |
| `omniASR_LLM_Unlimited_*_v2` | LLM | Unlimited audio length support |

**CTC models** are faster due to parallel generation but don't support language conditioning.

**LLM models** support language hints for better accuracy but are slower.

## Configuration

Configure via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `OMNILINGUAL_MODEL` | `omniASR_CTC_1B_v2` | Model to load |
| `OMNILINGUAL_BATCH_SIZE` | `4` | Max batch size for inference |
| `OMNILINGUAL_BATCH_TIMEOUT` | `0.05` | Seconds to wait for batch |
| `OMNILINGUAL_WORKERS` | `1` | Workers per GPU |
| `OMNILINGUAL_PORT` | `8000` | Server port |

### Example: Running the 3B Model

```bash
docker run --gpus all -p 8000:8000 \
  -e OMNILINGUAL_MODEL=omniASR_CTC_3B_v2 \
  -e OMNILINGUAL_BATCH_SIZE=2 \
  omnilingual-asr
```

## Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/audio/transcriptions` | POST | Transcribe audio file |
| `/v1/models` | GET | List available models |
| `/health` | GET | Health check |

## Audio Requirements

- **Supported formats**: WAV, FLAC, MP3, OGG, and other formats supported by libsndfile
- **Sample rate**: Automatically resampled to 16kHz
- **Channels**: Automatically converted to mono
- **Duration**: Models work best with audio ≤30 seconds. Use `Unlimited` variants for longer audio.

## Performance Tips

1. **Use CTC models** for highest throughput when language conditioning isn't needed
2. **Increase batch size** if you have memory headroom
3. **Use the right model size** for your GPU memory:
   - 300M: ~2GB VRAM
   - 1B: ~4GB VRAM
   - 3B: ~12GB VRAM
   - 7B: ~28GB VRAM

## License

This server code is MIT licensed. The Omnilingual ASR models are released under Apache 2.0 by Meta.

