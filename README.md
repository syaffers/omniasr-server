# Omnilingual-ASR Model Server

A FastAPI-based ASR model server for [Omnilingual ASR](https://github.com/facebookresearch/omnilingual-asr) with an OpenAI Whisper-compatible API.

## Quick Start

### Local Development

```bash
# Install dependencies with uv
uv sync

# Run the server
uv run python main.py
```

### Building a Docker Image

Currently I'm building with CUDA 12.6 and PyTorch 2.8.0. To build against a different CUDA version, you need to update the sources and indices in [pyproject.toml](pyproject.toml).

```bash
# Build the image
docker build --build-arg MODEL_NAME=omniASR_LLM_300M_v2 -t omniasr-server .

# Or...
bash build.sh
```

See [`build.sh`](build.sh) for the build script.

Then, run with GPU support:

```bash
docker run --gpus all -p 8080:8080 omniasr-server
```

I'm open to 💡 on how to streamline the build process so I can build for multiple CUDA and PyTorch versions.

## API Usage

The API is (somewhat) compatible with OpenAI's Whisper transcription endpoint. Some parameters (like `model`) are ignored (for now!) since the server only hosts one model and Omnilingual-ASR doesn't have all the features of Whisper.

### Transcribe Audio

```bash
curl -X POST http://localhost:8080/v1/audio/transcriptions \
  -H "Content-Type: multipart/form-data" \
  -F "file=@audio.wav" \
  -F "model=omniASR_CTC_300M_v2"
```

### With Language Hint

This works for the LLM variants only. For CTC and W2V, the `language` parameter is ignored.

**ISO 639-1 (OpenAI API native)**

```bash
curl -X POST http://localhost:8080/v1/audio/transcriptions \
  -F "file=@audio.wav" \
  -F "model=omniASR_LLM_1B_v2" \
  -F "language=en"
```

**ISO 639-3 / Script (Omnilingual-ASR native)**

```bash
curl -X POST http://localhost:8080/v1/audio/transcriptions \
  -F "file=@audio.wav" \
  -F "model=omniASR_LLM_1B_v2" \
  -F "language=eng_Latn"
```

Languages are mapped heuristically from ISO 639-1 (Whisper's API) to Omnilingual-ASR's format. See how it's mapped in [`app/languages.py`](app/languages.py). For the best results, use Omnilingual-ASR's language codes.

### Response Formats

**JSON (default)**
```bash
curl -X POST http://localhost:8080/v1/audio/transcriptions -F "file=@audio.wav"
```
```json
{"text": "Hello, world!"}
```

**Plain Text**
```bash
curl -X POST http://localhost:8080/v1/audio/transcriptions \
  -F "file=@audio.wav" \
  -F "response_format=text"
```

### Python Client

```bash
uv run scripts/openai_client.py
```

See the [openai_client.py](scripts/openai_client.py) code. It's pretty straightforward.


## Configuration


### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_NAME` | `omniASR_CTC_300M_v2` | Model to use for transcription |
| `OMNILINGUAL_PORT` | `8080` | Server port |
| `OMNILINGUAL_HOST` | `0.0.0.0` | Server host |

### Changing the Model

See [Omnilingual-ASR's GitHub page](https://github.com/facebookresearch/omnilingual-asr/tree/main?tab=readme-ov-file#model-architectures) for a list of available models.

Set the `MODEL_NAME` environment variable to switch models when building:

```bash
docker run --gpus all -p 8080:8080 \
  -e MODEL_NAME=omniASR_CTC_1B_v2 \
  omniasr-server
```

Or when running locally:

```bash
MODEL_NAME=omniASR_CTC_1B_v2 uv run python main.py
```

**NOTE:** When running locally, on the first run, `fairseq` will download the weights and cache it to your device. Subsequent runs only loads the cached weights.

## Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/audio/transcriptions` | POST | Transcribe audio file |
| `/v1/models` | GET | List the deployed model |
| `/health-check` | GET | Health check |

## License

This server code is MIT licensed. The Omnilingual ASR models are released under Apache 2.0 by Meta.

