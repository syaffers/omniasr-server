"""Pre-download model and tokenizer to cache."""

import os

import torch
from fairseq2.data.tokenizers.hub import load_tokenizer
from fairseq2.models.hub import load_model

MODEL_NAME = os.getenv("MODEL_NAME", "omniASR_CTC_300M_v2")


def preload_model():
    """Download model and tokenizer to cache (CPU only, for caching purposes)."""
    print(f"Pre-downloading model: {MODEL_NAME}")

    # Use CPU with float32 for download - actual inference will use appropriate device/dtype
    load_model(MODEL_NAME, device=torch.device("cpu"), dtype=torch.float32)
    print(f"Model {MODEL_NAME} downloaded")

    load_tokenizer(MODEL_NAME)
    print(f"Tokenizer {MODEL_NAME} downloaded")


if __name__ == "__main__":
    preload_model()
