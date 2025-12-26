.PHONY: build run dev

IMAGE_NAME=omnilingual-asr
MODEL_NAME ?= omniASR_CTC_300M_v2
IMAGE_TAG=$(shell echo $(MODEL_NAME) | sed 's/^omniASR_//' | tr 'A-Z_' 'a-z-')

build:
	docker buildx build --platform linux/amd64 --build-arg MODEL_NAME=$(MODEL_NAME) -t $(IMAGE_NAME):$(IMAGE_TAG) .

run:
	docker run --rm -it --gpus all -p 8080:8080 $(IMAGE_NAME):$(IMAGE_TAG)

local:
	uv run main.py

debug:
	uv run python -m debugpy --listen 0.0.0.0:5678 main.py
