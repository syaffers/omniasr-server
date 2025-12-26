start-local:
	uv run --no-dev main.py

start-debug:
	uv run python -m debugpy --listen 0.0.0.0:5678 main.py
