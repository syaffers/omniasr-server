"""
Entry point for the Omnilingual-ASR model server.

This module simply runs the server from server.py.
Use `python main.py` or `python server.py` to start the server.
"""

if __name__ == "__main__":
    from server import create_server, setup_routes
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
