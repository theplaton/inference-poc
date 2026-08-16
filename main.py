"""Entry point for the inference PoC: make sure the model is on disk, then report it."""

import logging
import os
import sys

from prepare_model import prepare_model

logger = logging.getLogger("main")


def run_inference(model_id: str) -> int:
    """For now this only downloads the model; serving comes later."""
    path = prepare_model(model_id)
    logger.info("Ready: %s -> %s", model_id, path)
    return 0


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)

    model_id = os.environ.get("MODEL_ID")
    if not model_id:
        logger.error("MODEL_ID is not set. Example: MODEL_ID=Qwen/Qwen2.5-0.5B-Instruct")
        return 1

    return run_inference(model_id)


if __name__ == "__main__":
    sys.exit(main())
