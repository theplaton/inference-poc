"""Pull the checkpoint down to the local Hub cache, ready for serve_model.sh.

    python download_model.py                        # MODEL_ID from .env
    python download_model.py MODEL_ID=Qwen/Qwen3-8B # or from an argument

V4-Pro is served with --trust-remote-code, so its custom modelling code (*.py)
has to come down with the shards; the weights-only default of ALLOW_PATTERNS
would silently skip it.

~893 GB across ~97 safetensors shards for V4-Pro -- expect this to take a while,
and run ./preflight.sh first: it checks the free disk for you.
"""

import logging
import os
import sys

from envfile import load_config, require
from prepare_model import ALLOW_PATTERNS, prepare_model

logger = logging.getLogger("download")

# Custom modelling code for --trust-remote-code, on top of the usual weights.
V4_PRO_PATTERNS = [*ALLOW_PATTERNS, "*.py"]


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)

    rest = load_config()
    if rest:
        print(f"unknown argument: {rest[0]} (settings are passed as KEY=value)", file=sys.stderr)
        return 2

    require("MODEL_ID")
    model_id = os.environ["MODEL_ID"]

    logger.info("Fetching %s into %s", model_id, os.environ.get("HF_HUB_CACHE", "the default cache"))

    path = prepare_model(model_id, allow_patterns=V4_PRO_PATTERNS)

    logger.info("Done: %s", path)
    logger.info("Serve it with: ./serve_model.sh")
    return 0


if __name__ == "__main__":
    sys.exit(main())
