"""Pull the checkpoint down to the local Hub cache, ready for serve_model.sh.

    python download_model.py                        # MODEL_ID from .env
    python download_model.py MODEL_ID=Qwen/Qwen3-8B # or from an argument
    python download_model.py DOWNLOAD_REMOTE_CODE=0 # weights only, no *.py

V4-Pro is served with --trust-remote-code, so its custom modelling code (*.py)
has to come down with the shards; the weights-only default of ALLOW_PATTERNS
would silently skip it. DOWNLOAD_REMOTE_CODE=0 turns that addition off for a
plain checkpoint that does not ship code.

~893 GB across ~97 safetensors shards for V4-Pro -- expect this to take a while,
and run ./preflight.sh first: it checks the free disk for you.
"""

import logging
import os
import sys

from envfile import load_config, require
from prepare_model import ALLOW_PATTERNS, prepare_model

logger = logging.getLogger("download")


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)

    rest = load_config()
    if rest:
        print(f"unknown argument: {rest[0]} (settings are passed as KEY=value)", file=sys.stderr)
        return 2

    require("MODEL_ID")
    model_id = os.environ["MODEL_ID"]

    patterns = list(ALLOW_PATTERNS)
    if os.environ.get("DOWNLOAD_REMOTE_CODE", "1") != "0":
        patterns.append("*.py")

    logger.info("Fetching %s into %s", model_id, os.environ.get("HF_HUB_CACHE", "the default cache"))

    path = prepare_model(model_id, allow_patterns=patterns)

    logger.info("Done: %s", path)
    logger.info("Serve it with: ./serve_model.sh")
    return 0


if __name__ == "__main__":
    sys.exit(main())
