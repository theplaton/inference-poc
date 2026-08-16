"""Pull the DeepSeek-V4-Pro checkpoint down to the local Hub cache.

Reuses the repo's prepare_model(), with one addition: V4-Pro is served with
--trust-remote-code, so the custom modelling code (*.py) has to come down with
the shards. The default ALLOW_PATTERNS would silently skip it.

~893 GB across ~97 safetensors shards -- expect this to take a while and check
`df -h` first (preflight.sh does it for you).
"""

import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from prepare_model import ALLOW_PATTERNS, prepare_model  # noqa: E402

logger = logging.getLogger("deepseek_poc.download")

DEFAULT_MODEL_ID = "deepseek-ai/DeepSeek-V4-Pro-0813"

# Custom modelling code for --trust-remote-code, on top of the usual weights.
V4_PRO_PATTERNS = [*ALLOW_PATTERNS, "*.py"]


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)

    model_id = os.environ.get("MODEL_ID") or DEFAULT_MODEL_ID
    logger.info("Fetching %s (this is ~893 GB)", model_id)

    path = prepare_model(model_id, allow_patterns=V4_PRO_PATTERNS)

    logger.info("Done: %s", path)
    logger.info("Serve it with: ./serve.sh")
    return 0


if __name__ == "__main__":
    sys.exit(main())
