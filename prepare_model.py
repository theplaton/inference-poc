"""Download a Hugging Face model to the local cache.

The model repo id comes from the MODEL_ID environment variable. Authentication is
optional: open-weight models (Qwen, Mistral, SmolLM, ...) are public, so we only log
in when HF_TOKEN is set. Gated repos (Llama, Gemma, ...) will fail without it.
"""

import logging
import os
from pathlib import Path

from huggingface_hub import login, snapshot_download

logger = logging.getLogger(__name__)

# Keep weights next to the repo instead of ~/.cache, so the PoC is self-contained.
# Resolved from __file__, not cwd, so it holds wherever the script is launched from.
REPO_HUB_CACHE = Path(__file__).resolve().parent / ".hf-cache" / "hub"

# Weight shards, config and tokenizer only -- skip duplicate formats and extras.
ALLOW_PATTERNS = [
    "*.safetensors",
    "*.safetensors.index.json",
    "*.json",
    "*.model",
    "*.txt",
]


def hub_cache_dir() -> str:
    """Where snapshots land. HF_HUB_CACHE / HF_HOME override the repo-local default.

    huggingface_hub reads these at import time, so we resolve them ourselves and pass
    the result to snapshot_download explicitly. HF_HOME is a root dir -- the hub cache
    lives in its `hub` subdirectory.
    """
    if hub_cache := os.environ.get("HF_HUB_CACHE"):
        return hub_cache
    if hf_home := os.environ.get("HF_HOME"):
        return str(Path(hf_home).expanduser() / "hub")
    return str(REPO_HUB_CACHE)


def hf_login() -> bool:
    """Log into the Hub if a token is available. Returns True when authenticated."""
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        logger.info("No HF_TOKEN set, continuing anonymously (public models only)")
        return False

    login(token=token, add_to_git_credential=False)
    logger.info("Logged into Hugging Face Hub")
    return True


def prepare_model(
    model_id: str | None = None,
    revision: str | None = None,
    allow_patterns: list[str] | None = None,
) -> str:
    """Download `model_id` and return the local path of the snapshot.

    `allow_patterns` overrides ALLOW_PATTERNS for models that need extra files,
    e.g. the custom modelling code a `--trust-remote-code` checkpoint ships.
    """
    model_id = model_id or os.environ.get("MODEL_ID")
    if not model_id:
        raise ValueError("MODEL_ID is not set (env var or argument)")

    revision = revision or os.environ.get("MODEL_REVISION") or "main"

    hf_login()

    cache_dir = hub_cache_dir()
    logger.info("Downloading %s (revision %s) into %s", model_id, revision, cache_dir)
    path = snapshot_download(
        repo_id=model_id,
        revision=revision,
        cache_dir=cache_dir,
        allow_patterns=allow_patterns or ALLOW_PATTERNS,
    )
    logger.info("Model available at %s", path)
    return path


if __name__ == "__main__":
    from envfile import load_env

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    load_env()
    print(prepare_model())
