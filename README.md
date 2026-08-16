# inference-poc

Downloads a Hugging Face model so it is ready for local inference.

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Run

```bash
MODEL_ID=Qwen/Qwen2.5-0.5B-Instruct .venv/bin/python main.py
```

Weights download into `.hf-cache/hub/` inside the repo (gitignored), not `~/.cache`,
so the PoC is self-contained. `prepare_model()` returns the snapshot path to hand to
a serving engine.

## Environment variables

| Variable | Required | Purpose |
| --- | --- | --- |
| `MODEL_ID` | yes | Hub repo id, e.g. `Qwen/Qwen2.5-0.5B-Instruct` |
| `MODEL_REVISION` | no | Branch/tag/commit, defaults to `main` |
| `HF_TOKEN` | no | Hugging Face access token for gated/private repos |
| `HF_HUB_CACHE` | no | Hub cache dir, overrides the repo-local default |
| `HF_HOME` | no | Cache root; the hub cache becomes `$HF_HOME/hub` |
| `HF_XET_HIGH_PERFORMANCE` | no | `1` for faster Xet-backed downloads |

Open-weight models on the Hub are public, so no credentials are required — the
script only authenticates when `HF_TOKEN` is populated.

## VS Code

One launch config, `run_inference`, runs `main.py` and reads `.env` via `envFile`, so
change the model by editing `MODEL_ID` there — no rebuild, no prompt.
Copy `.env.example` to `.env` if you don't have one; `.env` is gitignored so tokens
stay local. The configs use the `debugpy` adapter, so select the `.venv` interpreter
first (Python: Select Interpreter).
