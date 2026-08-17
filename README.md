# inference-poc

Serves an open-weight model with vLLM and measures what comes back. Pinned to the
official recipe for DeepSeek-V4-Pro on an 8x H200 node.

Two entrypoints at the root, each a thin wrapper around the folder that owns the
work:

| Entrypoint | Wraps | Does |
| --- | --- | --- |
| `./serve_model.sh` | [model_serving/serve_model.sh](model_serving/serve_model.sh) | Launch vLLM, wait for `/health`, shut down cleanly, report failures |
| `./benchmark.sh` | [benchmark/benchmark.sh](benchmark/benchmark.sh) | Smoke test the reasoning modes, then measure throughput and latency |

Everything else is a separately runnable tool inside one of the two folders. The
split is server side versus client side: [model_serving/](model_serving/) owns
the node, the checkpoint and the engine; [benchmark/](benchmark/) only speaks
HTTP to an OpenAI-compatible endpoint and runs anywhere.

## Quickstart

```bash
dev/install_system.sh                     # apt packages + uv (Ubuntu/Debian)
cp model_serving/.env.example model_serving/.env   # set MODEL_ID
cp benchmark/.env.example benchmark/.env           # the same MODEL_ID

model_serving/install.sh                  # build venv/ from requirements.txt
model_serving/preflight.sh                # fail fast on a node that cannot hold it
venv/bin/python model_serving/download_model.py    # ~893 GB for V4-Pro

./serve_model.sh                          # stays in the foreground once healthy
# in another shell:
./benchmark.sh
```

`./serve_model.sh --dry-run` prints the exact vLLM command and exits, which is
the fastest way to diff the flags against the recipe page.

## Configuration

Each folder is self-contained: its own `.env`, `.env.example`, `defaults.env` and
loader (`config.sh` for bash, `envfile.py` for python). No file outside a folder
configures the tools inside it, so either folder can be copied to another machine
on its own.

Four layers, highest precedence first:

| | Layer | Example |
| --- | --- | --- |
| 1 | `KEY=value` argument | `./serve_model.sh PORT=8001` |
| 2 | exported environment | `PORT=8001 ./serve_model.sh` |
| 3 | folder `.env` | `model_serving/.env`, gitignored, holds tokens |
| 4 | folder `defaults.env` | committed, every pinned recipe value |

Every setting is reachable from every layer — there are no flags for things that
are also settings. A tool aborts rather than guessing when a required value such
as `MODEL_ID` is missing anywhere in the chain.

Settings files are plain `KEY=VALUE`: no quoting, no shell expansion, no inline
comments. `defaults.env` in each folder is the reference list of what exists.

## Layout

```
serve_model.sh          root entrypoint -> model_serving/serve_model.sh
benchmark.sh            root entrypoint -> benchmark/benchmark.sh
model_serving/          the server, the node and the checkpoint
benchmark/              client-side evaluation over HTTP
dev/                    host bootstrap: system packages, the dev pod spec
venv/                   created by model_serving/install.sh, gitignored
.hf-cache/hub/          model weights, gitignored, many GB
```

Weights land in the repo-local cache rather than `~/.cache` so the PoC is
self-contained; `HF_HUB_CACHE` overrides it, which is what
[dev/dev-pod.yaml](dev/dev-pod.yaml) does to keep them on a volume that survives
a pod restart.

## VS Code

Two launch configs, `download_model` and `smoke_test`, run the python entrypoints
with the matching folder `.env` loaded through `envFile`. They use the `debugpy`
adapter, so select the `venv` interpreter first (Python: Select Interpreter).

## State of play

`model_serving/preflight.sh` passes on this node: 8 GPUs, 1123 GB of aggregate
VRAM, 128 GiB of `/dev/shm`, and vLLM 0.27.1 against the 0.25.0 floor. The
recipe flags match `h200.json` exactly.

The launch, readiness, signal and failure paths in `serve_model.sh` have been
exercised against stub servers rather than a full V4-Pro load; the first real
serve is recorded in commit 1ff6b4e.
