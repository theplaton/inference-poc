# inference-poc

Serves an open-weight model with vLLM and measures what comes back. Pinned to the
official recipe for DeepSeek-V4-Pro on an 8x H200 node.

Two entrypoints at the root, each a thin wrapper around the folder that owns the
work, and a third that drives both:

| Entrypoint | Wraps | Does |
| --- | --- | --- |
| `./serve_model.sh` | [model_serving/serve_model.sh](model_serving/serve_model.sh) | Launch vLLM, wait for `/health`, shut down cleanly, report failures |
| `./benchmark.sh` | [benchmark/benchmark.sh](benchmark/benchmark.sh) | Smoke test the reasoning modes, then measure throughput and latency |
| `./benchmark_sweep.sh` | both of the above | Benchmark each concurrency level in `benchmark_sweep_config.json`, reloading the model per level, into two CSVs |

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

## Concurrency sweep

```bash
./benchmark_sweep.sh --plan        # the plan and the exact per-level commands
./benchmark_sweep.sh               # the sweep itself; leave it running
```

`./benchmark_sweep.sh` answers one question — what does this node do as
concurrency rises — by measuring each level on its own server. For every level
it starts `./serve_model.sh` with `DEFAULT_NUM_SEQS` set to that level, waits
for `/health`, runs `./benchmark.sh --bench-only` at the same `CONCURRENCY` for
each request shape, and tears the server down before the next level. Reloading
the checkpoint per level is what makes it a multi-hour run, and the reason each
number describes a server actually configured for that batch size rather than
one wide server being under-driven.

What to measure lives in
[benchmark_sweep_config.json](benchmark_sweep_config.json), because which shapes
and concurrencies are worth an hour of this node is a judgement about the model
and the GPUs, not something to derive from a rule:

```json
{"sweeps": [
  {"isl": 2048,  "osl": 512,  "concurrency_levels": [1, 8, 64, 128, 164]},
  {"isl": 2048,  "osl": 8192, "concurrency_levels": [1, 8, 40], "prompts_per_level": 4},
  {"isl": 32768, "osl": 1024, "concurrency_levels": [1, 8, 12]}
]}
```

The shape is a client-side parameter, so every shape at a given concurrency is
measured against the *same* server: the checkpoint loads once per distinct
level, not once per (level, shape). Aligning the level lists on 1 and 8 is what
turns 11 measurements into 7 loads.

Each shape asks a different question, and each has its own ceiling — the KV pool
is ~508k tokens (18.95 GiB per GPU at 39.1 KiB per token), so a level is capped
by tokens per request:

| Shape | tok/req | KV ceiling | What it exercises |
| --- | --- | --- | --- |
| 2048/512 | 2,560 | ~198 | the comparable baseline; prefill-heavy |
| 2048/8192 | 10,240 | ~49 | a full reasoning trace, decode-dominated |
| 32768/1024 | 33,792 | ~15 | long context and the sparse-attention path |

Top levels sit near 80% of each ceiling; past it requests queue rather than run,
which shows up as latency climbing while throughput stays flat. Every row records
`kv_fit_pct` so a queue-limited measurement is visible rather than inferred.

Requests scale with the level (`prompts_per_level`, default 8) so every run does
the same number of batch rounds. A run that fails is reported at the end and does
not stop the sweep.

Runs are numbered: the first sweep writes `results/1`, the next `results/2`, and
everything one invocation measures lands inside its own folder, one row per
(level, shape):

| File | Holds |
| --- | --- |
| `benchmark_sweep.csv` | concurrency, ISL, OSL, mean request latency (s), total throughput (tok/s) |
| `benchmark_sweep_detailed.csv` | the same rows with p99s, TTFT/TPOT/ITL, actual output tokens, KV fit, spec-decode acceptance |
| `result-c<N>-i<ISL>o<OSL>.json` | the raw `vllm bench serve` result for one measurement |
| `serve-c<N>.log` | that level's server output — one per level, shared by its shapes |
| `bench-c<N>-i<ISL>o<OSL>.log` | that measurement's client output |

One file sits outside the run folders: `results/benchmark_sweep_all.csv`, which
every sweep appends to. It carries the run number and the settings that make two
sweeps different — strategy, max model length, ISL/OSL, `ignore_eos`,
checkpoint — so comparing today's run against last week's is one file, not two
folders. When a newer version of the sweep adds columns, the rollup is widened
in place and older rows keep empty cells rather than being refused.

All three CSVs are appended per measurement, so an interrupted sweep still
leaves what it finished. Read one at the terminal with `column -s, -t < FILE`.

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
benchmark_sweep.sh      the concurrency sweep, driving both of the above
benchmark_sweep_config.json   which concurrency levels the sweep measures
model_serving/          the server, the node and the checkpoint
benchmark/              client-side evaluation over HTTP
dev/                    host bootstrap: system packages, the dev pod spec
venv/                   created by model_serving/install.sh, gitignored
.hf-cache/hub/          model weights, gitignored, many GB
results/                sweep output, gitignored
```

`benchmark_sweep.sh` is the one root script with logic of its own, because a
sweep is neither server side nor client side: it belongs to neither folder and
drives both through the same entrypoints and settings a person would use by hand.

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
