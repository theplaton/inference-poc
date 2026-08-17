# model_serving — DeepSeek-V4-Pro on 8x H200

Everything that runs on the serving node: the engine, the checkpoint, and the
host it needs. Client-side evaluation lives in [benchmark/](../benchmark/).

Scripts for the official vLLM recipe, [`recipes.vllm.ai/deepseek-ai/DeepSeek-V4-Pro`](https://recipes.vllm.ai/deepseek-ai/DeepSeek-V4-Pro)
(recipe updated 2026-08-14). Flags are pinned from the recipe's machine-readable
[`h200.json`](https://recipes.vllm.ai/deepseek-ai/DeepSeek-V4-Pro/hw/h200.json), not
retyped from the prose.

1.6T total / 49B active MoE. The FP4+FP8 checkpoint needs ~960 GB against the
node's 1128 GB. vLLM only takes 95% of that (`--gpu-memory-utilization 0.95`),
so the real budget is ~1072 GB, leaving ~112 GB for KV cache and activations --
about 14 GB per GPU. That margin drives every tuning decision below.

## Major steps

Each is a tool you run yourself, in this order. Nothing here chains into
anything else, so a step you have already done is a step you can skip.

1. **Install the runtime — `./install.sh`.** Creates or reuses the repo-level
   `venv/` and installs vLLM, the OpenAI client, and the Hugging Face tooling
   from `requirements.txt`. The scripts use this environment directly, so it
   does not need to be activated.

2. **Validate the host — `./preflight.sh`.** Checks that all eight GPUs are
   visible, the checkpoint fits in aggregate VRAM, `/dev/shm` is large enough
   for vLLM workers, the model cache has enough free disk, and the installed
   vLLM version supports the checkpoint. Fix failures before downloading or
   starting the server. `serve_model.sh` does not run this for you: on a bad
   node it answers in seconds where the engine takes minutes to fail.

3. **Download the checkpoint — `python download_model.py`.** Fetches the roughly
   893 GB model and its remote-code Python files into the repo-local Hugging
   Face cache that `serve_model.sh` reads.

4. **Start inference — `./serve_model.sh`.** Launches the OpenAI-compatible
   server on port 8000 with the pinned H200 flags, waits for `/health`, and
   stays in the foreground. Ctrl-C stops it and its workers.

5. **Clear stragglers — `./cleanup_vllm.sh`.** `serve_model.sh` runs this on its
   way out. Run it standalone when an earlier run left workers holding VRAM.

```bash
./install.sh
./preflight.sh
../venv/bin/python download_model.py
./serve_model.sh
# Verify it from another shell, while serve_model.sh is still running:
../benchmark.sh
```

`./serve_model.sh --dry-run` prints the exact command without launching, which is
the fastest way to diff against the recipe page.

## Files

| File | Purpose |
| --- | --- |
| `serve_model.sh` | Launch, readiness, signals, failure reporting. Nothing else. |
| `install.sh` | vLLM into the repo-level `venv/`, from `requirements.txt`. |
| `preflight.sh` | Fails fast on GPU count, VRAM, `/dev/shm`, disk, vLLM version. |
| `download_model.py` | Wraps `prepare_model()`, plus `*.py` for `--trust-remote-code`. |
| `cleanup_vllm.sh` | Gracefully stops all vLLM processes, then force-stops stragglers. |
| `prepare_model.py` | `snapshot_download` into the repo-local hub cache. |
| `config.sh` | Settings for the bash tools: the four layers, and `require_config`. |
| `envfile.py` | The same four layers for the python tools. |
| `defaults.env` | Every pinned value, committed. The reference list of settings. |
| `.env.example` | Template for the local, gitignored `.env`. |
| `requirements.txt` | The serving stack. `MIN_VLLM_VERSION` is parsed from it. |

## Settings

Four layers, highest precedence first: a `KEY=value` argument, then the exported
environment, then `.env`, then `defaults.env`. See `defaults.env` for the full
list; anything in it can be set from any layer.

```bash
./serve_model.sh STRATEGY=dep              # data + expert parallel
./serve_model.sh PORT=8001 MAX_MODEL_LEN=131072
./serve_model.sh HEALTH_TIMEOUT=3600       # allow a slower cold start
./preflight.sh RUNTIME=docker              # check for docker, not a local vllm
./cleanup_vllm.sh VLLM_CLEANUP_TIMEOUT=60
python download_model.py MODEL_ID=Qwen/Qwen3-8B
```

`MODEL_ID` has no default on purpose — there is no sane checkpoint to guess, so
every tool that needs one aborts with the three places you can set it.

## Choices worth knowing

**Checkpoint.** `DeepSeek-V4-Pro-0813` is the official release and the recipe
default, superseding the preview (87.9 vs 72.1 on Terminal Bench 2.1). It carries
a fused DSpark speculative-decoding module, which is why `requirements.txt`
requires vLLM **0.25.0** rather than the model's 0.20.0 baseline, and why
`serve_model.sh` passes `--speculative-config '{"method":"dspark",...}'`.

**Parallelism.** Default is `tep` — `--tensor-parallel-size 8` with
`--enable-expert-parallel`. TP must equal the GPU count or replicated dense
layers OOM you. `STRATEGY=dep` swaps in `--data-parallel-size 8`, which the
guide names as the H200 recommendation; it trades KV capacity (dense params are
replicated per rank) for better throughput at high concurrency. Try `tep` first.

**Context is 200K, not 1M.** The recipe's H200 config sets `--max-model-len
200000` with `--max-num-seqs 16`. The model supports 1M, and the guide's prose
mentions capping at 800K — but the generated H200 recipe says 200K, and 200K is
what fits alongside ~960 GB of weights. Trust the generated config.

**Think Max will truncate.** It needs `--max-model-len >= 393216`, which does not
fit in the leftover VRAM. `benchmark/smoke_test.py` skips it unless you pass
`--all`. To actually use it, attach a host-DRAM KV tier — the recipe supports
`SimpleCPUOffloadConnector`, `MooncakeStoreConnector`, and `LMCacheMPConnector`.
Not wired up here.

**Don't reach for INT4 to buy headroom.** INT4 is software-emulated on Hopper:
you take a quality hit on reasoning and coding and get no tensor-core speedup.
The NVFP4 variant is for Blackwell only — it also can't use the `deep_gemm_mega_moe`
kernel, which is FP8-only.

**Shutdown signals a process group.** `serve_model.sh` starts vLLM in its own
process group so a TERM reaches the EngineCore workers, not just the parent.
Those workers hold VRAM and keep the log pipe open; a TERM to the parent alone
leaves both behind. `cleanup_vllm.sh` then sweeps anything that escaped.

## State of play

`preflight.sh` passes on this node — 8 GPUs, 1123 GB aggregate VRAM, 128 GiB of
`/dev/shm`, 1060 GB free in the cache, vLLM 0.27.1 against the 0.25.0 floor —
and the flags match `h200.json` exactly.

The launch, readiness, signal and failure paths have been exercised against stub
servers rather than a full V4-Pro load, which takes 10-20 minutes to reach
`/health`. The first real serve is recorded in commit 1ff6b4e.
