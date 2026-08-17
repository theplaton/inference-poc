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
| `config.sh` | Settings for the bash tools: the five layers, and `require_config`. |
| `profiles/` | One file per model: checkpoint, parallelism, memory envelope, footprint. |
| `envfile.py` | The same layers for the python tools. |
| `defaults.env` | Committed values that do not change with the model. |
| `.env.example` | Template for the local, gitignored `.env`. |
| `requirements.txt` | The serving stack. `MIN_VLLM_VERSION` is parsed from it. |

## Settings

Five layers, highest precedence first: a `KEY=value` argument, then the exported
environment, then `.env`, then `profiles/$PROFILE.env`, then `defaults.env`. See
`defaults.env` for what is the same for every model and `profiles/` for what is
not; anything in either can be set from any layer.

```bash
./serve_model.sh PROFILE=granite           # the small model, one GPU
./serve_model.sh TP_SIZE=4 DP_SIZE=2       # 4-way TP across 2 DP ranks
./serve_model.sh PORT=8001 MAX_MODEL_LEN=131072
./serve_model.sh HEALTH_TIMEOUT=3600       # allow a slower cold start
./preflight.sh RUNTIME=docker              # check for docker, not a local vllm
./cleanup_vllm.sh VLLM_CLEANUP_TIMEOUT=60
python download_model.py PROFILE=granite   # ~2.5 GB instead of ~893 GB
```

## Profiles

`PROFILE` names the model. `deepseek_v4_tp8dp1_speculative` is this recipe; `granite`
(Granite 3.1 1B A400M) is a small MoE that exercises the same tools on one GPU
in about a minute, which is how you shake out the sweep, the CSVs and the signal
handling without spending 15 minutes per checkpoint load.

A profile is one file, `profiles/$PROFILE.env`, and it holds everything that run
needs. Two kinds of line:

| Line | Is | Overridable per run |
| --- | --- | --- |
| `MAX_MODEL_LEN=200000` | a setting | yes — argument, environment or `.env` |
| `--kv-cache-dtype fp8` | an engine argument, passed to `vllm serve` as written | no |

Which one a thing is follows from whether you would ever change it for a single
run. `MAX_MODEL_LEN` is a setting because sweeps retune it; `--tokenizer-mode
deepseek_v4` is an argument because a checkpoint needing a different one would be
a different profile.

Argument lines split on the first run of whitespace and no further, so a JSON
value needs no quoting and no escaping — `--compilation-config {"mode": 0,
"cudagraph_mode": "FULL_DECODE_ONLY"}` is one flag and one value however many
spaces it contains. `serve_model.sh` splices the block without knowing what any
flag means, so a new model is a new file rather than a new branch, and the block
can be diffed line by line against the recipe page it came from.

`TP_SIZE=1 DP_SIZE=1` with `EXPERT_PARALLEL=0` is one GPU and no sharding, which
is what `granite` runs.

`PROFILE_BASE` makes one profile inherit another: the base fills in every key the
derived file leaves unset, which is how every layer here already works. Argument
lines merge the same way, by flag — a derived profile restating a flag overrides
it, and the value `off` drops it. That is the whole of `deepseek_v4_tp8dp1`:

```bash
PROFILE_BASE=deepseek_v4_tp8dp1_speculative

--speculative-config off
```

Two lines, and it serves this recipe without speculative decoding, so a sweep
under each profile measures what dspark is worth and nothing else. A variant that
copied the numbers instead could drift from them, and the comparison would
quietly stop being one.

`MODEL_ID` comes from the profile. Setting it in `.env` outranks that, which is
the intended precedence but a good way to serve the wrong checkpoint after a
profile switch — `serve_model.sh` prints a note when the two disagree.

## Choices worth knowing

**Checkpoint.** `DeepSeek-V4-Pro-0813` is the official release and the recipe
default, superseding the preview (87.9 vs 72.1 on Terminal Bench 2.1). It carries
a fused DSpark speculative-decoding module, which is why `requirements.txt`
requires vLLM **0.25.0** rather than the model's 0.20.0 baseline, and why
`deepseek_v4_tp8dp1_speculative` carries `--speculative-config {"method":"dspark",...}`.
`deepseek_v4_tp8dp1` is the same profile without it.

**Parallelism.** `TP_SIZE x DP_SIZE` must equal `GPU_COUNT`; `EXPERT_PARALLEL=1`
shards a MoE's experts across that whole world. The recipe's layout is TP8/DP1,
which is what `deepseek_v4_tp8dp1_speculative` sets.

The experts are 96.6% of this checkpoint — 803 of 831 GiB, summed from the
shards — and expert parallel shards them across all 8 GPUs whatever TP and DP
are. So TP only divides the 28 GiB of dense weight that is left, and lowering it
costs ~3.5 GiB per GPU while buying a whole extra KV pool, because DP ranks
serve different requests where TP ranks replicate the same ones. V4 uses MLA,
whose KV is a single head however wide TP is, so TP8 stores one cache eight
times over.

| Layout | dense/GPU | weights/GPU | est. node KV |
| --- | --- | --- | --- |
| TP8/DP1 (default) | 3.5 GiB | 103.9 GiB | ~18.9 GiB |
| TP4/DP2 (`deepseek_v4_tp4dp2_speculative`) | 7.0 GiB | 107.5 GiB | ~30.8 GiB |
| TP2/DP4 (`deepseek_v4_tp2dp4_speculative`) | 14.1 GiB | 114.5 GiB | ~33.4 GiB |
| TP1/DP8 | 28.2 GiB | 128.6 GiB | does not fit |

Estimates, against 124.8 GiB usable per H200. DP also pays all-to-all traffic
for expert routing that TP8 does not, so more KV is not automatically more
throughput — which is what the sweep's `strategy` column is there to settle.

**Context is 200K, not 1M.** The recipe's H200 config sets `--max-model-len
200000` with `--max-num-seqs 16` (`DEFAULT_NUM_SEQS`, which
`../benchmark_sweep.sh` overrides per level). The model supports 1M, and the
guide's prose
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
