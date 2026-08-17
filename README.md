# inference-poc

Serves an open-weight model with vLLM and measures what comes back. Pinned to the
official recipe for DeepSeek-V4-Pro on an 8x H200 node.

| Entrypoint | Does |
| --- | --- |
| `./serve_model.sh` | Launch vLLM, wait for `/health`, shut down cleanly |
| `./benchmark.sh` | Smoke test the reasoning modes, then measure throughput and latency |
| `./benchmark_sweep.sh` | Drive both across a list of concurrency levels, into CSVs |
| `./server_metrics.sh` | Watch a running server: its settings, queue, KV cache, latency, GPUs |

The first three are thin wrappers around the folder that owns the work.
[model_serving/](model_serving/) owns the node, the checkpoint and the engine;
[benchmark/](benchmark/) only speaks HTTP to an OpenAI-compatible endpoint and
runs anywhere. Both folders have their own README with the detail this one skips.

## Setup, once per machine

```bash
dev/install_system.sh        # apt packages + uv (Ubuntu/Debian)
model_serving/install.sh     # build venv/ from requirements.txt
```

`install.sh` is the one step nothing does for you. `serve_model.sh` and
`benchmark.sh` both check for the venv and stop with `run
model_serving/install.sh` rather than building it themselves, so the runtime is
always something you chose to install.

Everything else is optional, and worth knowing when to reach for:

```bash
cp model_serving/.env.example model_serving/.env   # HF token, cache location
model_serving/preflight.sh                         # fail fast on a node that cannot hold it
venv/bin/python model_serving/download_model.py    # ~830 GB for V4-Pro
```

**`.env`** — a missing one is skipped, and the profile plus `defaults.env` supply
everything needed to serve and benchmark on localhost. Create it for a Hugging
Face token, or to point `HF_HUB_CACHE` at weights that live outside the repo;
otherwise the cache is `.hf-cache/hub` here. `benchmark/.env` is the same story
for an endpoint that is not localhost.

**`preflight.sh`** — a diagnostic over GPU count, VRAM, `/dev/shm`, disk and vLLM
version. Nothing depends on it; skipping it only means a bad node fails minutes
later inside the engine instead of seconds earlier here.

**`download_model.py`** — vLLM fetches missing weights on demand, so the question
is only whether that finishes inside `HEALTH_TIMEOUT`: 1800s for the DeepSeek
profiles, 600s for granite. Granite's 2.5 GB clears it; V4-Pro's 830 GB does not,
and the failure reads as a readiness timeout rather than a download, retried once
per sweep level. Pre-download anything large.

## Serve a model

`PROFILE` picks the model and everything that comes with it. It is the one choice
you make here:

```bash
./serve_model.sh                              # deepseek_v4_tp8dp1_speculative, the default
./serve_model.sh PROFILE=deepseek_v4_tp8dp1   # same checkpoint and layout, speculation off
./serve_model.sh PROFILE=granite              # a 2.5 GB model, ~30s to load
```

It stays in the foreground once healthy; Ctrl-C stops it and its workers. A
V4-Pro load takes 10-20 minutes before `/health` answers.

`./serve_model.sh --dry-run` prints the exact vLLM command and exits, which is
both the fastest way to check a profile and how to diff the flags against the
recipe page.

## Watch what it is doing

From a third shell, at any point during a serve or a sweep:

```bash
./server_metrics.sh        # redraw every second
./server_metrics.sh 5      # every five seconds
./server_metrics.sh --once # one frame, for a log or a paste
```

What the server was launched with — model, context length, data-parallel ranks,
KV cache size and dtype, prefix caching, speculation and how much of each draft
lands — then what it is doing right now: requests running and waiting, KV cache
use, mean TTFT and ITL. Below that a row per GPU: core and memory utilization,
memory used, power and temperature.

It only reads, so starting or stopping it never disturbs a run, and with no
server up it says so and keeps showing the GPUs — safe to start before a serve
and leave running across it. Two things it cannot tell you: the tensor-parallel
width, which vLLM does not report anywhere over HTTP, and anything about the
last few seconds — TTFT and ITL are means since the server started. For
per-measurement numbers, use the sweep's CSVs.

## Benchmark it, from another shell

Pass the same `PROFILE` — the client names a checkpoint in every request and vLLM
rejects any other:

```bash
./benchmark.sh PROFILE=granite                     # smoke test, then throughput
./benchmark.sh --smoke-only                        # just the reasoning-mode round trips
./benchmark.sh --bench-only CONCURRENCY=32 NUM_PROMPTS=256
```

## Or sweep concurrency

One command, and no server of your own: the sweep starts one per level, benchmarks
it, and tears it down.

```bash
./benchmark_sweep.sh --plan                        # what it would do, then exit
./benchmark_sweep.sh                               # the sweep itself; leave it running
./benchmark_sweep.sh PROFILE=deepseek_v4_tp8dp1
```

Each run names a server — a profile and a `tp`/`dp` layout — and a concurrency
level, and the sweep starts one whenever that combination changes, with
`DEFAULT_NUM_SEQS` set to the level it is about to drive. So every number
describes a server actually configured for that batch size rather than a wide one
being under-driven, and one sweep can span several servers. Reloading the
checkpoint per server is what makes it a multi-hour run. A run that fails is
reported at the end and does not stop the sweep.

## Choosing what to measure

[benchmark_sweep_config.json](benchmark_sweep_config.json) is the whole plan —
which request shapes and concurrency levels are worth an hour of this node is a
judgement about the model and the GPUs, not something to derive from a rule:

```json
{"sweeps": [
  {"isl": 2048,  "osl": 512,  "concurrency_levels": [1, 8, 64, 128, 164]},
  {"isl": 2048,  "osl": 8192, "concurrency_levels": [1, 8, 40], "prompts_per_level": 4},
  {"isl": 32768, "osl": 1024, "concurrency_levels": [1, 8, 12]}
]}
```

| Key | Means | Default |
| --- | --- | --- |
| `concurrency_levels` | how many requests in flight — required | — |
| `isl` / `osl` | input and output length per request: the shape | 2048 / 512 |
| `prompts_per_level` | requests sent = this x the level | 8 |
| `profile` | which model and engine flags this run needs a server for | `PROFILE` |
| `tp` / `dp` | tensor- and data-parallel width of that server; `tp` x `dp` must equal `GPU_COUNT` | the profile's |

An entry that omits a key gets the sweep's own resolved setting, so a sweep naming
nothing but levels still fully describes a server. An unknown key is refused
before the first load rather than ignored — `prompts_per_run` would otherwise
quietly measure eight times what it meant to.

Because `profile`, `tp` and `dp` are per entry, one config can compare things a
single server cannot hold at once — speculation against baseline, or one
parallel layout against another:

```json
{"sweeps": [
  {"profile": "deepseek_v4_tp8dp1_speculative", "concurrency_levels": [8, 64]},
  {"profile": "deepseek_v4_tp8dp1",             "concurrency_levels": [8, 64]},
  {"profile": "deepseek_v4_tp4dp2_speculative", "tp": 4, "dp": 2, "concurrency_levels": [64]}
]}
```

The shape is a client-side parameter, so every shape at a given server and level
is measured against the *same* server: the checkpoint loads once per distinct
(profile, tp, dp, level), not once per run. Entries differing only in shape
therefore share a load; entries differing in `profile`, `tp` or `dp` never do,
however well their levels line up — the three above cost five loads, not three.
Comparing servers is priced in checkpoint loads, so `--plan` counts them first.

Runs are ordered by server first, then level ascending, so an interrupted sweep
leaves whole layouts finished rather than a fragment of each. Requests scale with
the level so every run does the same number of batch rounds; comparing a run of
two rounds against one of fifty would compare warm-up against steady state.

## Where the output lives

Runs are numbered: the first sweep writes `results/1`, the next `results/2`.
Everything one invocation measures lands in its own folder, one row per
(server, level, shape). Per-run files are named for the server that produced
them — `<profile>-tp<N>dp<M>-c<level>` below — because a sweep spanning two
layouts would otherwise write both to the same filename and keep the second:

| File | Holds |
| --- | --- |
| `benchmark_sweep.csv` | concurrency, ISL, OSL, profile, layout, mean request latency, total throughput |
| `benchmark_sweep_detailed.csv` | the same rows with p99s, TTFT/TPOT/ITL, KV fit, spec-decode acceptance |
| `gpu_telemetry.csv` | per-GPU temperature, power and utilization while each measurement ran |
| `result-<server>-i<ISL>o<OSL>.json` | the raw `vllm bench serve` result for one measurement |
| `serve-<server>.log` | that server's output, shared by every shape it measured |
| `bench-<server>-i<ISL>o<OSL>.log` | that measurement's client output |

Every row carries `profile`, `strategy` (the layout as one word, `tp8dp1`), `tp`
and `dp`, so two rows at the same concurrency and shape are still tellable apart —
which is the whole point of letting one sweep span servers.

One file sits outside the run folders: `results/benchmark_sweep_all.csv`, which
every sweep appends to. It carries the run number and the same identifying
columns, so comparing today's run against last week's is one file rather than two
folders. When a newer version of the sweep adds columns it is widened in place and
older rows keep empty cells.

Every CSV is appended per measurement, so an interrupted sweep still leaves what
it finished. Read one at the terminal with `column -s, -t < FILE`.

### GPU telemetry

`gpu_telemetry.csv` says what the hardware was doing while those numbers were
produced. Two runs at the same throughput are not the same result if one held
620 W at 62 °C and the other throttled at 84 °C, and one hot GPU can pace the
whole group without showing up anywhere else. The utilization columns say whether
the node was actually busy — memory utilization is the more telling figure for
long-output shapes, where decode is bandwidth-bound.

`benchmark.sh` writes it, not the sweep, and samples only while a measurement is
in flight: the server is up far longer than any one benchmark, so a poller running
with it would average the idle gaps into every figure. Set `GPU_TELEMETRY_CSV` to
get the same row from a standalone run.

## Profiles

A DeepSeek profile is named `deepseek_v4_tp<N>dp<M>` for its parallel layout, plus
`_speculative` when dspark speculative decoding is on. The name is the whole
configuration, so a `--plan` line, a filename and a CSV row all say what produced
them without a lookup.

The six DeepSeek profiles are a 2x3: three parallel layouts, each with and
without speculation.

| Layout | Speculative | Not |
| --- | --- | --- |
| TP8/DP1 — the recipe as published | `deepseek_v4_tp8dp1_speculative` (default) | `deepseek_v4_tp8dp1` |
| TP4/DP2 — 2 replicas of 4-way TP | `deepseek_v4_tp4dp2_speculative` | `deepseek_v4_tp4dp2` |
| TP2/DP4 — 4 replicas of 2-way TP | `deepseek_v4_tp2dp4_speculative` | `deepseek_v4_tp2dp4` |

All six serve DeepSeek-V4-Pro-0813 and take hours to sweep. `granite` (Granite
3.1 1B A400M, one GPU) is the seventh, and takes minutes.

`granite` exists to exercise the tooling: same MoE shape as the big checkpoint at
1/900th the weight, so a bug in the sweep costs a 30-second load instead of a
15-minute one. The sweep exports `PROFILE` to the client, so both halves agree
without being wired together.

The DeepSeek profiles are variants of one recipe, each named for the single thing
that separates it from the others, so no run and no CSV row leaves you guessing
which server produced it. They exist to be compared — what speculative decoding is
worth, what a parallel layout costs — and since a sweep entry can name its own
profile, one config file measures the lot:

```json
{"sweeps": [
  {"profile": "deepseek_v4_tp8dp1_speculative", "concurrency_levels": [8, 64]},
  {"profile": "deepseek_v4_tp8dp1",             "concurrency_levels": [8, 64]}
]}
```

```bash
./benchmark_sweep.sh                               # both, one invocation
column -s, -t < results/benchmark_sweep_all.csv
```

Then read `mean_tpot_ms` across the two `profile` values at matched concurrency:
that ratio is what speculation actually bought, which `spec_decode_acceptance_length`
can only put a ceiling on.

A profile is one file per side — [model_serving/profiles/](model_serving/profiles/)
for the checkpoint, memory envelope and engine flags,
[benchmark/profiles/](benchmark/profiles/) for what requests carry. Adding a model
is adding those files, not editing a script; the format and the `PROFILE_BASE`
inheritance that makes `deepseek_v4_tp8dp1` two lines long are in
[model_serving/README.md](model_serving/README.md#profiles).

## Configuration

Each folder is self-contained: its own `.env`, `.env.example`, `defaults.env`,
`profiles/` and loader (`config.sh` for bash, `envfile.py` for python). No file
outside a folder configures the tools inside it, so either folder can be copied to
another machine on its own.

Five layers, highest precedence first:

| | Layer | Example |
| --- | --- | --- |
| 1 | `KEY=value` argument | `./serve_model.sh PORT=8001` |
| 2 | exported environment | `PORT=8001 ./serve_model.sh` |
| 3 | folder `.env` | `model_serving/.env`, gitignored, holds tokens |
| 4 | folder `profiles/$PROFILE.env` | everything that changes with the model |
| 5 | folder `defaults.env` | committed, the same for any model |

Every setting is reachable from every layer — there are no flags for things that
are also settings, and a tool aborts rather than guessing when a required value is
missing. `defaults.env` in each folder is the reference list of what exists. One
trap worth knowing: `.env` outranks the profile, so a `MODEL_ID` pinned there
survives a profile switch and serves the wrong checkpoint. The shipped `.env`
leaves it commented out and `serve_model.sh` says so when the two disagree.

## Layout

```
serve_model.sh          root entrypoint -> model_serving/serve_model.sh
benchmark.sh            root entrypoint -> benchmark/benchmark.sh
benchmark_sweep.sh      the concurrency sweep, driving both of the above
benchmark_sweep_config.json   which shapes and levels the sweep measures
model_serving/          the server, the node and the checkpoint
model_serving/profiles/ one file per model: parallelism, memory, engine flags
benchmark/              client-side evaluation over HTTP
benchmark/profiles/     one file per model: name, smoke-test modes
dev/                    host bootstrap: system packages, the dev pod spec
venv/                   created by model_serving/install.sh, gitignored
.hf-cache/hub/          model weights, gitignored, many GB
results/                sweep output, gitignored
```

Weights land in the repo-local cache rather than `~/.cache` so the PoC is
self-contained; `HF_HUB_CACHE` overrides it, which is what
[dev/dev-pod.yaml](dev/dev-pod.yaml) does to keep them on a volume that survives a
pod restart.

## State of play

`model_serving/preflight.sh` passes on this node: 8 GPUs, 1123 GB of aggregate
VRAM, 128 GiB of `/dev/shm`, and vLLM 0.27.1 against the 0.25.0 floor. The recipe
flags match `h200.json` exactly.

Full sweeps have run against the real checkpoint under the speculative profile,
up to concurrency 384; `results/` holds them. The baseline profile has been
verified by `--dry-run` but not yet loaded.
