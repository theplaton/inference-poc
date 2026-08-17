# benchmark — client-side evaluation

Everything that talks to a running server over HTTP. Nothing here starts,
configures or stops one; the server side lives in
[model_serving/](../model_serving/).

Because it is only an OpenAI-compatible client, this folder runs anywhere — the
serving node, your laptop, a CI box — against any endpoint you point it at.

## Major steps

1. **Smoke test — `python smoke_test.py`.** Sends the same prompt through each
   reasoning mode and checks a complete response comes back. This is the check
   that says the server is actually serving, not merely answering `/health`.

2. **Throughput — `./benchmark.sh --bench-only`.** Measures TTFT, TPOT, ITL and
   end-to-end latency with `vllm bench serve`. The recipe publishes no H200
   numbers, so this measures your node rather than checking it against a target.

`./benchmark.sh` runs both, in that order, and stops if the smoke test fails.

```bash
../serve_model.sh          # in one shell
./benchmark.sh             # in another, once it reports ready
```

## Files

| File | Purpose |
| --- | --- |
| `benchmark.sh` | The entrypoint: health check, smoke test, throughput run. |
| `smoke_test.py` | Non-think / Think High / Think Max round trips. |
| `config.sh` | Settings for the bash tools: the five layers, and `require_config`. |
| `profiles/` | One file per model: which checkpoint to name, which smoke-test modes exist. |
| `envfile.py` | The same layers for the python tools. |
| `defaults.env` | Committed defaults. The reference list of settings. |
| `.env.example` | Template for the local, gitignored `.env`. |
| `requirements.txt` | `openai`. See below for the throughput half. |

## Settings

Five layers, highest precedence first: a `KEY=value` argument, then the exported
environment, then `.env`, then `profiles/$PROFILE.env`, then `defaults.env`. See
`defaults.env` for the full list and `profiles/` for what changes with the model;
anything in either can be set from any layer.

```bash
./benchmark.sh NUM_PROMPTS=128 CONCURRENCY=32
./benchmark.sh RANDOM_INPUT_LEN=8192 RANDOM_OUTPUT_LEN=1024
./benchmark.sh BASE_URL=http://gpu-01:8000/v1
./benchmark.sh RESULT_FILE=runs/c32.json
python smoke_test.py MODEL_ID=Qwen/Qwen3-8B
```

`RESULT_FILE` saves the throughput run's numbers as JSON alongside the table on
stdout, which is how a caller reads them back: `../benchmark_sweep.sh` sets it
per measurement and builds its CSVs from the files. Unset, the default, the run
leaves nothing behind.

`IGNORE_EOS` (default 1) makes every request generate the full
`RANDOM_OUTPUT_LEN` instead of stopping at the model's EOS, so a run that asks
for 8192 tokens measures 8192 tokens. `RANDOM_RANGE_RATIO` (default 0.0) samples
lengths around ISL/OSL rather than fixing them — 0.3 for traffic that looks more
real, 0.0 for a scaling curve you want clean. Both are recorded in the result
JSON as metadata, so a saved run says which were in force.

`MODEL_ID` comes from the profile, because the server decides which checkpoint is
loaded and rejects a request naming any other: run `PROFILE=granite` here when
the server was started with `PROFILE=granite`, and the names match by
construction. `../benchmark_sweep.sh` exports `PROFILE` for exactly that reason.

The endpoint is spelled two ways because the tools want different shapes:
`HOST` and `PORT` for `vllm bench serve`, `BASE_URL` for the OpenAI client.
`BASE_URL` is derived as `http://$HOST:$PORT/v1` unless you set it, which is
what you do for an endpoint the pair cannot express — https, a path prefix, a
proxy. The health probe is `BASE_URL` with `/v1` swapped for `/health`, and
`HEALTH_URL` overrides that.

## Installing

`model_serving/install.sh` already puts `openai` in the repo-level `venv/`, and
these scripts pick that up automatically. On a machine that is not the serving
node:

```bash
uv pip install -r requirements.txt
```

`vllm` is deliberately not declared here: pinning it in two places would let the
client and the server drift, so the throughput half borrows the `vllm` CLI from
the serving install. Without it, `./benchmark.sh --smoke-only` still works and
the throughput run reports what is missing.

## Reasoning modes

Which modes exist is a property of the chat template, so the profile names them:
`SMOKE_MODES` for a plain run, `SMOKE_MODES_ALL` for `--all`. The V4 template
exposes reasoning effort through `chat_template_kwargs` rather than a top-level
parameter, so each of its modes is a different `extra_body`; `plain` is the one
mode every model has, and is all the `granite` profile runs — sending V4's
kwargs to a template without them is rejected, not ignored.

`TEMPERATURE` and `TOP_P` default to DeepSeek's recommendation for the 0813
release: temperature 1.0 always, top_p 0.95 for agentic work and 1.0 otherwise.

Think Max is opt-in (`--all`) because it truncates unless the server was started
with `--max-model-len >= 393216`, which does not fit on 8x H200 without a KV
offload tier.

## GPU thermals

`GPU_THERMALS_CSV` appends one row per throughput run — the average temperature
and power draw of each GPU while that run was in flight:

```bash
./benchmark.sh --bench-only GPU_THERMALS_CSV=runs/gpu_thermals.csv RUN_LABEL=c8
```

```
run,concurrency,isl,osl,num_prompts,sampled_s,samples,GPU_0_avg_temp,GPU_0_avg_power,...
c8,8,2048,512,64,84,42,63.4,611.2,...
```

The window is the benchmark and nothing else. Polling from the serving side
would be easier, but the server is up across many measurements and the idle gaps
between them would be averaged into every figure; only the client knows when a
measurement starts and stops.

`sampled_s` is that window in seconds, and `samples` the number of readings
behind the thinnest average in the row — a figure from three samples deserves
less confidence than one from three hundred. `GPU_POLL_INTERVAL` (default 2s)
sets the spacing. Temperatures are °C, power W, columns ordered by `nvidia-smi`
index.

Unset — the default — nothing is polled. It needs `nvidia-smi` on the machine
running the benchmark, so it measures the local node or nothing: pointed at a
remote endpoint it prints a note and the run continues unaffected. A failed run
gets no row, and neither does one whose CSV was written by a node with a
different GPU count; both say so on stderr rather than failing a benchmark whose
numbers are good.

## Concurrency

`CONCURRENCY` above the batch width the server was started with does not
increase parallelism, it just queues. `benchmark.sh` warns when you cross
`DEFAULT_NUM_SEQS` from `defaults.env` (16, matching the recipe); keep it in
step with `model_serving/defaults.env` if you retune the server.

Sweeping both together — a server at `DEFAULT_NUM_SEQS=N` measured at
`CONCURRENCY=N`, for each N and each request shape in
`../benchmark_sweep_config.json` — is what `../benchmark_sweep.sh` does.
