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
| `config.sh` | Settings for the bash tools: the four layers, and `require_config`. |
| `envfile.py` | The same four layers for the python tools. |
| `defaults.env` | Committed defaults. The reference list of settings. |
| `.env.example` | Template for the local, gitignored `.env`. |
| `requirements.txt` | `openai`. See below for the throughput half. |

## Settings

Four layers, highest precedence first: a `KEY=value` argument, then the exported
environment, then `.env`, then `defaults.env`. See `defaults.env` for the full
list; anything in it can be set from any layer.

```bash
./benchmark.sh NUM_PROMPTS=128 CONCURRENCY=32
./benchmark.sh RANDOM_INPUT_LEN=8192 RANDOM_OUTPUT_LEN=1024
./benchmark.sh BASE_URL=http://gpu-01:8000/v1
./benchmark.sh RESULT_FILE=runs/c32.json
python smoke_test.py MODEL_ID=Qwen/Qwen3-8B
```

`RESULT_FILE` saves the throughput run's numbers as JSON alongside the table on
stdout, which is how a caller reads them back: `../benchmark_sweep.sh` sets it
per sweep level and builds its CSVs from the files. Unset, the default, the run
leaves nothing behind.

`MODEL_ID` has no default on purpose — the server decides which checkpoint is
loaded and rejects a request naming any other, so guessing is worse than
aborting. Set it to whatever `model_serving/` was started with.

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

The V4 chat template exposes reasoning effort through `chat_template_kwargs`
rather than a top-level parameter, so each mode is a different `extra_body`.
`TEMPERATURE` and `TOP_P` default to DeepSeek's recommendation for the 0813
release: temperature 1.0 always, top_p 0.95 for agentic work and 1.0 otherwise.

Think Max is opt-in (`--all`) because it truncates unless the server was started
with `--max-model-len >= 393216`, which does not fit on 8x H200 without a KV
offload tier.

## Concurrency

`CONCURRENCY` above the server's `--max-num-seqs` does not increase parallelism,
it just queues. `benchmark.sh` warns when you cross the value in `defaults.env`
(16, matching the recipe); keep the two in step if you retune the server.

Sweeping both together — a server at `MAX_NUM_SEQS=N` measured at
`CONCURRENCY=N`, for each N in `../benchmark_sweep_config.json` — is what
`../benchmark_sweep.sh` does.
