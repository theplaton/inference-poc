# DeepSeek-V4-Pro on 8x H200

Scripts for the official vLLM recipe, [`recipes.vllm.ai/deepseek-ai/DeepSeek-V4-Pro`](https://recipes.vllm.ai/deepseek-ai/DeepSeek-V4-Pro)
(recipe updated 2026-08-14). Flags are pinned from the recipe's machine-readable
[`h200.json`](https://recipes.vllm.ai/deepseek-ai/DeepSeek-V4-Pro/hw/h200.json), not
retyped from the prose.

1.6T total / 49B active MoE. The FP4+FP8 checkpoint needs ~960 GB against the
node's 1128 GB. vLLM only takes 95% of that (`--gpu-memory-utilization 0.95`),
so the real budget is ~1072 GB, leaving ~112 GB for KV cache and activations --
about 14 GB per GPU. That margin drives every tuning decision below.

## Run it

```bash
./install.sh                              # vllm >= 0.25.0 into ./venv
source venv/bin/activate
./preflight.sh                            # GPUs, VRAM, disk, version
python download.py                        # ~893 GB, go get a coffee
./serve.sh                                # OpenAI server on :8000
python smoke_test.py                      # in another shell
```

`./serve.sh --dry-run` prints the exact command without launching, which is the
fastest way to diff against the recipe page.

## Files

| File | Purpose |
| --- | --- |
| `recipe.env` | Every pinned value. Env vars override the defaults. |
| `install.sh` | vLLM into `./venv`, kept out of the repo's top-level `.venv`. |
| `preflight.sh` | Fails fast on GPU count, VRAM, disk, vLLM version. |
| `download.py` | Wraps the repo's `prepare_model()`, plus `*.py` for `--trust-remote-code`. |
| `serve.sh` | The launch command. `--strategy`, `--docker`, `--dry-run`. |
| `smoke_test.py` | Non-think / Think High / Think Max round trips. |
| `bench.sh` | Optional `vllm bench serve` against a live server. |

## Choices worth knowing

**Checkpoint.** `DeepSeek-V4-Pro-0813` is the official release and the recipe
default, superseding the preview (87.9 vs 72.1 on Terminal Bench 2.1). It carries
a fused DSpark speculative-decoding module, which is why `install.sh` requires
vLLM **0.25.0** rather than the model's 0.20.0 baseline, and why `serve.sh`
passes `--speculative-config '{"method":"dspark",...}'`.

**Parallelism.** Default is `tep` — `--tensor-parallel-size 8` with
`--enable-expert-parallel`. TP must equal the GPU count or replicated dense
layers OOM you. `--strategy dep` swaps in `--data-parallel-size 8`, which the
guide names as the H200 recommendation; it trades KV capacity (dense params are
replicated per rank) for better throughput at high concurrency. Try `tep` first.

**Context is 200K, not 1M.** The recipe's H200 config sets `--max-model-len
200000` with `--max-num-seqs 16`. The model supports 1M, and the guide's prose
mentions capping at 800K — but the generated H200 recipe says 200K, and 200K is
what fits alongside ~960 GB of weights. Trust the generated config.

**Think Max will truncate.** It needs `--max-model-len >= 393216`, which does not
fit in the leftover VRAM. `smoke_test.py` skips it unless you pass `--all`. To
actually use it, attach a host-DRAM KV tier — the recipe supports
`SimpleCPUOffloadConnector`, `MooncakeStoreConnector`, and `LMCacheMPConnector`.
Not wired up here.

**Don't reach for INT4 to buy headroom.** INT4 is software-emulated on Hopper:
you take a quality hit on reasoning and coding and get no tensor-core speedup.
The NVFP4 variant is for Blackwell only — it also can't use the `deep_gemm_mega_moe`
kernel, which is FP8-only.

## Not verified on hardware

None of this has been executed. The dev box here has one L4 (23 GB) and 109 GB of
free disk, so `preflight.sh` fails on it by design. The flags match the recipe
JSON exactly; the wrapper logic around them is unexercised.
