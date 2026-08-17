"""Build the sweep's three CSVs from one run's result JSON.

    python sweep_csv.py migrate SUMMARY DETAILED ROLLUP
    python sweep_csv.py write RESULT SUMMARY DETAILED ROLLUP \\
        LEVEL ISL OSL PROMPTS KV_TOKENS RUN_ID STRATEGY MAX_MODEL_LEN PROFILE

Three files, each answering a different question: the summary is "how does this
node scale", the detailed one is everything the run measured for when the
summary raises a question, and the rollup is "how does this sweep compare to the
last one". All three are appended per run, so an interrupted sweep still leaves
what it finished.

Two entry points, one row builder: `migrate` widens a CSV written by an older
version of this script, and `write` appends a row. Both call build_rows(), so
the columns cannot drift apart -- which is the whole reason this is one file
rather than a header list in the shell and a writer somewhere else.
"""

import csv
import json
import os
import sys


def build_rows(r, ctx):
    def num(key, digits=2):
        v = r.get(key)
        return round(v, digits) if isinstance(v, (int, float)) else ""

    def sec(key, digits=3):
        """A latency vLLM reports in ms, in seconds -- easier to compare."""
        v = r.get(key)
        return round(v / 1000, digits) if isinstance(v, (int, float)) else ""

    # What share of the engine's KV cache this run's requests want at once.
    # Over 100% means the level could not have run at full width whatever the
    # numbers say, because the scheduler was queueing.
    kv_fit = ""
    try:
        pool = int(ctx["kv_tokens"])
        wanted = int(ctx["level"]) * (int(ctx["isl"]) + int(ctx["osl"]))
        kv_fit = round(100 * wanted / pool, 1) if pool else ""
    except (KeyError, TypeError, ValueError):
        pass

    summary = {
        "concurrency": ctx["level"],
        "isl": ctx["isl"],
        "osl": ctx["osl"],
        "mean_request_latency_s": sec("mean_e2el_ms"),
        "total_throughput_tok_s": num("total_token_throughput"),
    }

    # No separate column for the server's batch width: the sweep always starts
    # the server at the concurrency it then drives, so it would repeat the first
    # column in every row.
    detailed = {
        "concurrency": ctx["level"],
        "num_prompts": ctx["prompts"],
        "completed": r.get("completed", ""),
        "input_len": ctx["isl"],
        "output_len": ctx["osl"],
        "total_output_tokens": r.get("total_output_tokens", ""),
        "duration_s": num("duration"),
        "request_throughput_req_s": num("request_throughput", 4),
        "output_throughput_tok_s": num("output_throughput"),
        "total_throughput_tok_s": num("total_token_throughput"),
        "mean_request_latency_s": sec("mean_e2el_ms"),
        "median_request_latency_s": sec("median_e2el_ms"),
        "p99_request_latency_s": sec("p99_e2el_ms"),
        "mean_ttft_ms": num("mean_ttft_ms"),
        "p99_ttft_ms": num("p99_ttft_ms"),
        "mean_tpot_ms": num("mean_tpot_ms"),
        "p99_tpot_ms": num("p99_tpot_ms"),
        "mean_itl_ms": num("mean_itl_ms"),
        "p99_itl_ms": num("p99_itl_ms"),
        "spec_decode_acceptance_rate": num("spec_decode_acceptance_rate", 4),
        # Mean tokens emitted per pass through the target model. The rate above
        # says what share of the drafts survived; this says what that bought,
        # and is the ceiling on what speculation can be worth: 2.3 means at best
        # 2.3x the decode speed, before the wider verify pass is paid for.
        "spec_decode_acceptance_length": num("spec_decode_acceptance_length", 3),
        "kv_pool_tokens": ctx["kv_tokens"],
        "kv_fit_pct": kv_fit,
        "ignore_eos": r.get("ignore_eos", ""),
        "random_range_ratio": r.get("random_range_ratio", ""),
        "model_id": r.get("model_id", ""),
        "date": r.get("date", ""),
    }

    # Carries the settings that make two sweeps different -- without them a row
    # from a dep/131072 sweep is indistinguishable from a tep/200000 one.
    rollup = {
        "run": ctx["run_id"],
        # Two sweeps can differ in a way no other column here shows: same
        # checkpoint, same strategy, same context, different engine flags. The
        # profile is the name of that difference -- without it a speculative run
        # and a baseline one are the same row twice.
        "profile": ctx["profile"],
        "concurrency": ctx["level"],
        "isl": ctx["isl"],
        "osl": ctx["osl"],
        "strategy": ctx["strategy"],
        "max_model_len": ctx["max_model_len"],
        "mean_request_latency_s": sec("mean_e2el_ms"),
        "total_throughput_tok_s": num("total_token_throughput"),
        "output_throughput_tok_s": num("output_throughput"),
        "p99_request_latency_s": sec("p99_e2el_ms"),
        # Here as well as in the detailed file because the question it answers --
        # is speculation still paying at this batch size -- is asked by comparing
        # two sweeps, which is what this file is for.
        "spec_decode_acceptance_length": num("spec_decode_acceptance_length", 3),
        "kv_fit_pct": kv_fit,
        "ignore_eos": r.get("ignore_eos", ""),
        "random_range_ratio": r.get("random_range_ratio", ""),
        "model_id": r.get("model_id", ""),
    }

    return summary, detailed, rollup


CTX_KEYS = (
    "level",
    "isl",
    "osl",
    "prompts",
    "kv_tokens",
    "run_id",
    "strategy",
    "max_model_len",
    # Appended rather than slotted in beside run_id: these keys are positional
    # arguments, so a new one goes last or every caller has to be rewritten.
    "profile",
)

BLANK_CTX = dict.fromkeys(CTX_KEYS, "")


def header_of(path):
    if not (os.path.exists(path) and os.path.getsize(path) > 0):
        return None
    with open(path, newline="") as f:
        return next(csv.reader(f), None)


mode = sys.argv[1]

# A rollup written by an older version of this script has fewer columns. Widen
# it in place rather than refusing to append: the file exists to accumulate,
# and old rows simply have nothing to say about the new columns.
if mode == "migrate":
    for path, row in zip(sys.argv[2:5], build_rows({}, BLANK_CTX)):
        have = header_of(path)
        if have is None or have == list(row):
            continue
        added = [c for c in row if c not in have]
        if not added:
            continue
        with open(path, newline="") as f:
            old_rows = list(csv.DictReader(f))
        merged = have + added
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=merged)
            writer.writeheader()
            for old in old_rows:
                writer.writerow({c: old.get(c, "") for c in merged})
        print(f"note: added {', '.join(added)} to {path}; earlier rows left blank there")
    sys.exit(0)

(result_path, summary_csv, detailed_csv, rollup_csv) = sys.argv[2:6]
# Over BLANK_CTX so a context key the caller did not pass becomes an empty cell
# rather than a KeyError: the positional list grows as the sweep records more,
# and a stale caller should write a thinner row, not fail a finished run.
ctx = {**BLANK_CTX, **dict(zip(CTX_KEYS, sys.argv[6:]))}

with open(result_path) as f:
    result = json.load(f)

for path, row in zip((summary_csv, detailed_csv, rollup_csv), build_rows(result, ctx)):
    # Follow the file's own column order when it already has one, so a widened
    # rollup keeps its shape and a row missing a column writes an empty cell
    # instead of shifting every value one place left.
    header = header_of(path)
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header or list(row), extrasaction="ignore")
        if header is None:
            writer.writeheader()
            writer.writerow(row)
        else:
            writer.writerow({c: row.get(c, "") for c in header})
