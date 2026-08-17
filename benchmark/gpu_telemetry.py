"""Average one benchmark's GPU samples into a row of gpu_telemetry.csv.

Reads the sample file the poller in benchmark.sh appends to -- one
"index, temperature, power, sm_util, memory_util" line per GPU per round -- and
appends a single row of per-GPU averages.

    python gpu_telemetry.py SAMPLES CSV RUN CONCURRENCY ISL OSL NUM_PROMPTS SAMPLED_S

Averaging and appending live here rather than in the shell for the same reason
they do in sweep_csv.py: one place builds the row, so the header and the values
cannot drift apart.

Temperatures are degrees C, power W, both utilizations percent, columns ordered
by nvidia-smi index. Every failure is a message on stderr and a non-zero exit
rather than an exception: the benchmark this describes has already produced its
numbers, and a telemetry problem must not turn a good run into a failed one.
"""

import csv
import os
import sys

samples_path, csv_path = sys.argv[1:3]
run, concurrency, isl, osl, num_prompts, sampled_s = sys.argv[3:9]

# index -> [samples, temperature sum, power sum, SM util sum, memory util sum]
totals = {}
with open(samples_path) as f:
    for line in f:
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 5:
            continue
        try:
            index = int(parts[0])
            temp, power, util, mem_util = (float(p) for p in parts[1:])
        except ValueError:
            continue  # "[N/A]" from a GPU that did not answer that round
        seen = totals.setdefault(index, [0, 0.0, 0.0, 0.0, 0.0])
        seen[0] += 1
        seen[1] += temp
        seen[2] += power
        seen[3] += util
        seen[4] += mem_util

if not totals:
    sys.exit("warning: no usable GPU samples, so no telemetry row was written")

row = {
    "run": run,
    "concurrency": concurrency,
    "isl": isl,
    "osl": osl,
    "num_prompts": num_prompts,
    "sampled_s": sampled_s,
    # The thinnest average in the row, so a figure from three samples is not
    # read with the confidence of one from three hundred.
    "samples": min(seen[0] for seen in totals.values()),
}
for index in sorted(totals):
    count, temp_sum, power_sum, util_sum, mem_util_sum = totals[index]
    row[f"GPU_{index}_avg_temp"] = round(temp_sum / count, 1)
    row[f"GPU_{index}_avg_power"] = round(power_sum / count, 1)
    row[f"GPU_{index}_avg_util"] = round(util_sum / count, 1)
    row[f"GPU_{index}_avg_mem_util"] = round(mem_util_sum / count, 1)

is_new = not (os.path.exists(csv_path) and os.path.getsize(csv_path) > 0)
if not is_new:
    with open(csv_path, newline="") as f:
        have = next(csv.reader(f), [])
    if have != list(row):
        # A different GPU count, or a different CUDA_VISIBLE_DEVICES. Not worth
        # failing a benchmark that has already produced good numbers.
        sys.exit(
            f"warning: {csv_path} has the columns of a different node, so this "
            f"run's telemetry was not appended.\n  in the file: {','.join(have)}"
            f"\n  this run writes: {','.join(row)}\nMove or delete it, or point "
            f"GPU_TELEMETRY_CSV somewhere else."
        )

with open(csv_path, "a", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(row))
    if is_new:
        writer.writeheader()
    writer.writerow(row)
