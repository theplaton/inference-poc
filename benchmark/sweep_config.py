"""Read benchmark_sweep_config.json and print one line per run of the sweep.

    python sweep_config.py CONFIG DEFAULT_ISL DEFAULT_OSL DEFAULT_PROMPTS_PER_LEVEL

Which shapes and concurrencies are worth an hour of this node is a judgement, so
it is written down in the config file rather than derived from a rule here. This
script only reads, validates and orders it; ../benchmark_sweep.sh reads the
"level isl osl prompts" lines back into parallel arrays.

Validation lives here because this is what parses the JSON anyway, and a bad
config should stop the sweep before the first checkpoint load rather than twenty
minutes in. Every rejection names the file and the offending sweep.
"""

import json
import sys

path, default_isl, default_osl, default_prompts = sys.argv[1:5]

try:
    with open(path) as f:
        doc = json.load(f)
except OSError as e:
    sys.exit(f"cannot read the sweep config: {e}")
except ValueError as e:
    sys.exit(f"{path} is not valid JSON: {e}")


def whole(value, what):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        sys.exit(f"{path}: {what} must be a positive whole number, got {value!r}")
    return value


# Three spellings, one meaning: a list of sweeps, the older bare level list, or
# a bare array of levels. The last two describe one sweep at the ISL/OSL
# settings, which is what they meant before shapes existed.
if isinstance(doc, list):
    entries = [{"concurrency_levels": doc}]
elif isinstance(doc, dict) and "sweeps" in doc:
    entries = doc["sweeps"]
    if not isinstance(entries, list) or not entries:
        sys.exit(f'{path}: "sweeps" must be a non-empty array')
elif isinstance(doc, dict) and "concurrency_levels" in doc:
    entries = [doc]
else:
    sys.exit(f'{path}: expected a "sweeps" array or a "concurrency_levels" array')

runs = []
for i, entry in enumerate(entries):
    if not isinstance(entry, dict):
        sys.exit(f"{path}: sweep {i} must be an object, got {entry!r}")

    levels = entry.get("concurrency_levels")
    if not isinstance(levels, list) or not levels:
        sys.exit(f'{path}: sweep {i} needs a non-empty "concurrency_levels" array')

    isl = whole(entry.get("isl", int(default_isl)), f"sweep {i} isl")
    osl = whole(entry.get("osl", int(default_osl)), f"sweep {i} osl")
    per_level = whole(
        entry.get("prompts_per_level", int(default_prompts)),
        f"sweep {i} prompts_per_level",
    )

    for level in levels:
        whole(level, f"sweep {i} concurrency level")
        runs.append((level, isl, osl, level * per_level))

# Level-ascending, and stable within a level so the config's order survives.
# Ascending because a level that cannot fit should fail after the smaller ones
# have already been recorded, not before.
runs.sort(key=lambda r: r[0])

for level, isl, osl, prompts in runs:
    print(level, isl, osl, prompts)
