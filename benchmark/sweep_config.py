"""Read benchmark_sweep_config.json and print one line per run of the sweep.

    python sweep_config.py CONFIG DEFAULT_ISL DEFAULT_OSL DEFAULT_PROMPTS \\
        DEFAULT_PROFILE DEFAULT_TP DEFAULT_DP

Which shapes, concurrencies and server layouts are worth an hour of this node is
a judgement, so it is written down in the config file rather than derived from a
rule here. This script only reads, validates and orders it.

Each line is "profile tp dp level isl osl prompts". The first three name the
server that run needs; ../benchmark_sweep.sh reloads the checkpoint whenever
they or the level change, so lines are sorted by (profile, tp, dp, level) to
keep every run that can share one server adjacent.

Sorting by server first rather than by level means an interrupted sweep leaves
whole layouts finished rather than a fragment of each -- the comparison between
two layouts is the point, and half of each answers nothing.

Validation lives here because this is what parses the JSON anyway, and a bad
config should stop the sweep before the first checkpoint load rather than twenty
minutes in. Every rejection names the file and the offending sweep.
"""

import json
import sys

(
    path,
    default_isl,
    default_osl,
    default_prompts,
    default_profile,
    default_tp,
    default_dp,
) = sys.argv[1:8]

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


def size(value, what):
    """A parallel dimension, or "-" for "whatever the profile says".

    An entry naming a profile but not a layout means that profile's own layout,
    which only the shell can resolve -- it is the side that reads profile files
    and follows PROFILE_BASE. Passing the marker through keeps this script from
    having to know where profiles live.
    """
    return value if value == "-" else whole(value, what)


# The defaults arrive as command-line text while an entry's own values arrive as
# JSON numbers, so they are made to agree here rather than at each use.
def size_arg(value, what):
    if value == "-":
        return value
    if not value.isdigit() or int(value) < 1:
        sys.exit(f"{what} must be a positive whole number or '-', got {value!r}")
    return int(value)


default_tp = size_arg(default_tp, "TP_SIZE")
default_dp = size_arg(default_dp, "DP_SIZE")


def word(value, what):
    """A profile name. It becomes a filename and a shell argument, so it is held
    to the characters a profile file is actually allowed to be called."""
    if not isinstance(value, str) or not value:
        sys.exit(f"{path}: {what} must be a non-empty string, got {value!r}")
    if not all(c.isalnum() or c in "_-." for c in value) or value.startswith("."):
        sys.exit(f"{path}: {what} is not a valid profile name: {value!r}")
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

# Keys a sweep may set. Anything else is a typo that would otherwise be ignored
# in silence -- "prompts_per_run" would quietly measure eight times what it
# meant to, and "dp_size" would quietly measure the wrong server.
KNOWN = {
    "profile",
    "tp",
    "dp",
    "isl",
    "osl",
    "concurrency_levels",
    "prompts_per_level",
}

runs = []
for i, entry in enumerate(entries):
    if not isinstance(entry, dict):
        sys.exit(f"{path}: sweep {i} must be an object, got {entry!r}")

    unknown = sorted(set(entry) - KNOWN)
    if unknown:
        sys.exit(
            f"{path}: sweep {i} has unknown key(s): {', '.join(unknown)}\n"
            f"  known keys: {', '.join(sorted(KNOWN))}"
        )

    levels = entry.get("concurrency_levels")
    if not isinstance(levels, list) or not levels:
        sys.exit(f'{path}: sweep {i} needs a non-empty "concurrency_levels" array')

    profile = word(entry.get("profile", default_profile), f"sweep {i} profile")
    tp = size(entry.get("tp", default_tp), f"sweep {i} tp")
    dp = size(entry.get("dp", default_dp), f"sweep {i} dp")
    isl = whole(entry.get("isl", int(default_isl)), f"sweep {i} isl")
    osl = whole(entry.get("osl", int(default_osl)), f"sweep {i} osl")
    per_level = whole(
        entry.get("prompts_per_level", int(default_prompts)),
        f"sweep {i} prompts_per_level",
    )

    for level in levels:
        whole(level, f"sweep {i} concurrency level")
        runs.append((profile, tp, dp, level, isl, osl, level * per_level))

# By server first, then level ascending, and stable within a group so the
# config's order of shapes survives. Level ascending because a level that cannot
# fit should fail after the smaller ones have been recorded, not before.
#
# tp and dp compare as text because one of them may be the "-" marker, and only
# adjacency matters for them -- it is the level that has to order numerically.
runs.sort(key=lambda r: (r[0], str(r[1]), str(r[2]), r[3]))

for run in runs:
    print(*run)
