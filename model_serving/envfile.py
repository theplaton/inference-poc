"""Configuration for the model_serving python tools -- the same layers config.sh applies.

Precedence, highest wins:

    1. KEY=value arguments      python download_model.py MODEL_ID=...
    2. exported environment     MODEL_ID=... python download_model.py
    3. model_serving/.env       local, gitignored, holds tokens
    4. model_serving/profiles/$PROFILE.env   everything model-specific
    5. model_serving/defaults.env   committed defaults, the same for any model

Each layer only fills in keys no higher layer has set, so the order the layers are
applied in *is* the precedence: arguments go straight into os.environ, which is
therefore already populated when the two files are merged in with setdefault.

model_serving/ carries its own copy of this loader, .env and defaults so it stays
runnable on its own; benchmark/ has an independent set.
"""

import os
import re
import sys
from collections.abc import Iterable, Iterator
from pathlib import Path

CONFIG_DIR = Path(__file__).resolve().parent
REPO_ROOT = CONFIG_DIR.parent

ENV_FILE = CONFIG_DIR / ".env"
DEFAULTS_FILE = CONFIG_DIR / "defaults.env"
PROFILES_DIR = CONFIG_DIR / "profiles"

KEY = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _pairs(path: Path) -> Iterator[tuple[str, str]]:
    """Yield KEY, VALUE from a plain env file. A missing file yields nothing.

    Plain KEY=VALUE only -- no quoting, no shell expansion, no inline comments,
    matching what config.sh accepts on the bash side.
    """
    if not path.is_file():
        return
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        key, sep, value = line.partition("=")
        if sep and KEY.match(key.strip()):
            yield key.strip(), value


def load_config(argv: Iterable[str] | None = None) -> list[str]:
    """Apply every configuration layer to os.environ.

    Returns the arguments that were not KEY=value, in order, for the caller's own
    parser. `argv` defaults to sys.argv[1:].
    """
    rest: list[str] = []
    for arg in sys.argv[1:] if argv is None else argv:
        key, sep, value = arg.partition("=")
        if sep and KEY.match(key):
            os.environ[key] = value  # outranks the inherited environment
        else:
            rest.append(arg)

    for key, value in _pairs(ENV_FILE):
        os.environ.setdefault(key, value)  # setdefault is what gives higher layers priority

    # The profile decides which checkpoint these tools fetch and how much room it
    # needs, so it is resolved from the layers already applied, then from
    # defaults.env -- which must not be merged in before it, or it would outrank
    # the layer it sits below.
    profile = os.environ.get("PROFILE") or dict(_pairs(DEFAULTS_FILE)).get(
        "PROFILE", "deepseek_v4_speculative"
    )
    os.environ["PROFILE"] = profile
    profile_file = PROFILES_DIR / f"{profile}.env"
    if not profile_file.is_file():
        available = sorted(p.stem for p in PROFILES_DIR.glob("*.env"))
        print(
            f'error: no profile "{profile}" -- {profile_file} does not exist.\n'
            f"Available: {' '.join(available)}",
            file=sys.stderr,
        )
        raise SystemExit(1)

    # A profile may name another as its base, and takes from it everything it
    # does not set itself -- the same rule as every layer above, so a variant
    # profile holds only what makes it a variant. Read from the file rather than
    # the environment, so an inherited PROFILE_BASE cannot redirect an unrelated
    # run. config.sh resolves the chain identically.
    #
    # The engine-argument lines of a profile are not this loader's business:
    # _pairs skips them, and only serve_model.sh has a command line to put them
    # on.
    chain = [profile_file]
    seen = {profile}
    while base := dict(_pairs(chain[-1])).get("PROFILE_BASE", "").strip():
        if base in seen:
            print(
                f'error: profile "{profile}" inherits in a cycle ({base})',
                file=sys.stderr,
            )
            raise SystemExit(1)
        seen.add(base)
        base_file = PROFILES_DIR / f"{base}.env"
        if not base_file.is_file():
            print(
                f'error: profile "{profile}" names base "{base}", but '
                f"{base_file} does not exist.",
                file=sys.stderr,
            )
            raise SystemExit(1)
        chain.append(base_file)

    for path in (*chain, DEFAULTS_FILE):
        for key, value in _pairs(path):
            os.environ.setdefault(key, value)

    return rest


def require(*keys: str) -> None:
    """Abort with an actionable message when a required setting has no value."""
    missing = [key for key in keys if not os.environ.get(key)]
    if not missing:
        return

    name = Path(sys.argv[0]).name or "the script"
    print(f"error: required setting(s) not set: {', '.join(missing)}", file=sys.stderr)
    print("\nSet each one in any of these, highest precedence first:", file=sys.stderr)
    print(f"  1. an argument:      python {name} {missing[0]}=...", file=sys.stderr)
    print(f"  2. the environment:  export {missing[0]}=...", file=sys.stderr)
    print(f"  3. {ENV_FILE}  (cp .env.example .env)", file=sys.stderr)
    raise SystemExit(1)
