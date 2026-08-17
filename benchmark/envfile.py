"""Configuration for the benchmark python tools -- the same layers config.sh applies.

Precedence, highest wins:

    1. KEY=value arguments      python smoke_test.py MODEL_ID=...
    2. exported environment     MODEL_ID=... python smoke_test.py
    3. benchmark/.env           local, gitignored
    4. benchmark/profiles/$PROFILE.env   everything model-specific
    5. benchmark/defaults.env   committed defaults, the same for any model

Each layer only fills in keys no higher layer has set, so the order the layers are
applied in *is* the precedence: arguments go straight into os.environ, which is
therefore already populated when the two files are merged in with setdefault.

benchmark/ carries its own copy of this loader, .env and defaults so it stays
runnable against any OpenAI-compatible endpoint, with or without model_serving/.
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

    # The profile decides which model these tools talk to and what its chat
    # template understands, so it is resolved from the layers already applied,
    # then from defaults.env -- which must not be merged in before it, or it
    # would outrank the layer it sits below.
    profile = os.environ.get("PROFILE") or dict(_pairs(DEFAULTS_FILE)).get(
        "PROFILE", "deepseek_v4"
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

    for path in (profile_file, DEFAULTS_FILE):
        for key, value in _pairs(path):
            os.environ.setdefault(key, value)

    return rest


def base_url() -> str:
    """The endpoint the OpenAI client should talk to.

    BASE_URL wins when set; otherwise it is built from HOST and PORT, the pair the
    vLLM bench CLI wants. config.sh derives it the same way.
    """
    if url := os.environ.get("BASE_URL"):
        return url
    host = os.environ.get("HOST") or "localhost"
    port = os.environ.get("PORT") or "8000"
    return f"http://{host}:{port}/v1"


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
