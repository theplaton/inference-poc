"""Configuration for the benchmark python tools -- the same four layers config.sh applies.

Precedence, highest wins:

    1. KEY=value arguments      python smoke_test.py MODEL_ID=...
    2. exported environment     MODEL_ID=... python smoke_test.py
    3. benchmark/.env           local, gitignored
    4. benchmark/defaults.env   committed defaults

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

    for path in (ENV_FILE, DEFAULTS_FILE):
        for key, value in _pairs(path):
            # setdefault is what gives every higher layer priority.
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
