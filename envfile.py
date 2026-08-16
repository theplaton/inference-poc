""".env is the single place every setting lives -- model id, tokens, cache paths.

Both halves of the project read it: recipe.env on the bash side, load_env() here for
python, applying the same precedence -- real environment > .env > per-script fallback.
So `MODEL_ID=... ./serve.sh` still overrides for one-offs.

.env is gitignored so tokens stay local; .env.example is the committed template.
Copy it on a fresh clone: cp .env.example .env
"""

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent

ENV_FILES = (".env",)


def load_env(*paths: str | os.PathLike) -> None:
    """Populate os.environ from `paths`, defaulting to ENV_FILES at the repo root.

    Relative paths resolve against the repo root, not the cwd, so it does not matter
    where a script was launched from. A missing file is skipped silently -- callers
    report the omission themselves, in terms of the variable they actually needed.
    """
    for name in paths or ENV_FILES:
        path = Path(name)
        if not path.is_absolute():
            path = REPO_ROOT / path
        if not path.is_file():
            continue

        for raw in path.read_text().splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            key, sep, value = line.partition("=")
            if not sep:
                continue
            # setdefault is what gives the real environment priority.
            os.environ.setdefault(key.strip(), value.strip())
