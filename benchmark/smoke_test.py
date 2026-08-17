"""Hit a running server with one round trip per reasoning mode.

Which modes exist is a property of the chat template, so the active profile
names them: SMOKE_MODES for a plain run, SMOKE_MODES_ALL for --all. V4 exposes
reasoning effort through chat_template_kwargs rather than a top-level parameter,
so each of its modes is a different extra_body; a model without that switch runs
the one mode every model has.

    python smoke_test.py                              # SMOKE_MODES
    python smoke_test.py --all                        # SMOKE_MODES_ALL
    python smoke_test.py PROFILE=granite              # the small model
    python smoke_test.py BASE_URL=http://gpu-01:8000/v1

Think Max truncates unless the server was started with --max-model-len >= 393216,
which does not fit on 8x H200 without KV offload, so it is opt-in.
"""

import argparse
import os
import sys
import time

from openai import OpenAI

from envfile import base_url, load_config, require

PROMPT = "What is 17*19? Return only the final integer."

MODES = {
    # Every model has this one: a chat request with nothing added to it.
    "plain": None,
    "non-think": None,
    "high": {"chat_template_kwargs": {"thinking": True, "reasoning_effort": "high"}},
    "max": {"chat_template_kwargs": {"thinking": True, "reasoning_effort": "max"}},
}


def run_mode(client: OpenAI, model: str, name: str, extra_body: dict | None) -> bool:
    print(f"--- {name} ---")
    started = time.monotonic()
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": PROMPT}],
            temperature=float(os.environ.get("TEMPERATURE", "1.0")),
            top_p=float(os.environ.get("TOP_P", "1.0")),
            **({"extra_body": extra_body} if extra_body else {}),
        )
    except Exception as exc:  # surface the server's error, keep going
        print(f"  FAILED: {exc}\n")
        return False

    elapsed = time.monotonic() - started
    message = resp.choices[0].message

    reasoning = getattr(message, "reasoning_content", None)
    if reasoning:
        print(f"  reasoning: {len(reasoning)} chars")

    print(f"  answer:    {(message.content or '').strip()!r}")
    if resp.usage:
        print(f"  tokens:    {resp.usage.completion_tokens} out in {elapsed:.1f}s")
    print()
    return True


def main() -> int:
    # Before the parser, so the settings layers are in place for the defaults below.
    rest = load_config()

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Every setting is also an argument: MODEL_ID=..., BASE_URL=..., TEMPERATURE=...",
    )
    parser.add_argument("--all", action="store_true", help="include Think Max")
    args = parser.parse_args(rest)

    require("MODEL_ID")
    model = os.environ["MODEL_ID"]
    url = base_url()

    client = OpenAI(base_url=url, api_key=os.environ.get("API_KEY") or "EMPTY")
    print(f"Server: {url}\nModel:  {model}  (profile {os.environ.get('PROFILE', '?')})\n")

    setting = "SMOKE_MODES_ALL" if args.all else "SMOKE_MODES"
    wanted = [name.strip() for name in os.environ.get(setting, "plain").split(",") if name.strip()]
    unknown = [name for name in wanted if name not in MODES]
    if unknown:
        print(f"error: {setting} names unknown mode(s): {', '.join(unknown)}", file=sys.stderr)
        print(f"Known modes: {', '.join(MODES)}", file=sys.stderr)
        return 2

    results = [run_mode(client, model, name, MODES[name]) for name in wanted]

    failed = results.count(False)
    print(f"{len(results) - failed}/{len(results)} modes OK")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
