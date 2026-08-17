"""Hit a running DeepSeek-V4-Pro server across its reasoning modes.

The V4 chat template exposes reasoning effort through chat_template_kwargs rather
than a top-level parameter, so each mode is a different extra_body.

    python smoke_test.py                              # non-think + high
    python smoke_test.py --all                        # also Think Max
    python smoke_test.py BASE_URL=http://gpu-01:8000/v1
    python smoke_test.py MODEL_ID=Qwen/Qwen3-8B

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
    print(f"Server: {url}\nModel:  {model}\n")

    modes = MODES if args.all else {k: v for k, v in MODES.items() if k != "max"}
    results = [run_mode(client, model, name, body) for name, body in modes.items()]

    failed = results.count(False)
    print(f"{len(results) - failed}/{len(results)} modes OK")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
