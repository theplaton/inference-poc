"""Hit the running DeepSeek-V4-Pro server across all three reasoning modes.

The V4 chat template exposes reasoning effort through chat_template_kwargs
rather than a top-level parameter, so each mode is a different extra_body.

    python smoke_test.py             # non-think + high
    python smoke_test.py --all       # also Think Max (needs a big --max-model-len)
"""

import argparse
import os
import sys
import time

from openai import OpenAI

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envfile import load_env  # noqa: E402

# DeepSeek's recommendation for the 0813 release: temperature 1.0 always,
# top_p 0.95 for agentic work and 1.0 otherwise.
TEMPERATURE = 1.0
TOP_P = 1.0

PROMPT = "What is 17*19? Return only the final integer."

MODES = {
    "non-think": None,
    "high": {"chat_template_kwargs": {"thinking": True, "reasoning_effort": "high"}},
    # Think Max truncates unless the server was started with
    # --max-model-len >= 393216, which does not fit on 8x H200 without KV offload.
    "max": {"chat_template_kwargs": {"thinking": True, "reasoning_effort": "max"}},
}


def run_mode(client: OpenAI, model: str, name: str, extra_body: dict | None) -> bool:
    print(f"--- {name} ---")
    started = time.monotonic()
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": PROMPT}],
            temperature=TEMPERATURE,
            top_p=TOP_P,
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
    load_env()  # before the parser, so MODEL_ID can supply the --model default

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all", action="store_true", help="include Think Max")
    parser.add_argument("--base-url", default=os.environ.get("BASE_URL", "http://localhost:8000/v1"))
    parser.add_argument("--model", default=os.environ.get("MODEL_ID"))
    args = parser.parse_args()

    if not args.model:
        parser.error("no model: set MODEL_ID in .env (cp .env.example .env) or pass --model")

    client = OpenAI(base_url=args.base_url, api_key="EMPTY")
    print(f"Server: {args.base_url}\nModel:  {args.model}\n")

    modes = MODES if args.all else {k: v for k, v in MODES.items() if k != "max"}
    results = [run_mode(client, args.model, name, body) for name, body in modes.items()]

    failed = results.count(False)
    print(f"{len(results) - failed}/{len(results)} modes OK")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
