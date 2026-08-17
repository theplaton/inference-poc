#!/usr/bin/env bash
# Root entrypoint. The implementation, its settings and the rest of the serving
# tools live in model_serving/ -- see model_serving/README.md.
set -euo pipefail
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/model_serving/serve_model.sh" "$@"
