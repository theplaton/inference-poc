#!/usr/bin/env bash
# Root entrypoint. The implementation, its settings and the rest of the
# client-side evaluation live in benchmark/ -- see benchmark/README.md.
set -euo pipefail
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/benchmark/benchmark.sh" "$@"
