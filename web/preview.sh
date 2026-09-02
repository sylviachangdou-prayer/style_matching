#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PYTHONS=()
if [[ -n "${VIRTUAL_ENV:-}" ]]; then
  PYTHONS+=("$VIRTUAL_ENV/bin/python")
fi
PYTHONS+=("/opt/anaconda3/bin/python" "python3")

for candidate in "${PYTHONS[@]}"; do
  if [[ -x "$candidate" ]] && "$candidate" -c "import fastapi, uvicorn, yaml" >/dev/null 2>&1; then
    export STYLEMATCH_INDEX_DIR="${STYLEMATCH_INDEX_DIR:-$ROOT/artifacts/multilingual_style_index_gutenberg_v3}"
    echo "StyleMatch preview: http://127.0.0.1:8000"
    exec "$candidate" -m uvicorn web.api.main:app --port 8000
  fi
done

echo "No local Python with FastAPI and Uvicorn was found."
echo "Install only the preview packages, then rerun: python3 -m pip install fastapi uvicorn pyyaml"
exit 1
