#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
bucket="decision-lab-models"

put() {
  local file="$1" key="$2" type="$3" cache="$4"
  echo "Uploading ${key}"
  npx wrangler r2 object put "${bucket}/${key}" --file "${file}" --content-type "${type}" --cache-control "${cache}" --remote
}

while IFS= read -r -d '' file; do
  key="${file#public/}"
  case "$file" in
    *.json) type="application/json" ;;
    *.onnx|*.weights) type="application/octet-stream" ;;
    *.jinja|*.md) type="text/plain; charset=utf-8" ;;
    *) type="application/octet-stream" ;;
  esac
  put "$file" "$key" "$type" "public, max-age=31536000, immutable"
done < <(find public/model-v3-q4 -type f -print0 | sort -z)

wasm="$(find dist/assets -maxdepth 1 -name 'ort-wasm-simd-threaded.asyncify-*.wasm' -print -quit)"
if [[ -z "$wasm" ]]; then
  echo "Build first: npm run build" >&2
  exit 1
fi
put "$wasm" "${wasm#dist/}" "application/wasm" "public, max-age=31536000, immutable"
