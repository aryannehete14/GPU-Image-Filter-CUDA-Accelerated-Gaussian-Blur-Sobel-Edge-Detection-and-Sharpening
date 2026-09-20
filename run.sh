#!/usr/bin/env bash
# run.sh
# Builds gpu_image_filter and runs it across every image in data/ with
# every supported operation, producing proof-of-execution artifacts
# (output images + a CSV timing log) in output/.
#
# Usage:
#   ./run.sh
#
# Prerequisites:
#   - A CUDA toolkit (nvcc) and an NVIDIA GPU.
#   - One or more binary PPM (P6) images placed in data/.
#     PNG/JPEG can be converted with ImageMagick, e.g.:
#       for f in data/*.png; do convert "$f" "${f%.png}.ppm"; done

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

DATA_DIR="data"
OUT_DIR="output"
LOG_FILE="$OUT_DIR/log.csv"

echo "=== Building gpu_image_filter ==="
make clean
make

mkdir -p "$OUT_DIR"
rm -f "$LOG_FILE"

shopt -s nullglob
images=("$DATA_DIR"/*.ppm)
shopt -u nullglob

if [ ${#images[@]} -eq 0 ]; then
  echo "No .ppm images found in $DATA_DIR/."
  echo "Add sample images (see data/README.md for sources), or convert"
  echo "PNG/JPEG files to PPM with ImageMagick, then re-run this script."
  exit 1
fi

echo "=== Running on ${#images[@]} image(s) ==="
for img in "${images[@]}"; do
  base="$(basename "${img%.*}")"
  for op in blur sobel sharpen; do
    out="$OUT_DIR/${base}_${op}.ppm"
    echo "--- $img -> $out (op=$op) ---"
    ./gpu_image_filter --input "$img" --output "$out" --op "$op" \
      --sigma 2.5 --log "$LOG_FILE"
  done
done

echo "=== Done. Proof-of-execution artifacts: ==="
echo "  Output images: $OUT_DIR/*.ppm"
echo "  Timing log:    $LOG_FILE"
cat "$LOG_FILE"
