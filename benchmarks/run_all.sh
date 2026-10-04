#!/usr/bin/env bash
# Rebuild everything derived from outputs/: re-render the saved masks, run the comparison against vtracer and
# SuperSVG, and regenerate examples/, the README figures and RESULTS.md. Takes about 10 minutes per photo, mostly
# the vtracer tuning. Run from the repository root. No SAM model is used: it works from the saved masks.npz files.
#
#   benchmarks/run_all.sh rerender   # re-render every outputs/*/{auto,care}/masks.npz (seconds each)
#   benchmarks/run_all.sh bench      # SuperSVG + vtracer comparison for the six examples (long)
#   benchmarks/run_all.sh assets     # RESULTS.md, examples/, close-ups, pipeline and layers figures
#
# Environment: SUPERSVG_DIR (default ~/dev/vector-bench/SuperSVG, see benchmarks/SUPERSVG.md); the VTracer 1.0 binary
# is found on PATH or at ~/.cargo/bin/vtracer (cargo install vtracer-cli --version 1.0.0-alpha.4).
set -euo pipefail
cd "$(dirname "$0")/.."
EXAMPLES=(oostpoort hikers delft-street mountain-lake lone-hiker canal)
SUPERSVG_DIR="${SUPERSVG_DIR:-$HOME/dev/vector-bench/SuperSVG}"
export SUPERSVG_DIR

case "${1:-}" in
  rerender)
    for d in outputs/*/auto outputs/*/care; do
      [ -f "$d/masks.npz" ] && uv run tracesmart rerender "$d/source.png" "$d/masks.npz" --max-side 5000 >/dev/null && echo "redrawn $d"
    done
    uv run tracesmart index --folder outputs ;;
  bench)
    for n in "${EXAMPLES[@]}"; do
      paths=$(grep -o "<path" "outputs/$n/care/vector.svg" | wc -l | tr -d ' ')
      echo "== $n ($paths paths)"
      mkdir -p "outputs/$n/methods"
      rm -f "outputs/$n/methods/vtracer_matched.svg" outputs/$n/methods/vtracer1_*.svg   # re-tune to the current path count
      # call the SuperSVG venv's python directly: activating it from a bash script made DiffVG segfault
      "$SUPERSVG_DIR/.venv/bin/python" benchmarks/run_supersvg.py "outputs/$n/care/source.png" "$paths" "outputs/$n/methods/supersvg.svg"
      uv run --group bench python benchmarks/compare.py "$n" --care "outputs/$n/care" --auto "outputs/$n/auto" \
        --extra "SuperSVG (CVPR 2024)=outputs/$n/methods/supersvg.svg"
    done ;;
  assets)
    uv run python benchmarks/summarise.py outputs > benchmarks/RESULTS.md
    uv run --group bench python benchmarks/make_examples.py
    uv run --group bench python benchmarks/closeup.py hikers 250,540,650,900 examples/hikers/closeup.jpg
    uv run --group bench python benchmarks/closeup.py delft-street 330,230,730,520 examples/delft-street/closeup.jpg
    uv run --group bench python benchmarks/pipeline_figure.py hikers docs/pipeline.jpg
    uv run --group bench python benchmarks/layers_figure.py hikers docs/layers.jpg
    tail -9 benchmarks/RESULTS.md ;;
  *) sed -n '2,12p' "$0"; exit 1 ;;
esac
