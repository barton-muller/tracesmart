# tracesmart

Image tracing that understands the picture. A photo is segmented with Meta's Segment Anything models (SAM 2.1 for
everything, SAM 3 for things the user names with `--care`) and written as **one flat-colour SVG path per object**.
The method is a re-implementation of SAMVG (arXiv:2311.05276) from the paper text, because no code was released. Keep
that credit prominent in the README.

The goal is a simplified, editable illustration (Inkscape, Affinity, Illustrator), not a pixel-faithful trace. Low
PSNR/SSIM against the photo is expected; do not "fix" it by adding detail or colour variation. Colours are plain
means on purpose.

## Commands

Python 3.13, managed with **uv**. Always `uv run ...` / `uv add ...`; never pip.

```bash
uv run tracesmart trace photo.jpg -o out/ [--care "window, roof"]   # the main command
uv run tracesmart rerender photo.jpg out/masks.npz                  # redraw from saved masks, no model
uv run tracesmart index --folder outputs                            # HTML page over outputs/<image>/<variant>/
uv run pytest -q                                                    # fast, no models needed
uv run ruff check .                                                 # line length 120
uv run --group bench python benchmarks/compare.py --help            # vs vtracer / SuperSVG
```

## Layout

| Path | What |
|---|---|
| `src/tracesmart/backends.py` | SAM 2.1 (auto masks, point prompts) and SAM 3 (text); device selection |
| `src/tracesmart/pipeline.py` | `vectorise()`: raw masks, "filter by impact", gap prompting, `--care` descriptions |
| `src/tracesmart/trace.py` | mask to path: corner-preserving Beziers, mask smoothing |
| `src/tracesmart/render.py` | stack masks to SVG, segment map, PNGs, HTML index; shape naming |
| `src/tracesmart/tones.py` | optional colour splitting (`--tone-rms`) |
| `src/tracesmart/cli.py` | typer CLI: `trace`, `rerender`, `index` |
| `tests/` | synthetic-image tests only; no models, no network |
| `benchmarks/` | comparison scripts, `RESULTS.md`, `SUPERSVG.md` (running SuperSVG on a Mac) |
| `examples/` | curated, downscaled results, committed. Regenerate with `benchmarks/make_examples.py` |
| `docs/` | `USAGE.md`, `COMPARISON.md`, `CREDITS.md` and `pipeline.jpg` (made by `benchmarks/pipeline_figure.py`); keep the README short and link here |
| `outputs/`, `tests/images/` | **gitignored**: local runs and the user's own photos |

## Conventions and decisions

- British English in prose and identifiers (colour, grey).
- A run folder holds `vector.svg`, `segments.svg/.png` (every shape in its own colour, described shapes outlined in
  magenta, to tell segmentation problems from colour problems), `vector.png`, `compare.png`, `masks.npz`,
  `raw_masks.npz` (cached slow stage), `run.json`.
- `masks.npz` stores `masks` and `phrases`; described shapes are named after their phrase in the SVG (`window-12`) via
  `render.shape_name`. Anything that reorders or adds masks must keep `phrases` aligned.
- Painter's order: masks are stacked bottom first; a shape's colour is the mean of its *visible* pixels. Areas no mask
  covers become extra bottom shapes at render time.
- Tracing keeps corners where the outline turns more than 35 degrees (windows stay rectangular); do not replace it
  with plain Catmull-Rom smoothing.
- Don't add features the README doesn't need. Failed experiments were removed on purpose (tiled colour regions,
  per-object tone splitting of the dog): the dog's SAM 3 mask is mostly hidden by smaller shapes, so there was
  nothing left to split.

## Gotchas

- **SAM 3 is gated** on Hugging Face. The user must accept the licence and run `uv run hf auth login`. Never ask for,
  print or store a token.
- Apple GPU (MPS): set `PYTORCH_ENABLE_MPS_FALLBACK=1` for long runs; a few ops fall back to the CPU. `--care` retries
  on the CPU if SAM 3 fails on the device.
- transformers' `mask-generation` pipeline crashes with its own crop mode on some image sizes, so zoomed crops are
  done by hand in `backends.auto_masks`.
- The shell is **zsh**: unquoted `$var` is not word-split, and BSD `sed -i ''` needs the empty argument.
- Running SuperSVG: call the SuperSVG venv's `python` directly. Activating it from a bash script made DiffVG
  segfault. See `benchmarks/SUPERSVG.md`.
- A full run takes minutes on an M-series GPU. Run long jobs in the background and poll; `raw_masks.npz` caches the
  slow stage so changing `--care`/`--impact` only costs the later stages.

## Privacy and licensing

- Never commit or publish the user's own photos or anything generated from them (`tests/images/`, `outputs/`).
  `examples/` uses Unsplash photos only, with credits in `examples/README.md`.
- The package metadata lists the author's name only; don't add an email address.
- MIT licensed. SAM 2.1 / SAM 3 weights are under Meta's licences.
