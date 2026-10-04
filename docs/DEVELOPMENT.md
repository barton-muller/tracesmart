# Development notes and handoff

For anyone (human or agent) picking this project up. Written 2026-10-04. Agent-oriented rules are in
[../CLAUDE.md](../CLAUDE.md); papers and notes on related work are in [../references/](../references/README.md).

## Where things stand

- Public repo: <https://github.com/barton-muller/tracesmart>, branch `main`. Python 3.13, uv, MIT.
- Works: `trace` (automatic and `--care`), `rerender`, `index`; layering with `--layers objects|levels|depth`;
  sharp-corner tracing; seam closing; gap fill; 19 fast tests; ruff clean.
- Measured on one machine only (Apple M2, 16 GB): default run about 4 min, detailed run about 7 min, `--care` about
  1 min for 3 phrases with cached masks.
- Everything above, including layering, `references/` and these docs, is committed and pushed. Run `git status` anyway: the
  benchmark below rewrites `benchmarks/RESULTS.md` and `examples/*` (panels, `methods.jpg`) when it finishes, and
  those changes still need committing.
- **A benchmark run may still be going or have died.** It re-tunes vtracer and re-runs SuperSVG for each of the six
  examples at tracesmart's current path counts, then writes `benchmarks/RESULTS.md`. Check
  `ls outputs/*/methods/vtracer1_*.svg` (six of each kind means done) and `ps aux | grep compare.py`. To finish or
  redo it: `benchmarks/run_all.sh bench` then `benchmarks/run_all.sh assets` (about 10 minutes per photo).

## Next steps, in rough order

1. **Finish the benchmark and update the docs.** Add the two VTracer 1.0 rows (watershed, colour) to the tables in
   `README.md` and `docs/COMPARISON.md` (the numbers come from `benchmarks/RESULTS.md`), remove the "1.0 was not
   tested" caveats there and in `README.md` (Related work), mention `vector-objects/levels/depth.svg` in
   `examples/README.md`, then commit and push.
2. **Try the layered SVGs in Inkscape.** They were only checked by rendering with resvg and by tests; nobody has
   opened them in an editor yet. Decide the default for `--layers` (currently `none`).
3. **Multi-level SAM** (cheap idea from the Wang et al. paper, see `references/notes/semantic-simplification.md`): run SAM
   on progressively simplified copies of the photo and pool the masks before "filter by impact".
4. **Click-to-fix interface** (planned early on): SAM 2.1 point prompts to add, carve, merge and delete shapes,
   with live SVG preview. The image is encoded once (`backends.Prompter`), each click is cheap.
5. Quality items: faces and fine texture, the dog (its SAM 3 mask is mostly hidden by smaller shapes), foreground
   grass. Mean colours wash out saturation; a better colour choice is possible (it is easy to fix in an editor).
6. Optional: DiffVG refinement of shapes and colours (SAMVG stage 3), PyPI release (the name `tracesmart` was free
   on 2026-10-02).

## Environment map

| What | Where |
|---|---|
| This repo | `~/dev/tracesmart` (was `~/dev/segvec`, renamed) |
| SuperSVG with its own venv and patches | `~/dev/vector-bench/SuperSVG` (see `benchmarks/SUPERSVG.md`); `SUPERSVG_DIR` points there |
| VTracer 1.0 CLI | `~/.cargo/bin/vtracer` (`cargo install vtracer-cli --version 1.0.0-alpha.4`) |
| Model weights | Hugging Face cache (`facebook/sam2.1-hiera-large` about 0.9 GB, `facebook/sam3` about 3.3 GB). SAM 3 is gated: `hf auth login` was done on the dev machine |
| GitHub | `gh` is logged in as the repo owner |
| Example photos | Unsplash, originals in `~/Downloads/*-unsplash.jpg`; downscaled copies and credits in `examples/` |
| Local-only, gitignored | `outputs/` (all runs, about 1 GB with caches), `tests/images/` (**includes a private family photo, never publish**), `references/papers/*.pdf` |

The six examples were made with `--grid 48 --impact 3e-5 --max-side 1280 --rounds 3 --care-threshold 0.4`, the phrases
are in `examples/<name>/run.json`, and the original photos are listed in `examples/README.md`. The automatic stage is
cached in `outputs/<name>/raw_cache.npz` so a rerun with other phrases costs only the later stages.

## Measured results (six Unsplash photos, averages)

Per photo in `benchmarks/RESULTS.md`. By PSNR and SSIM tracesmart comes last at a similar path count, by design (flat
colour per shape). Visually it is the only one where shapes are objects. vtracer's defaults give about 14,000 paths
and 15 MB; tuned to about 115 paths it is speckled; SuperSVG is painterly and blurs objects together.

## Decisions and dead ends (so they are not repeated)

- Early attempts generated a wall of rocks procedurally (Voronoi, skyline packing, drop-and-settle). The user wanted
  the real stone shapes from a photo, which led to segmentation. That generator lives in a different repository.
- Colouring *regions* with k-means tone patches per SAM region (the first "paint" pipeline) gave noisy results; it
  was replaced by the SAMVG-style stacked masks. `--tone-rms` is what remains and is off by default.
- Splitting the dog into black and white tones failed: its SAM 3 mask is almost entirely covered by smaller shapes, so
  only about 135 pixels were visible. Removed. Hand-tracing difficult items on top is the intended workflow.
- Thin strips of a darker shape between neighbours (sky and mountain) came from masks that do not tile plus smoothing.
  Fixed by `render.close_seams` (grow each shape by up to about 2 px). Exact order-preserving layering was tried and
  rejected: the 2 px strips constrain the order of neighbours and pull most details into the structure layer, so
  `layers.significant` uses a tolerance (overlap below max(60 px, 12% of the smaller shape) is ignored). Layered output
  can therefore differ from the flat one by a pixel or two along shared edges (about 0.1 to 0.5% of pixels).
- transformers' `mask-generation` pipeline crashed with its own crop option on some sizes, so zoomed crops are done
  by hand in `backends.auto_masks`.
- DiffVG on the CPU segfaults under load and when the venv is activated from a bash script; call the venv's python
  directly. `run_supersvg.py` retries.

## Conventions

- Commit messages end with the `Co-Authored-By` trailer for the assistant that wrote them. Do not commit without being
  asked. Never add an email address to the package metadata. Keep prose in British English.
- Long jobs run in the background and are polled; `uv run` re-syncs the environment, so do not change dependencies while
  a background job is using it.
- Regenerate derived assets with `benchmarks/run_all.sh` rather than by hand, so figures and tables stay consistent.
