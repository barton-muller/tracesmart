# Development notes and handoff

For anyone (human or agent) picking this project up. Written 2026-10-04. Agent-oriented rules are in
[../CLAUDE.md](../CLAUDE.md); papers and notes on related work are in [../references/](../references/README.md).

## Where things stand

- Public repo: <https://github.com/barton-muller/tracesmart>, branch `main`. Python 3.13, uv, MIT.
- Works: `trace` (automatic and `--care`), `rerender`, `index`; layering with `--layers objects|levels|depth`;
  sharp-corner tracing; seam closing; gap fill; 19 fast tests; ruff clean.
- Measured on one machine only (Apple M2, 16 GB): default run about 4 min, detailed run about 7 min, `--care` about
  1 min for 3 phrases with cached masks.
- Everything, including layering, `references/`, the VTracer 1.0 benchmark and these docs, is committed and pushed.
  Run `git status` anyway.

## Next steps, in rough order

1. **Try the layered SVGs in Inkscape**, especially `--layers objects` (hiker groups plus completed ground; the
   `move_figure.py` panels show what to expect). They were only checked by rendering with resvg and by tests; nobody has
   opened them in an editor yet. Decide the default for `--layers` (currently `none`).
2. **Multi-level SAM** (cheap idea from the Wang et al. paper, see `references/notes/semantic-simplification.md`): run SAM
   on progressively simplified copies of the photo and pool the masks before "filter by impact".
3. **Click-to-fix interface** (planned early on): SAM 2.1 point prompts to add, carve, merge and delete shapes,
   with live SVG preview. The image is encoded once (`backends.Prompter`), each click is cheap.
4. Quality items: faces and fine texture, the dog (its SAM 3 mask is mostly hidden by smaller shapes), foreground
   grass. Mean colours wash out saturation; a better colour choice is possible (it is easy to fix in an editor).
5. Optional: DiffVG refinement of shapes and colours (SAMVG stage 3), PyPI release (the name `tracesmart` was free
   on 2026-10-02).

## Faces (`tracesmart faces`, `src/tracesmart/faces.py`)

Added 2026-10-04, uncommitted at the time of writing. A second stage over a finished `masks.npz`; optional extra
`faces` (`face-alignment`). Usage is in `docs/USAGE.md`.

- Pipeline: RetinaFace boxes (via face-alignment; SAM 3 "face" gave a spurious tiny box, BlazeFace missed faces) ->
  crop with margin -> SegFormer face parsing (hair, neck, skin, glasses) and FAN landmarks (eyes, brows, mouth).
  Hair/neck/skin are inserted by area like `--care` shapes (skin above its hair); parts go on top of everything.
- Drawn from landmarks, not traced: eye dot at the darkest spot of the eye (arc if closed), mouth line or dark open
  shape, teeth = pale unsaturated pixels of the open mouth, brows = darkest half of the dropped landmark band
  (`BROW_DROP`: FAN traces the brow's top edge), glasses = parsed region (sunglasses) or a ring per eye from the
  parsed region's extent. `render.PRIOR_COLOURS` nudges a part's sampled mean towards its known colour (teeth white,
  pupils dark), keyed on the phrase, so `rerender` keeps it.
- Detail tiers by face size (`SMALL_FACE`, `FULL_FACE`); no nose shape, ever (the user's call). A face whose landmarks
  do not fall on the parsed skin (turned head) gets only hair, neck and skin.
- Tried and failed: face parsing alone (eyes are specks); plain-mean colours over landmark outlines (eyes and brows
  came out skin-coloured); MediaPipe Face Landmarker (478 points, but the Python 3.13 macOS wheel aborts with "Service
  is unavailable" in its Metal helper, in the sandbox and in the user's terminal; `mediapipe-silicon` is 0.9.3 for
  older Pythons with `protobuf<4`, not tried).
- Ideas: Sapiens2 segmentation (`facebook/sapiens2-seg-0.8b`, 29 classes including teeth, lips, tongue, eyeglasses,
  loads in transformers; multi-GB, custom licence) would replace the colour heuristics for teeth and glasses.
- Test photos: family photo (60 px faces), and two Unsplash group photos in `tests/images/joel.jpg`, `tim.jpg`
  (gitignored; credits: Joel Muniz, Tim Mossholder).
- `uv sync --extra faces` also installs `opencv-python` and `opencv-contrib-python` next to the project's
  `opencv-python-headless`. They share the `cv2` module and work, but it is untidy; worth pinning.

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

Per photo in `benchmarks/RESULTS.md`; table in the README. At about 130 paths tracesmart (--care) scores PSNR 17.0 /
SSIM 0.41; VTracer 1.0 watershed 18.1 / 0.44 (188 KB); VTracer 1.0 colour 18.5 / 0.51 (626 KB); SuperSVG 19.1 / 0.44
(70 KB); VTracer 0.6 tuned 15.3 / 0.47 (1.5 MB). tracesmart's files are about 64 KB. Pixel scores favour the others
by design (tracesmart uses one flat colour per shape); visually it is the only one whose shapes are objects.

## Decisions and dead ends (so they are not repeated)

- Object grouping (`layers.object_groups`) is geometric: SAM 3's "person" and "hiker" masks exclude backpacks, and
  longer prompts did not fix that, so membership uses inside-share, outline contact and (for thin things) hull share.
  `layers.complete_under` fills the notch an object leaves in the shapes below by nearest owner. The generated
  gap-fill shapes (first `n_gaps` masks) must never be group members or owners; letting one join a group made the
  completion silently do nothing.

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
