# segvec

Content-aware image vectorising. Instead of tracing colours or edges, `segvec` first works out **what is in the
picture** with Meta's Segment Anything models, then writes **one clean, flat-colour SVG path per object**: a red
jumper is one shape, a window is one shape, a roof is one shape.

It is aimed at turning photos into simplified, editable illustrations (Inkscape, Affinity, Illustrator): low
detail, key features, and shapes you can recolour or redraw by hand.

- **Automatic mode**: SAM 2.1 finds the objects. A greedy "does this shape reduce the error?" filter keeps only
  the ones that matter, and a second pass fills what is left.
- **Described mode**: tell it what matters (`--care "window, red shutters, dog"`). SAM 3 finds every instance of
  each phrase and those shapes are always kept, drawn on top, and **named after their phrase** in the SVG
  (`window-12`, `red-shutters-3`), so you can select or delete all the windows at once.
- **Sharp where it should be**: the tracer keeps real corners sharp (windows stay rectangular) and smooths only
  where the outline actually curves.
- **Runs locally**, including on the Apple GPU (MPS). No cloud service.

The method follows [SAMVG](https://arxiv.org/abs/2311.05276) (Zhu et al., ICASSP 2024), minus its
differentiable-rendering optimisation step: colours are simply the mean of each shape's visible pixels.
No SAMVG code has been released, so this is a re-implementation from the paper.

## Install

Requires Python 3.13 and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/barton-muller/segvec
cd segvec
uv sync
```

Model weights download from Hugging Face on first use: SAM 2.1 large is about 0.9 GB.

### SAM 3 (for `--care`)

The SAM 3 weights (about 3.5 GB) are **gated**. Request access on the
[facebook/sam3](https://huggingface.co/facebook/sam3) model page, accept Meta's licence, then log in once:

```bash
uv run hf auth login
```

Automatic mode does not need SAM 3.

## Use

```bash
# automatic
uv run segvec trace photo.jpg -o out/

# plus the things you care about
uv run segvec trace photo.jpg -o out/ --care "person, face, hair, jumper, jeans, window, roof"
```

`uv run segvec trace --help` lists every option. The ones you will touch most:

| Option | Effect |
|---|---|
| `--care "a, b, c"` | Describe what matters; one SAM 3 prompt per comma-separated phrase |
| `--care-threshold 0.4` | Lower finds more and fainter instances (default 0.5) |
| `--impact 5e-5` | Lower keeps more automatic shapes (default 2e-4) |
| `--grid 64` | Denser SAM 2.1 prompt grid: finds more, slower (default 32) |
| `--max-side 1280` | Resolution the image is processed at (default 1024) |
| `--round-px 0` | Mask smoothing before tracing; 0 gives the crispest shapes |
| `--tone-rms 45` | Split shapes with strong internal colour contrast into tone patches |

### What you get

Everything lands in the output folder:

| File | What |
|---|---|
| `vector.svg` | The result: stacked shapes, bottom first, one path each |
| `vector.png` | The same, rendered at 3x for quick looking |
| `segments.svg` / `.png` | **The segment map**: every shape in its own arbitrary colour, described shapes outlined in magenta. Use it to tell segmentation problems from colour problems |
| `compare.png` | Source, segments and vector side by side |
| `masks.npz` | The masks and their phrases, for `rerender` |
| `raw_masks.npz` | Cached SAM 2.1 masks: rerunning with different `--care`/`--impact` skips the slow stage |
| `run.json` | The settings used and, per phrase, how many instances SAM 3 found |

Colours are intentionally plain averages; recolour in your vector editor.

### Re-render without a model

```bash
uv run segvec rerender photo.jpg out/masks.npz --round-px 0
```

### See many runs at once

Lay runs out as `outputs/<image>/<variant>/` and build a page that shows them all:

```bash
uv run segvec index --folder outputs
```

## How it works

1. **SAM 2.1 automatic masks** from a point grid plus zoomed crops.
2. **Filter by impact**: masks are painted large to small in their mean colour; one is kept only if it lowers the
   error against the photo by at least `--impact`. Redundant and unimportant masks drop out.
3. **Fill the gaps**: big badly covered regions are found with a distance transform and SAM 2.1 is prompted at
   their centres; the same filter runs again.
4. **Describe** (optional): SAM 3 text prompts. These masks replace near-duplicate automatic shapes and sit in
   area order, so smaller automatic details inside them stay on top.
5. **Trace and stack**: every mask becomes one path (corner-preserving Beziers), each filled with the mean colour
   of its visible pixels.

## Limitations

- SAM 3 finds what you name. Anything you forget to mention, and that SAM 2.1 misses, is missing.
- Fine texture (fur, foliage, foreground grass) is flattened to a few blobs. Trace those by hand on top.
- Colours are averages, so a black-and-white animal comes out grey.
- Described shapes are as good as SAM 3's masks, which are low-resolution and can be blobby at the edges.
- The automatic stage takes about a minute or two per image on an Apple M-series GPU.

## Development

```bash
uv run pytest        # fast tests, no models needed
uv run ruff check .
```

## Credits and licences

- [SAM 2.1](https://github.com/facebookresearch/sam2) and [SAM 3](https://github.com/facebookresearch/sam3) by
  Meta, used through Hugging Face `transformers`. Their weights are under Meta's licences; check them before
  commercial use.
- [SAMVG](https://arxiv.org/abs/2311.05276), the method this follows.
- SVG-to-PNG rendering by [resvg](https://github.com/linebender/resvg).

segvec itself is MIT licensed (see `LICENSE`).
