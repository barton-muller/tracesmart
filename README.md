# tracesmart

**Image tracing that understands the picture. It segments the photo with Meta's Segment Anything models (SAM 2.1,
and SAM 3 for things you name) and writes one clean, flat-colour SVG shape per object, not thousands of colour
patches.**

![Hikers: the photo, what tracesmart found, and the result](examples/hikers/compare.jpg)

*Left: the photo. Middle: what it found, every shape in its own colour. Right: the result, 127 flat-colour shapes in
a 75 KB SVG. Photo by [Dan Ordze](https://unsplash.com/photos/4GoNeNKEB1M) on Unsplash.*

## Why use it

- **One shape per thing.** A hat, a backpack, a window: shapes you can recolour, move or redraw, not fragments. A
  colour-clustering tracer gives about 14,000 patches and 15 MB for the same kind of photo.
- **Tell it what matters.** `--care "window, red shutters, dog"` finds every instance by name, always keeps it, and
  names the shapes (`window-12`) so you can select them all in your editor.
- **Light and editable.** Typically 50 to 100 KB and 70 to 250 paths.
- **Local.** Runs on your machine, including the Apple GPU. Nothing is uploaded.

**The goal** is a simplified, editable illustration: low on detail, key features kept, flat colours you finish in
Inkscape, Affinity or Illustrator. It is not a pixel-faithful trace. What is still too fine for any segmenter (fur,
foreground grass, faces) you trace by hand on top, in the same file.

## Quick start

Needs Python 3.13 and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/barton-muller/tracesmart && cd tracesmart && uv sync
uv run tracesmart trace photo.jpg -o out/                                  # automatic
uv run tracesmart trace photo.jpg -o out/ --care "person, hat, backpack"   # plus things you name
```

Open `out/vector.svg`. `out/segments.png` shows what was found (use it to tell segmentation problems from colour
problems) and `out/compare.png` puts the photo, segments and result side by side. All options and files:
[docs/USAGE.md](docs/USAGE.md).

`--care` uses SAM 3, whose weights are gated: request access on the
[facebook/sam3](https://huggingface.co/facebook/sam3) page, accept Meta's licence, then run
`uv run hf auth login` once. Automatic mode does not need it.

## How it works

![Pipeline: photo, SAM masks, filtered shapes, described shapes, result](docs/pipeline.jpg)

1. **SAM 2.1 proposes objects** across the image (a grid of point prompts plus zoomed crops): 202 masks here.
2. **Filter by impact** keeps only masks that earn their place. They are painted large to small in their mean
   colour, and each must measurably reduce the error against the photo. That leaves 139 shapes here.
3. **Gaps are filled**: big badly covered areas are found and SAM 2.1 is prompted there, then filtered again.
4. **`--care` (optional)**: SAM 3 finds every instance of each phrase. Those shapes are always kept, replace
   near-duplicate automatic ones, and are outlined in magenta in the segment map.
5. **Tracing**: each mask becomes one path. Real corners stay sharp (windows stay rectangular) and curves are
   smoothed. Each shape gets the mean colour of its visible pixels.

## Hardware and speed

Developed and measured on **one machine: an Apple M2 with 16 GB of RAM**.

| | |
|---|---|
| Downloads | SAM 2.1 large about 0.9 GB; SAM 3 about 3.3 GB (only for `--care`) |
| Default run (1024 px, `--grid 32`) | about **4 minutes**, about 1 GB of memory |
| Detailed run (1280 px, `--grid 48`, 3 rounds; the examples' settings) | about **7 minutes**, about 1.5 GB |
| Adding `--care`, masks already cached | about **1 minute** for 3 phrases, a few seconds per extra phrase |
| `rerender`, `index` | seconds, no model |

- The slow part is SAM 2.1 on the whole image. It is cached in `raw_masks.npz`, so changing `--care`, `--impact`
  or the rendering afterwards does not repeat it.
- The Apple GPU (MPS) is used automatically. CUDA should be picked up by PyTorch but is **untested**. The CPU
  should work but is **untested and expected to be several times slower**.
- Output detail is limited by `--max-side` (default 1024) and by SAM's own mask resolution.

## Examples

Six Unsplash photos, each traced automatically and with `--care` (click through for every panel, the SVGs and the
scores):

<table>
<tr>
<td align="center"><a href="examples/hikers/"><img src="examples/hikers/panels/tracesmart-care.jpg" width="260"><br>hikers</a></td>
<td align="center"><a href="examples/delft-street/"><img src="examples/delft-street/panels/tracesmart-care.jpg" width="260"><br>Delft street</a></td>
<td align="center"><a href="examples/oostpoort/"><img src="examples/oostpoort/panels/tracesmart-care.jpg" width="260"><br>Oostpoort</a></td>
</tr>
<tr>
<td align="center"><a href="examples/mountain-lake/"><img src="examples/mountain-lake/panels/tracesmart-care.jpg" width="260"><br>mountain lake</a></td>
<td align="center"><a href="examples/lone-hiker/"><img src="examples/lone-hiker/panels/tracesmart-care.jpg" width="260"><br>lone hiker</a></td>
<td align="center"><a href="examples/canal/"><img src="examples/canal/panels/tracesmart-care.jpg" width="260"><br>canal</a></td>
</tr>
</table>

## How it compares

![The hikers traced by vtracer, SuperSVG and tracesmart](examples/hikers/methods.jpg)

*Photo, vtracer (defaults), vtracer (tuned to about 110 paths), SuperSVG (CVPR 2024, about 110 paths), tracesmart
(automatic), tracesmart (`--care`).*

Averages over the six photos:

| Method | Paths | File size | PSNR | SSIM |
|---|---|---|---|---|
| vtracer, defaults | 14,074 | 15.4 MB | 22.8 | 0.76 |
| vtracer, tuned to about 115 paths | 116 | 1.6 MB | 17.2 | 0.50 |
| SuperSVG (CVPR 2024) | 115 | 63 KB | 18.8 | 0.43 |
| tracesmart, automatic | 107 | 62 KB | 16.2 | 0.41 |
| tracesmart, `--care` | 128 | 64 KB | 17.0 | 0.41 |

- **By pixel scores tracesmart comes last** at a similar path count. PSNR and SSIM reward copying the photo;
  tracesmart flattens every shape to one colour on purpose.
- vtracer's defaults look exactly like the photo because they *are* thousands of tiny colour fragments, not
  objects. Tuned down, it turns speckled; SuperSVG gives a painterly blur in which objects run together.
- What the scores miss is whether a shape is an *object*. That is the point of tracesmart and no score here
  measures it, so judge it from the images.

Close-ups with every shape outlined, the full gallery, caveats and how to reproduce:
[docs/COMPARISON.md](docs/COMPARISON.md).

## Limitations

- SAM 3 finds what you name. Anything you forget to mention that SAM 2.1 also misses is missing.
- Fine texture (fur, foliage, foreground grass) is flattened to a few blobs. Trace those by hand on top.
- Colours are averages, so a black-and-white animal comes out grey. Recolour in your vector editor.
- Described shapes are only as good as SAM 3's masks, which are low-resolution and can be blobby at the edges.

## Related work

This is not a new idea. Turning Segment Anything masks into vector shapes is the SAMVG paper (2023), and others
have built on it. Read from their READMEs and abstracts, **not run** by me except where noted:

- **[KU-MIIL/semantic-svg-generation](https://github.com/KU-MIIL/semantic-svg-generation)** (paper:
  [*Compositional SVG Generation via VLM-Driven Hierarchical Semantic Parsing*](https://arxiv.org/abs/2609.14657))
  is the closest in spirit: a vision-language model (Gemini) names the parts, SAM 3 masks them, vtracer vectorises
  each part and a painter's-algorithm compositor stacks the layers. It is a benchmark and pipeline for icons, emoji
  and illustrations and needs a cloud model. tracesmart targets photos, runs locally, and you supply the names.
- **SAMVG re-implementations** exist as small student repos (for example
  [kevin20010808/MultimediaProcessingTermProject](https://github.com/kevin20010808/MultimediaProcessingTermProject)).
  I did not find a packaged tool.
- **[VTracer](https://github.com/visioncortex/vtracer)** is now at 1.0, with a desktop app and newer modes such as
  seam-free cutout and watershed clustering. My comparison used the `vtracer` 0.6.15 Python package and did not
  test those.
- **Research methods** such as [AmodalSVG](https://arxiv.org/abs/2604.10940),
  [*Controlling Your Image via Simplified Vector Graphics*](https://arxiv.org/abs/2602.14443),
  [*Layered Image Vectorization via Semantic Simplification*](https://arxiv.org/abs/2406.05404), LIVE and
  [SuperSVG](https://github.com/sjtuplayer/SuperSVG) (the only one I ran) are mostly GPU- or diffusion-heavy.
- **Related tools**: [gimpsegany](https://github.com/Shriinivas/gimpsegany) puts SAM masks into GIMP as raster
  layers, [lang-segment-anything](https://github.com/luca-medeiros/lang-segment-anything) gives text-prompted masks
  without vectors, and Vector Magic, Vectorizer.ai and Illustrator's Image Trace are the commercial tracers.

If you know of something closer, please open an issue.

## Built on SAMVG

The method is a re-implementation of **[SAMVG](https://arxiv.org/abs/2311.05276)** (Haokun Zhu, Juang Ian Chong,
Teng Hu, Ran Yi, Yu-Kun Lai and Paul L. Rosin, *SAMVG: A Multi-stage Image Vectorization Model with the
Segment-Anything Model*, ICASSP 2024). The core ideas come from that paper: using Segment Anything masks as the
shapes of the vector image, **"filter by impact"** to decide which masks matter, and prompting the model again at
badly covered regions. No SAMVG code has been released, so tracesmart is written from the paper, not from its code,
and any shortcomings are ours.

What differs: SAM 2.1 instead of the original SAM; no differentiable-rendering optimisation step (each shape is
simply the mean colour of its visible pixels); **text descriptions with SAM 3** (`--care`) with shapes named after
their phrase, which SAMVG does not have; corner-preserving tracing; a fill for uncovered areas; and optional tone
patches.

The BibTeX entry for citing SAMVG is in [docs/CREDITS.md](docs/CREDITS.md).

## Status

This is a **vibe-coded** project: written with an AI coding assistant (Claude Code) while a human steered and
checked the results by eye. There are fast tests for the plumbing, but nothing automatically checks that the
segmentations are *good*. Expect rough edges and check the output before relying on it.

## Development and credits

```bash
uv run pytest        # fast tests, no models needed
uv run ruff check .
```

Contributor and agent notes are in [CLAUDE.md](CLAUDE.md) (also linked as `AGENTS.md`).

[SAM 2.1](https://github.com/facebookresearch/sam2) and [SAM 3](https://github.com/facebookresearch/sam3) by Meta
(weights under Meta's licences; check them before commercial use), used through Hugging Face `transformers`.
[resvg](https://github.com/linebender/resvg) renders the PNGs. The comparisons use
[vtracer](https://github.com/visioncortex/vtracer) and [SuperSVG](https://github.com/sjtuplayer/SuperSVG). Example
photos are from [Unsplash](https://unsplash.com), credited in [examples/README.md](examples/README.md).
tracesmart itself is MIT licensed (see `LICENSE`).
