# Usage

```bash
# automatic
uv run tracesmart trace photo.jpg -o out/

# plus the things you care about
uv run tracesmart trace photo.jpg -o out/ --care "person, face, hair, jumper, jeans, window, roof"
```

`uv run tracesmart trace --help` lists every option. The ones you will touch most:

| Option | Effect |
|---|---|
| `--care "a, b, c"` | Describe what matters; one SAM 3 prompt per comma-separated phrase |
| `--care-threshold 0.4` | Lower finds more and fainter instances (default 0.5) |
| `--impact 5e-5` | Lower keeps more automatic shapes (default 2e-4) |
| `--grid 64` | Denser SAM 2.1 prompt grid: finds more, slower (default 32) |
| `--max-side 1280` | Resolution the image is processed at (default 1024) |
| `--round-px 0` | Mask smoothing before tracing; 0 gives the crispest shapes |
| `--tone-rms 45` | Split shapes with strong internal colour contrast into tone patches |

## Faces

Full description, examples and limits: [FACES.md](FACES.md).

SAM cannot resolve an eye a few pixels wide, so facial detail is a **second stage** that runs on a finished trace
and draws the parts from where they are. It finds each face with RetinaFace, crops and upscales it, and reads it with
a face-parsing SegFormer (hair, neck, skin, glasses) and `face-alignment` (68 landmarks: eyes, brows, mouth).

```bash
uv sync --extra faces      # once: installs face-alignment (it downloads its weights on first use)
uv run tracesmart trace photo.jpg -o out/ --max-side 1280
uv run --extra faces tracesmart faces photo.jpg out/masks.npz --max-side 1280      # writes out/faces/
```

`--style cartoon` (default) draws each eye as a dot (an arc if it is closed), the mouth as a line or a dark open
shape with white teeth, brows as thin strokes, and no nose. Faces of 60 px and up take these from the segmentation. `--style detailed` draws eye whites, irises,
pupils, lids, lips and teeth. Detail falls with face size: under 40 px only hair, skin and brows, under 60 px also
eyes and mouth, and from 60 px the parse gives teeth too. Glasses are drawn when the parser finds them (filled lenses for sunglasses,
rings otherwise). Colours are the photo's, nudged towards what a part must look like (`render.PRIOR_COLOURS`).

The shapes are named `face-<part>-<n>` (`face-eye-81`, `face-hair-60`) and are ordinary described shapes, so `--layers`
treats them like `--care` shapes. Earlier face shapes are replaced, so the stage can be rerun, and your own `--care`
shapes are never touched: face parts go on top of them, and a `hair`, `neck` or `face` shape you described is used
instead of the parsed one. It works best on frontal
faces; a face whose landmarks fall off the skin (turned head) gets only hair, neck and skin.

## What you get

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

## Re-render without a model

```bash
uv run tracesmart rerender photo.jpg out/masks.npz --round-px 0
```

## See many runs at once

Lay runs out as `outputs/<image>/<variant>/` and build a page that shows them all:

```bash
uv run tracesmart index --folder outputs
```

## Layers for editing

By default the SVG is a flat stack of shapes (bottom first). `--layers` organises it so it is easier to edit, on both
`trace` and `rerender`:

```bash
uv run tracesmart rerender photo.jpg out/masks.npz --layers objects
uv run tracesmart rerender photo.jpg out/masks.npz --layers levels
uv run tracesmart rerender photo.jpg out/masks.npz --layers depth
```

| `--layers` | Result |
|---|---|
| `none` (default) | A flat stack of shapes |
| `objects` | Each object becomes a group. With `--group person` (the default) every person's shirt, shorts, boots, hat, backpack and pole are nested in that person's group (`person-18-group`); other parts nest in the shape that contains them. Select the group to move or recolour the whole person. The shapes below it (ground, forest) are extended under it, so lifting or moving the group leaves no hole; `--no-complete` turns that off |
| `levels` | Three Inkscape layers from coarse to fine: `1 Structure`, `2 Objects`, `3 Details`. Hide the last to simplify the picture |
| `depth` | Inkscape layers from back to front (8 on the hikers). Shapes in one layer never overlap each other, so every layer is a clean cut-out. This is the layering of *Layered Image Vectorization via Semantic Simplification* (Wang et al., CVPR 2025), derived here from the painter's stack |

A shape's level comes from its *impact*: how much it reduces the error against the photo when it is painted, the same
measure SAMVG uses to decide which masks matter.

Both modes keep every pair of noticeably overlapping shapes in its original order, so the picture is the same, with
one tolerance: overlaps smaller than about 12% of the smaller shape (such as the 2 px strips where neighbours meet)
are ignored, so along shared edges the two can differ by a pixel or two (about 0.1 to 0.5% of the pixels on the
hikers). `benchmarks/layers_figure.py` makes the figure in the README.

### How objects are grouped and completed

![A hiker lifted out as one group and moved aside](move.jpg)

*Panel 2: lifted out without completion leaves a hole, the flat backdrop colour. Panel 3: with completion the ground
and forest continue behind. Panel 4: moved aside.*

- **Grouping** (`--group`, default `person`). The described shapes whose phrase matches are the anchors. Another
  shape joins an anchor if at least half of it lies inside the anchor, or at least 30% of its outline borders the
  anchor (a backpack, which SAM does not count as part of the person), or, if it is small (at most 10% of the anchor),
  at least 70% of it lies in the anchor's convex hull (a hiking pole). A shape never joins an anchor it is bigger than
  60% of, so ground and trees are never taken. Use `--group "person, dog"` for more kinds of object.
- **Completion** (`--complete`, default on). Under an object's silhouette, any pixel that no lower shape covers is
  given to the nearest lower shape. That area is hidden by the object, so the picture does not change; it only means
  the ground has no notch.
- **Limits.** The edge between, say, ground and forest behind a lifted object is the halfway line between them, not
  the true horizon, and a faint seam can show where two lower shapes meet. Things the rules miss stay behind when the
  group moves (a stray hand, a part with little contact). Where another object overlaps, what is behind that one is
  not completed.
