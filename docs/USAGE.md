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
