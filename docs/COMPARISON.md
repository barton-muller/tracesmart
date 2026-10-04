# Examples and comparison

Six Unsplash photos, each traced automatically and with `--care`, and compared with other tools. Every panel of
every example, plus the SVGs and per-photo scores, is in [`examples/`](../examples/). Note that `--care` does not always
add shapes: described shapes also replace near-duplicate automatic ones, so the count can go down as the shapes
get better named.

### Hikers

![Hikers: photo · VTracer 0.6 (defaults) · VTracer 0.6 (tuned) · VTracer 1.0 watershed (tuned) · VTracer 1.0 colour (tuned) · SuperSVG · tracesmart (automatic) · tracesmart (`--care`)](../examples/hikers/methods.jpg)

*photo · VTracer 0.6 (defaults) · VTracer 0.6 (tuned) · VTracer 1.0 watershed (tuned) · VTracer 1.0 colour (tuned) · SuperSVG · tracesmart (automatic) · tracesmart (`--care`). Photo by [Dan Ordze](https://unsplash.com/photos/4GoNeNKEB1M) on Unsplash. `--care`: person, hat, backpack, shorts, shirt, boot, hiking pole, tree, mountain, sky, gravel path. All panels, SVGs and metrics: [`examples/hikers/`](../examples/hikers/).*

### Delft market street

![Delft market street: photo · VTracer 0.6 (defaults) · VTracer 0.6 (tuned) · VTracer 1.0 watershed (tuned) · VTracer 1.0 colour (tuned) · SuperSVG · tracesmart (automatic) · tracesmart (`--care`)](../examples/delft-street/methods.jpg)

*photo · VTracer 0.6 (defaults) · VTracer 0.6 (tuned) · VTracer 1.0 watershed (tuned) · VTracer 1.0 colour (tuned) · SuperSVG · tracesmart (automatic) · tracesmart (`--care`). Photo by [Folco Masi](https://unsplash.com/photos/yvByaC2YqPs) on Unsplash. `--care`: sky, cloud, house, gable, window, roof, door, person, bicycle, lamp, building. All panels, SVGs and metrics: [`examples/delft-street/`](../examples/delft-street/).*

### Oostpoort gate, Delft

![Oostpoort gate, Delft: photo · VTracer 0.6 (defaults) · VTracer 0.6 (tuned) · VTracer 1.0 watershed (tuned) · VTracer 1.0 colour (tuned) · SuperSVG · tracesmart (automatic) · tracesmart (`--care`)](../examples/oostpoort/methods.jpg)

*photo · VTracer 0.6 (defaults) · VTracer 0.6 (tuned) · VTracer 1.0 watershed (tuned) · VTracer 1.0 colour (tuned) · SuperSVG · tracesmart (automatic) · tracesmart (`--care`). Photo by [Alex vd Slikke](https://unsplash.com/photos/GYIGt-MQwZ8) on Unsplash. `--care`: sky, tree, tower, spire, roof, window, chimney, brick wall, arch, bridge, water, road, fence, lamp. All panels, SVGs and metrics: [`examples/oostpoort/`](../examples/oostpoort/).*

### Mountain lake

![Mountain lake: photo · VTracer 0.6 (defaults) · VTracer 0.6 (tuned) · VTracer 1.0 watershed (tuned) · VTracer 1.0 colour (tuned) · SuperSVG · tracesmart (automatic) · tracesmart (`--care`)](../examples/mountain-lake/methods.jpg)

*photo · VTracer 0.6 (defaults) · VTracer 0.6 (tuned) · VTracer 1.0 watershed (tuned) · VTracer 1.0 colour (tuned) · SuperSVG · tracesmart (automatic) · tracesmart (`--care`). Photo by [Kalen Emsley](https://unsplash.com/photos/mgJSkgIo_JI) on Unsplash. `--care`: mountain, snow, lake, forest, tree, rock, sky, cloud, person, backpack. All panels, SVGs and metrics: [`examples/mountain-lake/`](../examples/mountain-lake/).*

### Lone hiker

![Lone hiker: photo · VTracer 0.6 (defaults) · VTracer 0.6 (tuned) · VTracer 1.0 watershed (tuned) · VTracer 1.0 colour (tuned) · SuperSVG · tracesmart (automatic) · tracesmart (`--care`)](../examples/lone-hiker/methods.jpg)

*photo · VTracer 0.6 (defaults) · VTracer 0.6 (tuned) · VTracer 1.0 watershed (tuned) · VTracer 1.0 colour (tuned) · SuperSVG · tracesmart (automatic) · tracesmart (`--care`). Photo by [Robert Bye](https://unsplash.com/photos/JvUVo08dndQ) on Unsplash. `--care`: person, backpack, shorts, sky, cloud, lake, forest, hill, rock, island, sea. All panels, SVGs and metrics: [`examples/lone-hiker/`](../examples/lone-hiker/).*

### Canal and church tower

![Canal and church tower: photo · VTracer 0.6 (defaults) · VTracer 0.6 (tuned) · VTracer 1.0 watershed (tuned) · VTracer 1.0 colour (tuned) · SuperSVG · tracesmart (automatic) · tracesmart (`--care`)](../examples/canal/methods.jpg)

*photo · VTracer 0.6 (defaults) · VTracer 0.6 (tuned) · VTracer 1.0 watershed (tuned) · VTracer 1.0 colour (tuned) · SuperSVG · tracesmart (automatic) · tracesmart (`--care`). Photo by [Casper van Battum](https://unsplash.com/photos/25i3kDguOAE) on Unsplash. `--care`: sky, tree, church tower, clock, building, car, bicycle, bridge, water, road. All panels, SVGs and metrics: [`examples/canal/`](../examples/canal/).*

## How it compares

The same six photos traced by [VTracer](https://github.com/visioncortex/vtracer) (a conventional colour-clustering
tracer; colour-clustering tracing is the common approach in image-trace tools), in two versions, and by
[SuperSVG](https://github.com/sjtuplayer/SuperSVG) (CVPR 2024, a learned vectoriser). The VTracer versions are the 0.6.15
Python package and the 1.0.0-alpha.4 command-line tool in its watershed and colour-clustering modes (with
`--simplify 2`). Every "tuned" method was given a path budget equal to tracesmart's `--care` count for that photo, and
VTracer's defaults are shown too. Averages over the six photos (per-photo numbers in
[`benchmarks/RESULTS.md`](../benchmarks/RESULTS.md)):

| Method | Paths | File size | PSNR | SSIM |
|---|---|---|---|---|
| VTracer 0.6, defaults | 14,074 | 15.4 MB | 22.8 | 0.76 |
| VTracer 0.6, tuned to about 130 paths | 129 | 1.5 MB | 15.3 | 0.47 |
| VTracer 1.0, watershed, tuned | 128 | 188 KB | 18.1 | 0.44 |
| VTracer 1.0, colour clustering, tuned | 129 | 626 KB | 18.5 | 0.51 |
| SuperSVG (CVPR 2024) | 128 | 70 KB | 19.1 | 0.44 |
| tracesmart, automatic | 107 | 62 KB | 16.2 | 0.41 |
| tracesmart, `--care` | 128 | 64 KB | 17.0 | 0.41 |

How to read this, honestly:

- **By pixel scores tracesmart is near the bottom** at a similar path count: its PSNR (17.0) beats only the tuned
  VTracer 0.6 (15.3), and its SSIM (0.41) is the lowest. PSNR and SSIM reward copying the photo; tracesmart fills
  every shape with one flat colour and drops texture on purpose.
- **VTracer's defaults are not a simplification**: about 14,000 paths and 15 MB per photo, with 0.6 and 1.0 alike.
- **VTracer 1.0 is a big step up from 0.6.** Tuned to about 130 paths its files are 188 KB (watershed) to 626 KB
  (colour clustering) instead of 1.5 MB, and it scores above tracesmart (PSNR 18.1 to 18.5). The watershed mode keeps
  edges such as cloud outlines and you can pick out the hikers, but its boundaries are ragged and parts of a person
  merge; the colour mode is speckled. Its files are still 3 to 10 times the size of tracesmart's.
- **SuperSVG** gives light files and the best PSNR of the tuned methods (19.1), but objects run together in a
  painterly blur: the hikers and the buildings are not clearly separate things.
- **What the scores miss** is whether a shape is an *object*. tracesmart's shapes are a hat, a backpack, a window,
  and with `--care` they are named. That is the thing it is built for, and no score here measures it, so judge it
  from the images.

### Why does vtracer's default look exactly like the photo?

Because it is made of thousands of tiny shapes. vtracer clusters pixels by colour and traces every cluster, so with
its defaults it produces about 14,000 paths per photo, roughly one per little patch of similar pixels. That is
close to a posterised copy of the photo (hence the high PSNR and SSIM), not a simplified illustration, and each
patch is a fragment of something, not an object. Colour-clustering tracers in general behave this way. Here is the same
region with every shape outlined:

![Close-up of the hikers with every shape outlined: vtracer defaults against tracesmart](../examples/hikers/closeup.jpg)

![Close-up of the houses with every shape outlined: vtracer defaults against tracesmart](../examples/delft-street/closeup.jpg)

To get the shape count down, vtracer's detail knobs have to be turned until whole regions merge by colour alone,
which is what the "tuned" column does: it keeps the speckle and loses the objects.

Caveats: six photos, one run each. VTracer was tuned by search on path count only (0.6: a grid over speckle,
precision and layer difference; 1.0: bisection on `--watershed-detail` or `--filter-speckle`), and the 1.0 CLI's
seam-free cutout mode and the desktop app were not tested. SuperSVG ran on the CPU with patches
([`benchmarks/SUPERSVG.md`](../benchmarks/SUPERSVG.md)); its path budget and the tuned VTracer settings were re-set
from tracesmart's final path counts. Reproduce with `benchmarks/run_all.sh`.
