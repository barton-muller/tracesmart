# Examples and comparison

Six Unsplash photos, each traced automatically and with `--care`, and compared with other tools. Every panel of
every example, plus the SVGs and per-photo scores, is in [`examples/`](../examples/). Note that `--care` does not always
add shapes: described shapes also replace near-duplicate automatic ones, so the count can go down as the shapes
get better named.

### Hikers

![Hikers: photo · vtracer (defaults) · vtracer (tuned to the same path count) · SuperSVG · tracesmart (automatic) · tracesmart (`--care`)](../examples/hikers/methods.jpg)

*photo · vtracer (defaults) · vtracer (tuned to the same path count) · SuperSVG · tracesmart (automatic) · tracesmart (`--care`). Photo by [Dan Ordze](https://unsplash.com/photos/4GoNeNKEB1M) on Unsplash. `--care`: person, hat, backpack, shorts, shirt, boot, hiking pole, tree, mountain, sky, gravel path. All panels, SVGs and metrics: [`examples/hikers/`](../examples/hikers/).*

### Delft market street

![Delft market street: photo · vtracer (defaults) · vtracer (tuned to the same path count) · SuperSVG · tracesmart (automatic) · tracesmart (`--care`)](../examples/delft-street/methods.jpg)

*photo · vtracer (defaults) · vtracer (tuned to the same path count) · SuperSVG · tracesmart (automatic) · tracesmart (`--care`). Photo by [Folco Masi](https://unsplash.com/photos/yvByaC2YqPs) on Unsplash. `--care`: sky, cloud, house, gable, window, roof, door, person, bicycle, lamp, building. All panels, SVGs and metrics: [`examples/delft-street/`](../examples/delft-street/).*

### Oostpoort gate, Delft

![Oostpoort gate, Delft: photo · vtracer (defaults) · vtracer (tuned to the same path count) · SuperSVG · tracesmart (automatic) · tracesmart (`--care`)](../examples/oostpoort/methods.jpg)

*photo · vtracer (defaults) · vtracer (tuned to the same path count) · SuperSVG · tracesmart (automatic) · tracesmart (`--care`). Photo by [Alex vd Slikke](https://unsplash.com/photos/GYIGt-MQwZ8) on Unsplash. `--care`: sky, tree, tower, spire, roof, window, chimney, brick wall, arch, bridge, water, road, fence, lamp. All panels, SVGs and metrics: [`examples/oostpoort/`](../examples/oostpoort/).*

### Mountain lake

![Mountain lake: photo · vtracer (defaults) · vtracer (tuned to the same path count) · SuperSVG · tracesmart (automatic) · tracesmart (`--care`)](../examples/mountain-lake/methods.jpg)

*photo · vtracer (defaults) · vtracer (tuned to the same path count) · SuperSVG · tracesmart (automatic) · tracesmart (`--care`). Photo by [Kalen Emsley](https://unsplash.com/photos/mgJSkgIo_JI) on Unsplash. `--care`: mountain, snow, lake, forest, tree, rock, sky, cloud, person, backpack. All panels, SVGs and metrics: [`examples/mountain-lake/`](../examples/mountain-lake/).*

### Lone hiker

![Lone hiker: photo · vtracer (defaults) · vtracer (tuned to the same path count) · SuperSVG · tracesmart (automatic) · tracesmart (`--care`)](../examples/lone-hiker/methods.jpg)

*photo · vtracer (defaults) · vtracer (tuned to the same path count) · SuperSVG · tracesmart (automatic) · tracesmart (`--care`). Photo by [Robert Bye](https://unsplash.com/photos/JvUVo08dndQ) on Unsplash. `--care`: person, backpack, shorts, sky, cloud, lake, forest, hill, rock, island, sea. All panels, SVGs and metrics: [`examples/lone-hiker/`](../examples/lone-hiker/).*

### Canal and church tower

![Canal and church tower: photo · vtracer (defaults) · vtracer (tuned to the same path count) · SuperSVG · tracesmart (automatic) · tracesmart (`--care`)](../examples/canal/methods.jpg)

*photo · vtracer (defaults) · vtracer (tuned to the same path count) · SuperSVG · tracesmart (automatic) · tracesmart (`--care`). Photo by [Casper van Battum](https://unsplash.com/photos/25i3kDguOAE) on Unsplash. `--care`: sky, tree, church tower, clock, building, car, bicycle, bridge, water, road. All panels, SVGs and metrics: [`examples/canal/`](../examples/canal/).*

## How it compares

The same six photos traced by [vtracer](https://github.com/visioncortex/vtracer) (a conventional colour-clustering
tracer; colour-clustering tracing is the common approach in image-trace tools) and by
[SuperSVG](https://github.com/sjtuplayer/SuperSVG) (CVPR 2024, a learned vectoriser). vtracer and SuperSVG were
given a path budget close to tracesmart's; vtracer's defaults are shown too. Averages over the six photos
(per-photo numbers in [`benchmarks/RESULTS.md`](../benchmarks/RESULTS.md)):

| Method | Paths | File size | PSNR | SSIM |
|---|---|---|---|---|
| vtracer, defaults | 14,074 | 15.4 MB | 22.8 | 0.76 |
| vtracer, tuned to about 115 paths | 116 | 1.6 MB | 17.2 | 0.50 |
| SuperSVG (CVPR 2024) | 115 | 63 KB | 18.8 | 0.43 |
| tracesmart, automatic | 107 | 62 KB | 16.2 | 0.41 |
| tracesmart, `--care` | 128 | 64 KB | 17.0 | 0.41 |

How to read this, honestly:

- **By pixel scores tracesmart comes last** at a similar path count. PSNR and SSIM reward copying the photo;
  tracesmart fills every shape with one flat colour and drops texture on purpose.
- **vtracer's defaults are not a simplification**: about 14,000 paths and 15 MB per photo. Tuned down to about 115
  paths it is speckled, and its files are about 25 times the size of tracesmart's at the same path count (each path
  has many more nodes), which makes them harder to edit.
- **SuperSVG** gives light files and a painterly look, but it blurs objects: the hikers and the buildings are not
  recognisable as separate things.
- **What the scores miss** is whether a shape is an *object*. tracesmart's shapes are a hat, a backpack, a window,
  and with `--care` they are named. That is the thing it is built for, and it is not measured by any score here;
  judge it from the images in [`examples/`](../examples/).

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

Caveats: six photos, one run each; vtracer was tuned by a coarse search on path count only; SuperSVG ran on the
CPU with patches ([`benchmarks/SUPERSVG.md`](../benchmarks/SUPERSVG.md)) and its path budget was set from tracesmart's
count before gap-filling shapes were added, so tracesmart has up to a quarter more paths than SuperSVG on some photos.
Reproduce with `benchmarks/compare.py`.
