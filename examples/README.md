# Examples

Six Unsplash photos traced automatically and with `--care`, and compared with other tools (see the [README](../README.md#how-it-compares) for the numbers and how to read them).

| Example | Photo | `--care` phrases |
|---|---|---|
| [hikers](hikers/) | [Dan Ordze](https://unsplash.com/photos/4GoNeNKEB1M) | person, hat, backpack, shorts, shirt, boot, hiking pole, tree, mountain, sky, gravel path |
| [delft-street](delft-street/) | [Folco Masi](https://unsplash.com/photos/yvByaC2YqPs) | sky, cloud, house, gable, window, roof, door, person, bicycle, lamp, building |
| [oostpoort](oostpoort/) | [Alex vd Slikke](https://unsplash.com/photos/GYIGt-MQwZ8) | sky, tree, tower, spire, roof, window, chimney, brick wall, arch, bridge, water, road, fence, lamp |
| [mountain-lake](mountain-lake/) | [Kalen Emsley](https://unsplash.com/photos/mgJSkgIo_JI) | mountain, snow, lake, forest, tree, rock, sky, cloud, person, backpack |
| [lone-hiker](lone-hiker/) | [Robert Bye](https://unsplash.com/photos/JvUVo08dndQ) | person, backpack, shorts, sky, cloud, lake, forest, hill, rock, island, sea |
| [canal](canal/) | [Casper van Battum](https://unsplash.com/photos/25i3kDguOAE) | sky, tree, church tower, clock, building, car, bicycle, bridge, water, road |

All six were run with `--grid 48 --impact 3e-5 --max-side 1280 --rounds 3 --care-threshold 0.4`.

## What is in each folder

| File | What |
|---|---|
| `source.jpg` | The downscaled photo (1280 px on the long side) |
| `vector.svg` | tracesmart with `--care`; described shapes are named after their phrase (`window-12`) |
| `vector-auto.svg` | tracesmart, automatic only |
| `vector-objects.svg`, `vector-levels.svg`, `vector-depth.svg` | The `--care` result organised for editing: parts nested in their object, Inkscape layers coarse to fine, Inkscape layers back to front (see [../docs/USAGE.md](../docs/USAGE.md#layers-for-editing)) |
| `supersvg.svg` | The SuperSVG (CVPR 2024) result with the same path budget |
| `methods.jpg` | All panels in one row |
| `compare.jpg` | source, segment map and vector |
| `panels/` | Each panel on its own: `photo`, `vtracer-defaults`, `vtracer-matched`, `vtracer1-watershed`, `vtracer1-colour`, `supersvg`, `tracesmart-automatic`, `tracesmart-care`, `segments` |
| `metrics.json` | Paths, file size, PSNR and SSIM per method |
| `closeup.jpg` | (hikers, delft-street) the same region with every shape outlined, vtracer against tracesmart |

The vtracer SVGs are not included: the default one is about 15 to 20 MB per photo and the tuned one 1 to 2 MB.
Regenerate them with `benchmarks/compare.py`.

## Gallery

### Hikers

![Hikers](hikers/methods.jpg)

### Delft market street

![Delft market street](delft-street/methods.jpg)

### Oostpoort gate, Delft

![Oostpoort gate, Delft](oostpoort/methods.jpg)

### Mountain lake

![Mountain lake](mountain-lake/methods.jpg)

### Lone hiker

![Lone hiker](lone-hiker/methods.jpg)

### Canal and church tower

![Canal and church tower](canal/methods.jpg)

## Photo credits

The photographs are from [Unsplash](https://unsplash.com) and are used under the [Unsplash License](https://unsplash.com/license). The vectors here are derivative illustrations made from them; please credit the photographers if you reuse them:

- Dan Ordze, [photo 4GoNeNKEB1M](https://unsplash.com/photos/4GoNeNKEB1M)
- Folco Masi, [photo yvByaC2YqPs](https://unsplash.com/photos/yvByaC2YqPs)
- Alex vd Slikke, [photo GYIGt-MQwZ8](https://unsplash.com/photos/GYIGt-MQwZ8)
- Kalen Emsley, [photo mgJSkgIo_JI](https://unsplash.com/photos/mgJSkgIo_JI)
- Robert Bye, [photo JvUVo08dndQ](https://unsplash.com/photos/JvUVo08dndQ)
- Casper van Battum, [photo 25i3kDguOAE](https://unsplash.com/photos/25i3kDguOAE)
