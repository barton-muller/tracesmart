# Examples

Each folder holds the downscaled source photo, `vector.svg` (made with `--care`), `vector-auto.svg` (automatic only),
`compare.jpg` (source | segment map | vector), `methods.jpg` (the same photo traced by other methods, see
[../benchmarks](../benchmarks)) and `run.json` (the settings and how many instances each phrase found).

| Example | Photo | Shapes (auto) | Shapes (`--care`) | Phrases |
|---|---|---|---|---|
| [oostpoort](oostpoort/) | [Alex vd Slikke](https://unsplash.com/photos/GYIGt-MQwZ8) | 70 | 94 | sky, tree, tower, spire, roof, window, chimney, brick wall, arch, bridge, water, road, fence, lamp |
| [canal](canal/) | [Casper van Battum](https://unsplash.com/photos/25i3kDguOAE) | 122 | 126 | sky, tree, church tower, clock, building, car, bicycle, bridge, water, road |
| [hikers](hikers/) | [Dan Ordze](https://unsplash.com/photos/4GoNeNKEB1M) | 139 | 126 | person, hat, backpack, shorts, shirt, boot, hiking pole, tree, mountain, sky, gravel path |
| [delft-street](delft-street/) | [Folco Masi](https://unsplash.com/photos/yvByaC2YqPs) | 155 | 234 | sky, cloud, house, gable, window, roof, door, person, bicycle, lamp, building |
| [mountain-lake](mountain-lake/) | [Kalen Emsley](https://unsplash.com/photos/mgJSkgIo_JI) | 63 | 105 | mountain, snow, lake, forest, tree, rock, sky, cloud, person, backpack |
| [lone-hiker](lone-hiker/) | [Robert Bye](https://unsplash.com/photos/JvUVo08dndQ) | 85 | 82 | person, backpack, shorts, sky, cloud, lake, forest, hill, rock, island, sea |

All six were run with `--grid 48 --impact 3e-5 --max-side 1280 --rounds 3 --care-threshold 0.4`.

## Photo credits

The photographs are from [Unsplash](https://unsplash.com) and are used under the
[Unsplash License](https://unsplash.com/license). The vectors here are derivative illustrations made from them;
the original photographers retain their credit. Please credit them if you reuse the vectors:

- Alex vd Slikke, [photo GYIGt-MQwZ8](https://unsplash.com/photos/GYIGt-MQwZ8)
- Casper van Battum, [photo 25i3kDguOAE](https://unsplash.com/photos/25i3kDguOAE)
- Dan Ordze, [photo 4GoNeNKEB1M](https://unsplash.com/photos/4GoNeNKEB1M)
- Folco Masi, [photo yvByaC2YqPs](https://unsplash.com/photos/yvByaC2YqPs)
- Kalen Emsley, [photo mgJSkgIo_JI](https://unsplash.com/photos/mgJSkgIo_JI)
- Robert Bye, [photo JvUVo08dndQ](https://unsplash.com/photos/JvUVo08dndQ)
