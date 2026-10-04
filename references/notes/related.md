# Related work, read from READMEs and abstracts (not run unless stated)

- **KU-MIIL/semantic-svg-generation** ([paper](https://arxiv.org/abs/2609.14657), EMNLP 2026). Pipeline: Gemini labels
  parts and predicts boxes, SAM 3 turns each label + box into a mask, vtracer vectorises each part's RGBA layer, a
  painter's-algorithm compositor stitches them back, optional amodal-completion stage. Ships a benchmark
  (HF `KU-MIIL/Semantic-SVG-Benchmark`, 203 samples: emoji, icons, illustrations, each with a semantic tree JSON that
  labels sets of SVG shapes) and an evaluation harness. Idea: score tracesmart's *semantic structure* with their
  harness (their input must be an image, ours produces SVG; their ground truth is icon-like, so photos do not fit
  directly). No licence declared at the time of writing.
- **AmodalSVG** (2604.10940): amodal vectorisation by semantic layer peeling, i.e. completing parts hidden behind
  others so a layer can be moved. tracesmart's lower shapes already extend behind upper ones, but hidden parts of
  *objects* (the body behind a backpack) are not completed.
- **Controlling Your Image via Simplified Vector Graphics** (2602.14443): image-to-SVG parsing with SAM and diffusion,
  hierarchical. Not read in depth.
- **LIVE** (2206.04655): progressive layer-wise vectorisation with DiffVG; adds paths where the render differs most.
  Needs CUDA or a CPU DiffVG build.
- **SuperSVG** (2406.09794): feed-forward superpixel-based vectoriser. Run on CPU with patches; results in
  `docs/COMPARISON.md`.
- **VTracer 1.0** (alpha): desktop app and `vtracer-cli` with `--clustering watershed`, `--hierarchical cutout`,
  `--max-colors`, `--simplify`. Installed locally with `cargo install vtracer-cli --version 1.0.0-alpha.4`.
- **gimpsegany** (SAM masks as GIMP layers), **lang-segment-anything** (text-prompted SAM masks, no vectors): adjacent.
