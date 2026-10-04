# References

Papers and projects tracesmart builds on or is compared with, with short notes on what matters for this repo.

The PDFs are **not committed** (arXiv's default licence does not allow redistributing them). Fetch them with

```bash
./references/fetch.sh          # downloads about 160 MB into references/papers/ (gitignored), skips what is there
```

| Paper | Local file | Why it matters | Notes |
|---|---|---|---|
| **SAMVG**: A Multi-stage Image Vectorization Model with the Segment-Anything Model (Zhu, Chong, Hu, Yi, Lai, Rosin; ICASSP 2024), [2311.05276](https://arxiv.org/abs/2311.05276) | `SAMVG_2311.05276_Zhu.pdf` | The method tracesmart re-implements. No code released | [notes/SAMVG.md](notes/SAMVG.md) |
| **Layered Image Vectorization via Semantic Simplification** (Wang, Huang, Sun, Gong, Cohen-Or, Lu; CVPR 2025), [2406.05404](https://arxiv.org/abs/2406.05404), [code](https://github.com/SZUVIZ/layered_vectorization) | `LayeredVec-SemanticSimplification_2406.05404_Wang.pdf` | The layering idea (`--layers depth`) and the coarse-to-fine idea (`--layers levels`). The most relevant paper for editing-friendly output. Needs CUDA, SD 1.5 and DiffVG | [notes/semantic-simplification.md](notes/semantic-simplification.md) |
| **SuperSVG** (Hu et al.; CVPR 2024), [2406.09794](https://arxiv.org/abs/2406.09794), [code](https://github.com/sjtuplayer/SuperSVG) | `SuperSVG_2406.09794_Hu.pdf` | Learned vectoriser used as the "state of the art" in the benchmark; runs on a Mac CPU with patches | [../benchmarks/SUPERSVG.md](../benchmarks/SUPERSVG.md) |
| **Towards Layer-wise Image Vectorization** (LIVE; Ma et al.; CVPR 2022), [2206.04655](https://arxiv.org/abs/2206.04655) | `LIVE-LayerwiseVectorization_2206.04655_Ma.pdf` | The DiffVG-based progressive method the later papers build on | [notes/related.md](notes/related.md) |
| **Compositional SVG Generation via VLM-Driven Hierarchical Semantic Parsing** (Park et al.; EMNLP 2026), [2609.14657](https://arxiv.org/abs/2609.14657), [code](https://github.com/KU-MIIL/semantic-svg-generation) | `SemanticSVG-VLM_2609.14657_Park.pdf` | Closest in spirit: VLM names parts, SAM 3 masks, vtracer per part, painter's compositor. Also a benchmark with semantic trees | [notes/related.md](notes/related.md) |
| **AmodalSVG**: Amodal Image Vectorization via Semantic Layer Peeling (Hu et al.), [2604.10940](https://arxiv.org/abs/2604.10940) | `AmodalSVG_2604.10940_Hu.pdf` | Completes occluded parts so layers can be moved | [notes/related.md](notes/related.md) |
| **Controlling Your Image via Simplified Vector Graphics** (Guo et al.), [2602.14443](https://arxiv.org/abs/2602.14443) | `SimplifiedVectorGraphics-Control_2602.14443_Guo.pdf` | Image-to-SVG parsing with SAM and diffusion, hierarchical | [notes/related.md](notes/related.md) |
| **SAM 2** (Ravi et al.), [2408.00714](https://arxiv.org/abs/2408.00714) | `SAM2_2408.00714_Ravi.pdf` | The automatic and point-prompted segmenter | |
| **SAM 3**: Segment Anything with Concepts (Carion et al.), [2511.16719](https://arxiv.org/abs/2511.16719) | `SAM3_2511.16719_Carion.pdf` | The text-prompted segmenter behind `--care` | |

Software compared against: [VTracer](https://github.com/visioncortex/vtracer) (0.6 Python package and the 1.0 CLI),
SuperSVG. See [../docs/COMPARISON.md](../docs/COMPARISON.md).
