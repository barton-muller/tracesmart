# Credits

## SAMVG

tracesmart re-implements the method of SAMVG; please cite the paper if you use the idea:

```bibtex
@inproceedings{zhu2024samvg,
  title     = {{SAMVG}: A Multi-stage Image Vectorization Model with the Segment-Anything Model},
  author    = {Zhu, Haokun and Chong, Juang Ian and Hu, Teng and Yi, Ran and Lai, Yu-Kun and Rosin, Paul L.},
  booktitle = {ICASSP 2024 - IEEE International Conference on Acoustics, Speech and Signal Processing},
  year      = {2024}
}
```

- Paper: <https://arxiv.org/abs/2311.05276>
- No code was released by the authors; tracesmart is written from the paper.

## Models and tools

- [SAM 2.1](https://github.com/facebookresearch/sam2) and [SAM 3](https://github.com/facebookresearch/sam3) by Meta; weights are under Meta's licences.
- Faces stage: [face-alignment](https://github.com/1adrianb/face-alignment) (FAN landmarks, Bulat and Tzimiropoulos 2017; RetinaFace detector, Deng et al. 2019), and the [face-parsing SegFormer](https://huggingface.co/jonathandinu/face-parsing) fine-tuned on CelebAMask-HQ (SegFormer, Xie et al. 2021). Check each model's licence before reuse.
- [resvg](https://github.com/linebender/resvg) for SVG to PNG.
- [vtracer](https://github.com/visioncortex/vtracer) and [SuperSVG](https://github.com/sjtuplayer/SuperSVG) for the comparison.
- Example photographs: [Unsplash](https://unsplash.com), credited in [../examples/README.md](../examples/README.md).
