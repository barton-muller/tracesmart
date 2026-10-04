# Layered Image Vectorization via Semantic Simplification (Wang et al., CVPR 2025), arXiv 2406.05404

Code: https://github.com/SZUVIZ/layered_vectorization (**no licence declared**, so do not copy it into this repo).
Read from the paper text (sections 3 to 6).

## Method

1. **Progressive simplification with Score Distillation Sampling (SDS).** Run SDS on the input with a Stable Diffusion
   model. Plain SDS abstracts the image but distorts shapes ("feature-average effect"). The fix: increase the share of
   *unconditional* noise, either by setting the CFG scale to zero or by using an **empty text prompt** (they use the
   empty prompt). Keep an image every **N = 20** SDS iterations; by default **five images** (including the original).
   Compared with Gaussian, bilateral and superpixel simplification, SDS keeps macro shapes and gives smooth mask
   boundaries. (Their ablation uses Gaussian kernels 2, 6, 10, 14; bilateral `(10+5N, 100+50N, 100+50N)` for N = 0..3;
   superpixel counts 400, 200, 100, 50.)
2. **Stage I, structural construction.** Run SAM on *every* image in the simplified sequence, so coarse images give
   whole objects ("the entire body of Captain America", "grassland") that the detailed image misses. **Layer the masks**:
   large masks at the back, small in front; **masks within a layer never intersect**. Iterating from the most simplified
   image to the least, each mask goes into the furthest-back layer where it does not overlap a mask already there.
   Boundaries are simplified with Douglas-Peucker; each mask becomes a closed cubic-Bezier shape, optimised with DiffVG
   against its mask using a layer-wise MSE loss (random colours, shape only) plus an overlap penalty (weights 1 and 1e-8).
3. **Stage II, visual refinement.** Fit colours to the structure vectors (dominant visible colour, or MSE fit), freeze
   them, then add extra "visual" vector primitives at the largest regions where the render differs from the target
   (as in LIVE), optimise, and periodically merge or remove redundant vectors.

Layers are described as "well aligned with the image's explicit and implicit semantic structures", easier to edit than
a flat result. They compare against four methods at 64, 128 and 256 primitives.

## Cost and requirements (why it was not run)

- Python 3.10, PyTorch 1.13.1 with CUDA 11.6, DiffVG built from source, SAM ViT-H (about 2.5 GB), Stable Diffusion
  v1.5 (several GB). Authors used four NVIDIA A40 GPUs (48 GB each). **Time per image is not stated.**
- On the dev machine (Apple M2, 16 GB) a port is conceivable (SuperSVG's DiffVG built on the CPU with patches, see
  `benchmarks/SUPERSVG.md`; diffusers can use MPS) but the SDS loop plus DiffVG optimisation of hundreds of paths per
  level would likely take hours per image. Not attempted.

## What tracesmart took from it, and what could still be done

Done:
- `--layers depth`: this paper's layering, derived from the painter's stack: a shape's layer is one more than the
  deepest shape it covers, so layers never contain overlapping shapes and run back to front (`layers.depths`).
- `--layers levels`: coarse-to-fine layers, by *impact* (error reduction, SAMVG's measure) instead of SDS levels.
- `--layers objects`: parts nested in the shape that contains them.

Cheap ideas not done yet:
- **Multi-level SAM**: run SAM on progressively simplified copies of the photo (bilateral filtering or mean shift
  instead of SDS) and pool the masks before filtering by impact. The paper's Figure 9 argues this finds implicit
  semantic masks (whole bodies, grassland) that the detailed image misses. A day of work at most.
- Assigning layers progressively from simplified to detailed masks (as in Stage I) instead of from the final stack.
- A DiffVG refinement of colours and shapes (Stage II) using the SuperSVG environment.
