# SAMVG (Zhu et al., ICASSP 2024), arXiv 2311.05276

Read from the paper text. No code was released, so tracesmart's `pipeline.py` is a re-implementation from this.

## Stages

1. **Masks.** SAM's Automatic Mask Generator, prompted with a 32 x 32 point grid, with test-time augmentation
   (several overlapping zoomed crops). Filtered by confidence score and IoU; small components and holes removed.
2. **Filter by impact.** Sort masks by area, descending. Start from a blank canvas `C0`. For each mask `m_i`, give it
   the mean colour of the photo under it and paint it: `C_i = f(C_{i-1}, m_i, c_i)`. Error
   `e_i = ||I - C_i||^2 / MaskArea`, with **uncovered pixels counted at the maximum error** so the measure is not biased
   towards bright pixels. Impact `gamma_i = e_i - e_{i-1}`. If a mask's impact is below a threshold, discard it and
   revert the canvas. This keeps sub-parts of objects that have high IoU with their parent, and drops small or wrong
   masks. tracesmart: `pipeline.Canvas.try_add` (`--impact`).
3. **Uncovered regions.** OR all masks into an alpha mask; convolve with a circular kernel (radius is a fraction of the
   image size); zero values mark centres of large uncovered regions; **mean-shift** clustering turns those points into
   prompts for a second SAM round; filter again. tracesmart: `prompt_points` (distance transform, equivalent to the
   convolution) and `backends.Prompter`. The paper also detects *missing components* from the difference map between
   the render and the target, the same way.
4. **Tracing.** Corner points chosen as the global maximum first, then others after removing nearby candidates, up to a
   fixed number per path (so path complexity is comparable with baselines). Initial SVG of cubic Beziers.
5. **Optimisation.** DiffVG with MSE loss plus regularisers for smoothness and artefacts, for a set number of iterations,
   then repeat 3 and 4 for missing components and optimise again.

## What tracesmart changes

- SAM 2.1 instead of SAM. No DiffVG optimisation: colours are the mean of the *visible* pixels (`render.render`).
- Corner-preserving tracing (`trace.mask_path`) instead of a fixed number of corners per path.
- Uncovered areas become shapes at render time; `close_seams` grows shapes by about 2 px.
- SAM 3 text descriptions (`--care`), shapes named after their phrase, layer modes. None of these are in SAMVG.

## Ideas from the paper not done

- Optimising shapes and colours with DiffVG (needs the heavy dependency; SuperSVG shows it can build on a Mac CPU).
- Mean-shift clustering of candidate prompt points (tracesmart takes distance-transform maxima instead).
