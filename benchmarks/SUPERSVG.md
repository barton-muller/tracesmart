# Running SuperSVG on an Apple-silicon Mac (CPU)

[SuperSVG](https://github.com/sjtuplayer/SuperSVG) (CVPR 2024) is the learned, DiffVG-based method used as the
"state of the art" reference in the comparison. It targets CUDA and Python 3.7, so it needs a few patches to run on
a Mac. This is what worked here (Python 3.13, PyTorch 2.14, macOS arm64). Nothing in `tracesmart` depends on it.

```bash
mkdir vector-bench && cd vector-bench
git clone --depth 1 --recurse-submodules https://github.com/sjtuplayer/SuperSVG.git
cd SuperSVG
uv venv --python 3.13 .venv && source .venv/bin/activate
uv pip install torch torchvision numpy scikit-image svgwrite svgpathtools cssutils numba scikit-fmm easydict \
    opencv-python-headless huggingface_hub setuptools wheel pybind11 timm einops scikit-learn scipy matplotlib lpips
```

**Build DiffVG** (bundled in `diffvg/`). Its pybind11 and thrust are too old for current toolchains:

```bash
cd diffvg
rm -rf pybind11 && git clone --depth 1 --branch v2.13.6 https://github.com/pybind/pybind11.git pybind11
sed -i '' 's/_VSTD::__wrap_iter/std::__wrap_iter/' thrust/thrust/type_traits/is_contiguous_iterator.h
CMAKE_POLICY_VERSION_MINIMUM=3.5 python setup.py install
cd ..
```

**Run on the CPU.** The code hard-codes CUDA; swap it for the CPU:

```bash
sed -i '' -e 's/\.cuda()/.cpu()/g' -e "s/'cuda'/'cpu'/g" -e 's/set_use_gpu(True)/set_use_gpu(False)/g' models/*.py util/*.py
sed -i '' -e 's/\.cuda()/.cpu()/g' -e 's/set_use_gpu(True)/set_use_gpu(False)/g' inference.py
mkdir weights && curl -L -o weights/dino_deitsmall16_pretrain.pth \
    https://dl.fbaipublicfiles.com/dino/dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth   # ~87 MB
```

The SuperSVG checkpoints (`coarse.pt`, `refine.pt`) download from Hugging Face on the first run.

```bash
SUPERSVG_DIR=$PWD .venv/bin/python /path/to/tracesmart/benchmarks/run_supersvg.py photo.png 120 out.svg
```

Notes:

- SuperSVG works on a 512x512 square. The wrapper squashes the photo to a square and stretches the SVG back to
  the photo's proportions with a `viewBox` change, which loses nothing.
- Call the venv's `python` directly. Running it after `source .venv/bin/activate` from a bash script made DiffVG
  segfault here. Rare segfaults also happened under load; the wrapper retries.
- A 512 px render with 100-200 paths takes about a minute on the CPU.
