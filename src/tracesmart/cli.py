"""tracesmart command line.

  tracesmart trace photo.jpg -o out/                                  # automatic: one SVG path per region
  tracesmart trace photo.jpg -o out/ --care "window, red shutters"    # plus things you name (SAM 3)
  tracesmart rerender photo.jpg out/masks.npz                         # redraw from saved masks, no model
  tracesmart index --folder outputs                                   # one HTML page showing every run
"""
import json
from pathlib import Path

import numpy as np
import typer
from PIL import Image

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    rich_markup_mode=None,
    help="Image tracing that understands the picture: SAM 2.1 / SAM 3, one flat-colour SVG path per object.",
    epilog="Examples: tracesmart trace photo.jpg -o out/ | tracesmart trace photo.jpg --care 'window, roof' | "
           "tracesmart rerender photo.jpg out/masks.npz",
)


def load_image(path: Path, max_side: int) -> Image.Image:
    img = Image.open(path).convert("RGB")
    if max(img.size) > max_side:
        k = max_side / max(img.size)
        img = img.resize((round(img.width * k), round(img.height * k)), Image.LANCZOS)
    return img


@app.command()
def trace(
    image: Path,
    out: Path = typer.Option(Path("tracesmart-out"), "-o", "--out", help="output folder"),
    care: str = typer.Option(
        None, "--care", "--text",
        help="things to describe, comma separated: 'window, red shutters, dog'. "
             "Found by SAM 3 (gated: accept the licence and run `hf auth login`)."),
    care_threshold: float = typer.Option(0.5, help="SAM 3 confidence; lower finds more and fainter things"),
    grid: int = typer.Option(32, help="SAM 2.1 prompt grid (points per side); higher finds more, slower"),
    crops: bool = typer.Option(True, help="add zoomed crops as test-time augmentation"),
    impact: float = typer.Option(
        2e-4, help="how much a shape must reduce the image error to be kept; lower = more detail"),
    rounds: int = typer.Option(2, help="how many times to fill badly covered regions"),
    min_area: float = typer.Option(0.0004, help="smallest automatic shape, as a fraction of the image"),
    max_side: int = typer.Option(1024, help="downscale the image so its long side is at most this"),
    tone_rms: float = typer.Option(
        0.0, help="split shapes whose colour varies more than this into tone patches (0 = off)"),
    round_px: float = typer.Option(None, help="mask smoothing radius in pixels (0 = off)"),
    seam_px: float = typer.Option(None, help="grow shapes by up to this many pixels so neighbours overlap (0 = off)"),
    layers: str = typer.Option(
        "none", help="organise the SVG: none (flat), objects (parts nested in their object), "
                     "levels (Inkscape layers, coarse to fine), depth (layers back to front, no overlaps in a layer)"),
    group: str = typer.Option(
        "person", help="with --layers objects: phrases whose shapes become groups, e.g. 'person, dog'"),
    complete: bool = typer.Option(True, help="with --layers objects: extend the shapes below a group under it, so "
                                             "moving the group leaves no hole"),
    zoom: float = typer.Option(3.0, help="scale of the PNG renders"),
):
    """Segment IMAGE and write stacked, flat-colour SVG shapes plus PNG previews."""
    from .pipeline import vectorise
    from .render import write_outputs
    from .tones import tone_patches

    img = load_image(image, max_side)
    out.mkdir(parents=True, exist_ok=True)
    phrases_wanted = [p.strip() for p in care.split(",") if p.strip()] if care else None
    masks, phrases, report = vectorise(
        img, grid=grid, crops=crops, impact=impact, rounds=rounds, min_area=min_area, care=phrases_wanted,
        care_threshold=care_threshold, cache=out / "raw_masks.npz", log=typer.echo)
    if tone_rms > 0:
        extra = tone_patches(img, masks, tone_rms, log=typer.echo)
        masks, phrases = masks + extra, phrases + [None] * len(extra)
    (out / "run.json").write_text(json.dumps({
        "image": image.name, "care": report, "grid": grid, "impact": impact, "rounds": rounds,
        "care_threshold": care_threshold, "max_side": max_side}, indent=1))
    np.savez_compressed(out / "masks.npz", masks=np.stack(masks), phrases=np.array([p or "" for p in phrases]))
    img.save(out / "source.png")
    write_outputs(img, masks, phrases, out, round_px, zoom, seam_px, layers, group, complete)
    typer.echo(f"{len(masks)} shapes -> {out}/vector.svg")


@app.command()
def rerender(
    image: Path,
    masks: Path = typer.Argument(..., help="masks.npz from an earlier `trace` run"),
    out: Path = typer.Option(None, "-o", "--out", help="output folder (default: next to masks.npz)"),
    max_side: int = typer.Option(1024, help="use the same value as the original run"),
    round_px: float = typer.Option(None, help="mask smoothing radius in pixels (0 = off)"),
    seam_px: float = typer.Option(None, help="grow shapes by up to this many pixels so neighbours overlap (0 = off)"),
    layers: str = typer.Option(
        "none", help="organise the SVG: none (flat), objects (parts nested in their object), "
                     "levels (Inkscape layers, coarse to fine), depth (layers back to front, no overlaps in a layer)"),
    group: str = typer.Option(
        "person", help="with --layers objects: phrases whose shapes become groups, e.g. 'person, dog'"),
    complete: bool = typer.Option(True, help="with --layers objects: extend the shapes below a group under it, so "
                                             "moving the group leaves no hole"),
    zoom: float = 3.0,
):
    """Redraw the outputs from saved masks in a few seconds (no model needed)."""
    from .render import write_outputs

    data = np.load(masks)
    out = out or masks.parent
    out.mkdir(parents=True, exist_ok=True)
    phrases = [p or None for p in data["phrases"].tolist()] if "phrases" in data else None
    write_outputs(load_image(image, max_side), list(data["masks"]), phrases, out, round_px, zoom, seam_px, layers,
                  group, complete)
    typer.echo(f"redrawn -> {out}")


@app.command()
def faces(
    image: Path,
    masks: Path = typer.Argument(..., help="masks.npz from an earlier `trace` run"),
    out: Path = typer.Option(None, "-o", "--out", help="output folder (default: a `faces` folder next to masks.npz)"),
    max_side: int = typer.Option(1024, help="use the same value as the original run"),
    from_original: bool = typer.Option(True, help="face models read the original photo, not the downscaled trace image "
                                                  "(sharper on small faces; shapes keep the trace size)"),
    style: str = typer.Option("cartoon", help="cartoon (eyes are dots, mouth a line or open shape) or detailed "
                                              "(eye whites, irises, lips, teeth)"),
    layers: str = typer.Option("none", help="as for `trace`: none, objects, levels or depth"),
    group: str = typer.Option("person", help="with --layers objects: phrases whose shapes become groups"),
    zoom: float = 3.0,
):
    """Second stage: add facial detail (eyes, brows, lips, teeth, glasses, hair, skin) to a finished trace.

    Needs the optional extra: uv sync --extra faces. Earlier face shapes in MASKS are replaced, so it can be rerun.
    """
    from . import backends
    from .faces import add_details
    from .render import write_outputs

    data = np.load(masks)
    out = out or masks.parent / "faces"
    out.mkdir(parents=True, exist_ok=True)
    img = load_image(image, max_side)
    source = Image.open(image).convert("RGB") if from_original else None
    if source is not None and source.width <= img.width:
        source = None  # nothing sharper to read
    phrases = [p or None for p in data["phrases"].tolist()] if "phrases" in data else [None] * len(data["masks"])
    items, counts = add_details(list(zip(data["masks"], phrases, strict=True)), img, backends.best_device(),
                                style, log=typer.echo, source=source)
    new_masks, new_phrases = [m for m, _ in items], [p for _, p in items]
    np.savez_compressed(out / "masks.npz", masks=np.stack(new_masks), phrases=np.array([p or "" for p in new_phrases]))
    img.save(out / "source.png")
    write_outputs(img, new_masks, new_phrases, out, None, zoom, None, layers, group, True)
    typer.echo(f"{len(new_masks)} shapes ({sum(counts.values())} face parts) -> {out}/vector.svg")


@app.command()
def index(folder: Path = typer.Option(Path("outputs"), help="folder of runs laid out as <image>/<variant>/")):
    """Write FOLDER/index.html: every run's source | segments | vector, with its care words."""
    from .render import build_index

    typer.echo(f"{build_index(folder)} runs -> {folder}/index.html")


def main():
    app()
