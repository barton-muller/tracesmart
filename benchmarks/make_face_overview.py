"""One overview figure per photo: automatic trace on row 1, --care trace on row 2 (when there is one).

    uv run --extra faces python benchmarks/make_face_overview.py outputs/facebench/<name> \
        [--care outputs/facebench-care/<name>] [--original tests/images/bench/<name>.jpg] [-o overview.jpg]

Columns: photo | segment map with face boxes, face parse and landmarks | trace without faces | cartoon | detailed.
See face_figures.py. Run it from the repository root.
"""
import sys
from pathlib import Path

import typer

sys.path.insert(0, str(Path(__file__).parent))
from face_figures import overview  # noqa: E402

from tracesmart import backends  # noqa: E402


def main(auto: Path, care: Path | None = None, original: Path | None = None, out: Path | None = None,
         height: int = 520) -> None:
    out = out or auto / "overview.jpg"
    sheet = overview(auto, care, original, backends.best_device(), height)
    sheet.save(out, quality=88)
    print(f"{out} ({sheet.width}x{sheet.height})")


if __name__ == "__main__":
    typer.run(main)
