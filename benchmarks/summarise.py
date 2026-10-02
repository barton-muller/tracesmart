"""Collect outputs/*/methods/methods.json into one Markdown table (per image and averaged).

    uv run python benchmarks/summarise.py [outputs] > benchmarks/RESULTS.md
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

root = Path(sys.argv[1] if len(sys.argv) > 1 else "outputs")
runs = {p.parent.parent.name: json.loads(p.read_text()) for p in sorted(root.glob("*/methods/methods.json"))}
avg = defaultdict(list)
print("# Results\n")
print("PSNR/SSIM compare each rendering with the photo, so they reward copying pixels, not simplifying. "
      "Read them next to the path count and file size.\n")
for name, rows in runs.items():
    print(f"## {name}\n\n| method | paths | KB | PSNR | SSIM |\n|---|---|---|---|---|")
    for r in rows:
        print(f"| {r['method']} | {r['paths']} | {r['kb']} | {r['psnr']} | {r['ssim']} |")
        avg[r["method"]].append(r)
    print()
print(f"## Average over {len(runs)} photos\n\n| method | paths | KB | PSNR | SSIM |\n|---|---|---|---|---|")
for method, rs in avg.items():
    mean = {k: sum(r[k] for r in rs) / len(rs) for k in ("paths", "kb", "psnr", "ssim")}
    print(f"| {method} | {mean['paths']:.0f} | {mean['kb']:.0f} | {mean['psnr']:.1f} | {mean['ssim']:.2f} |")
