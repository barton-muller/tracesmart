#!/usr/bin/env bash
# Download the reference papers into references/papers/ (not committed: arXiv's default licence does not allow
# redistributing the PDFs). Re-run any time; existing files are skipped.
cd "$(dirname "$0")/papers" || exit 1
fetch() {  # fetch FILENAME ARXIV_ID
  if [ -s "$1" ]; then echo "have  $1"; return; fi
  curl -sL -o "$1" "https://arxiv.org/pdf/$2" && head -c 4 "$1" | grep -q '%PDF' && echo "got   $1" || { echo "FAIL  $1"; rm -f "$1"; }
}
fetch SAMVG_2311.05276_Zhu.pdf 2311.05276
fetch LayeredVec-SemanticSimplification_2406.05404_Wang.pdf 2406.05404
fetch SuperSVG_2406.09794_Hu.pdf 2406.09794
fetch LIVE-LayerwiseVectorization_2206.04655_Ma.pdf 2206.04655
fetch SemanticSVG-VLM_2609.14657_Park.pdf 2609.14657
fetch AmodalSVG_2604.10940_Hu.pdf 2604.10940
fetch SimplifiedVectorGraphics-Control_2602.14443_Guo.pdf 2602.14443
fetch SAM2_2408.00714_Ravi.pdf 2408.00714
fetch SAM3_2511.16719_Carion.pdf 2511.16719
