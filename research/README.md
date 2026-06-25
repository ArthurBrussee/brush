# Research

**Resource-Aware Optimization of a Cross-Platform 3D Gaussian Splatting Engine: An Empirical
Study on Apple Silicon** — Alex Gastone Mkwizu (Seede XR Group Limited · Seede XR Studios).
*AI-assisted research conducted with Claude (Anthropic).* Artifacts: https://github.com/SeedeXR/brush

- `brush-optimization-paper.md` — the paper (Mermaid figures render on GitHub/markdown viewers).
- `brush-optimization-paper.pdf` — typeset PDF with the figures rasterized.
- `build_pdf.py` — regenerates the PDF from the `.md`.

## Rebuild the PDF
Requires `markdown-it-py` (pip) and Google Chrome. Mermaid is loaded from a local
`mermaid.min.js` (e.g. `curl -L https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.min.js -o /tmp/mermaid.min.js`).
```bash
python3 research/build_pdf.py    # md -> self-contained HTML (inline mermaid)
"/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" --headless --disable-gpu \
  --print-to-pdf=research/brush-optimization-paper.pdf --no-pdf-header-footer \
  --virtual-time-budget=25000 --run-all-compositor-stages-before-draw \
  "file://$PWD/research/brush-optimization-paper.html"
```
Underlying data: `memory/results/measurements.csv` and the per-experiment notes.
