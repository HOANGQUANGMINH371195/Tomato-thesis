# LaTeX thesis report

This folder contains the thesis source and a self-contained copy of every figure used in
the document. It uses an article layout, 2.5 cm margins, double spacing, a USTH framed
cover, section-based numbering, and standard preliminary pages. The
student name, student ID, external supervisor, and internal supervisor are intentionally
blank in `main.tex`.

Build from this folder with:

```bash
~/.local/bin/tectonic main.tex
```

The command creates `main.pdf`. Tectonic 0.16.9 is installed at
`~/.local/bin/tectonic` in the current environment.

The numerical results come from the successful experiment stored under
`artifacts/runs/overnight_20260816_183758`. Yield-impact assumptions and their academic
sources are explained directly in the Methodology and bibliography. Backbone figures are
cropped from the cited papers; project-specific architecture and result figures come from
the training pipeline.

Wide workflows, architecture diagrams, training curves, and qualitative masks are placed
on dedicated landscape pages so their labels and image details remain clear. Every figure
has a separate analysis paragraph.

Figure 1 and Figure 7 are defined as Mermaid diagrams in `diagrams/`. Their rendered PNG
files are stored with the other report assets so rebuilding the PDF does not require Node.js.
