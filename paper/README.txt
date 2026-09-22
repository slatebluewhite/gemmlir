gemmlir -- LNCS full paper, submission package
==============================================

Build (LaTeX2e, pdfLaTeX; run twice for cross-references):

    pdflatex gemmlir
    pdflatex gemmlir

Files
    gemmlir.tex            the paper
    gemmlir-pipeline.png   Figure 1 (raster; pdflatex, not latex+dvips)
    llncs.cls              Springer LNCS class, v2.24
    splncs04.bst           LNCS BibTeX style -- not needed, the paper uses an
                           inline thebibliography, but included in case the
                           references are moved to a .bib

No non-standard packages: fontenc, graphicx, amsmath, amssymb only.

Before submitting
    - \author: the ORCID in the title block is a placeholder (0000-...-0000),
      and co-authors and affiliations need filling in.
    - \subsubsection{\ackname}: add funding/grant text if any.
    - Target length is 15 pages including bibliography and appendix. The header
      comment in gemmlir.tex lists, in order, what to drop if the built PDF
      runs long.

Two accuracy figures in the source are deliberately absent rather than
estimated: relative L2 for torchvision mobilenet_v2, mnasnet0_5 and
densenet121 was never recorded in the project log, so no number is claimed.
