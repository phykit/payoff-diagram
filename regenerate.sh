#!/usr/bin/env bash
# One command: validate the quoted numbers, regenerate the seven figures, typeset the article.
set -e
python validate.py
python figs_v2.py
pdflatex -interaction=nonstopmode article_v2.tex >/dev/null && pdflatex -interaction=nonstopmode article_v2.tex >/dev/null
echo "built article_v2.pdf"
