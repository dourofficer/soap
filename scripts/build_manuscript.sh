#!/usr/bin/env bash
# Compile the manuscript and leave ONLY the PDF, at artifacts/manuscript_preview.pdf.
# All aux files stay in a temp build directory; manuscript/ is left untouched.
# Needs the local TinyTeX (~/.TinyTeX); see the pdflatex/bibtex/pdflatex x2 chain.
set -euo pipefail

repo="$(cd "$(dirname "$0")/.." && pwd)"
ms="$repo/manuscript"
out="$repo/artifacts/manuscript_preview.pdf"

command -v pdflatex >/dev/null || export PATH="$HOME/.TinyTeX/bin/x86_64-linux:$PATH"

build="$(mktemp -d)"
trap 'rm -rf "$build"' EXIT
mkdir "$build/sections"    # \include{sections/appendix} writes its .aux here

tex() { (cd "$ms" && pdflatex -interaction=nonstopmode -output-directory "$build" main) \
          >"$build/pdflatex.out" 2>&1 || { tail -40 "$build/pdflatex.out"; exit 1; }; }

tex
(cd "$build" && BIBINPUTS="$ms" BSTINPUTS="$ms" bibtex main) > "$build/bibtex.out" 2>&1 \
  || { tail -20 "$build/bibtex.out"; exit 1; }
tex
tex

cp "$build/main.pdf" "$out"
grep -E "^Output written" "$build/pdflatex.out"
echo "-> ${out#$repo/}"
