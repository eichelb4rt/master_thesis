#!/bin/bash
# SHEBANG

file_stem="thesis.tex"
biblatex "$file_stem"
pdflatex --shell-escape "$file_stem" && pdflatex --shell-escape "$file_stem"