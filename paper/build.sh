#!/bin/bash
# SHEBANG

file_stem="thesis"
bibtex "$file_stem"
pdflatex --shell-escape "$file_stem" && pdflatex --shell-escape "$file_stem"