#!/bin/bash

# name of the file
FILENAME="paper"
OUTPUT_DIR="."


# compile once
pdflatex -interaction=nonstopmode -output-directory $OUTPUT_DIR $FILENAME.tex
biber $FILENAME 

# compile twice for correct bibliography
pdflatex -interaction=nonstopmode -output-directory $OUTPUT_DIR $FILENAME.tex
pdflatex -interaction=nonstopmode -output-directory $OUTPUT_DIR $FILENAME.tex

# delete log files
find "$OUTPUT_DIR" -type f -name "$FILENAME.*" ! -name "$FILENAME.bib" ! -name "$FILENAME.tex" ! -name "$FILENAME.pdf" -delete

# open
if command -v xdg-open >/dev/null; then
    xdg-open $OUTPUT_DIR/$FILENAME.pdf  # Linux
elif command -v open >/dev/null; then
    open $OUTPUT_DIR/$FILENAME.pdf      # macOS
else
    echo "Consider using Unix :)"
fi

# commit & push
# git add -A
# git commit -m "AUTO-COMMIT $(date +"%Y-%m-%d %H:%M:%S %Z %z")"
# git push
# if [[ $? -eq 0 ]]; then
#     echo "Pushed successfully"
# else
#     echo "Not pushed"
# fi
