#!/usr/bin/env bash
set -e

# If a kaggle.json file has been mounted, move it to the location expected
KAGGLE_SRC="/opt/kaggle.json"
KAGGLE_DST="$HOME/.config/kaggle/kaggle.json"
if [ -f "$KAGGLE_SRC" ]; then
  mkdir -p "$(dirname "$KAGGLE_DST")"
  cp "$KAGGLE_SRC" "$KAGGLE_DST"
  chmod 600 "$KAGGLE_DST"
fi

exec jupyter lab \
      --ip=0.0.0.0 --port=8888 \
      --no-browser --NotebookApp.token='' --NotebookApp.password=''
