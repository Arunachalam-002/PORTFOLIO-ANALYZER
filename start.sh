#!/usr/bin/env bash
set -e

echo "Starting Portfolio Analyzer..."

# If data files missing → try generating
if [ ! -f "data/kite_instruments_list.json" ]; then
  echo "Instrument file missing. Trying to generate..."
  python scripts/build_instruments.py || echo "Instrument build failed, continuing..."
else
  echo "Instrument file already exists."
fi

exec gunicorn -w 4 -b 0.0.0.0:$PORT app:app
