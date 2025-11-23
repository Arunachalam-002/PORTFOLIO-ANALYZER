#!/usr/bin/env bash
set -euo pipefail

echo "Starting Portfolio Analyzer..."

# If data files missing → try generating (runtime)
if [ ! -f "data/kite_instruments_list.json" ]; then
  echo "Instrument file missing. Trying to generate..."
  # run builder but don't fail the whole process if it errors out
  if python scripts/build_instruments.py; then
    echo "Instrument build succeeded."
  else
    echo "Instrument build failed; continuing startup (degraded mode)."
  fi
else
  echo "Instrument file already exists."
fi

# Start Gunicorn. Add timeout and logs so workers don't silently die.
exec gunicorn -w 4 -b 0.0.0.0:$PORT app:app --timeout 120 --access-logfile - --error-logfile -
