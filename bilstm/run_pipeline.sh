#!/usr/bin/env bash
# Quick pipeline: augment -> train -> evaluate -> ensemble
set -euo pipefail
cd "$(dirname "$0")"
source venv/bin/activate

echo "=== 1/4 Augment minority classes ==="
python augment_minority.py

echo "=== 2/4 Train (CSV_PATH should be data/augmented_train.csv) ==="
python train.py

echo "=== 3/4 Evaluate best checkpoint ==="
python evaluate.py

echo "=== 4/4 Ensemble top-3 checkpoints ==="
NUM_WORKERS=0 python evaluate_ensemble.py

echo "Done."
