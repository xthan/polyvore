#!/bin/bash
CHECKPOINT_DIR="model/model_final/model.ckpt-34865"

python polyvore/run_inference.py \
  --checkpoint_path=${CHECKPOINT_DIR} \
  --json_file="data/label/test_no_dup.json" \
  --image_dir="data/images/test_no_dup/" \
  --feature_file="data/features/test_features.pkl" \
  --rnn_type="lstm"
