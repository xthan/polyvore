#!/bin/bash
CHECKPOINT_DIR="model/model_final/model.ckpt-34865"

python polyvore/fashion_compatibility.py \
  --checkpoint_path=${CHECKPOINT_DIR} \
  --label_file="data/label/fashion_compatibility_prediction.txt" \
  --feature_file="data/features/test_features.pkl" \
  --rnn_type="lstm" \
  --direction="2" \
  --result_file="fashion_compatibility.pkl"
