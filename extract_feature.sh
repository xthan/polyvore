#!/bin/bash
CHECKPOINT_DIR="model/model_final/model.ckpt-34865"

python polyvore/run_inference.py \
  --checkpoint_path=${CHECKPOINT_DIR} \
  --json_file="data/label/test_no_dup.json" \
  --image_dir="data/images/" \
  --feature_file="data/features/test_features.pkl" \
  --rnn_type="lstm"

# # Extract features of Bi-LSTM without VSE
# CHECKPOINT_DIR="model/model_final/model_bi_no_emb.ckpt"
# python polyvore/run_inference.py \
#   --checkpoint_path=${CHECKPOINT_DIR} \
#   --json_file="data/label/test_no_dup.json" \
#   --image_dir="data/images/" \
#   --feature_file="data/features/test_features_bi_no_emb.pkl" \
#   --rnn_type="lstm"


# # Extract features of VSE model without LSTM
# CHECKPOINT_DIR="model/model_final/model_emb.ckpt"
# python polyvore/run_inference_vse.py \
#   --checkpoint_path=${CHECKPOINT_DIR} \
#   --json_file="data/label/test_no_dup.json" \
#   --image_dir="data/images/" \
#   --feature_file="data/features/test_features_emb.pkl" \

# # Extract features of Siamese Network
# CHECKPOINT_DIR="model/model_final/model_siamese.ckpt"

# python polyvore/run_inference_siamese.py \
#   --checkpoint_path=${CHECKPOINT_DIR} \
#   --json_file="data/label/test_no_dup.json" \
#   --image_dir="data/images/" \
#   --feature_file="data/features/test_features_siamese.pkl"
