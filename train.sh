#!/bin/bash

# Inception v3 checkpoint file.
INCEPTION_CHECKPOINT="model/inception_v3.ckpt"

# Directory to save the model.
MODEL_DIR="model/bi_lstm/"

# Run the training code.
python polyvore/train.py \
  --input_file_pattern="data/tf_records/train-no-dup-?????-of-00128" \
  --inception_checkpoint_file="${INCEPTION_CHECKPOINT}" \
  --train_dir="${MODEL_DIR}/train" \
  --train_inception=true \
  --number_of_steps=100000