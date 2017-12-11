# Copyright 2017 Xintong Han. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Bi-LSTM Polyvore model and training configurations."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


class ModelConfig(object):
  """Wrapper class for model hyperparameters."""

  def __init__(self):
    """Sets the default model hyperparameters."""
    # File pattern of sharded TFRecord file containing SequenceExample protos.
    # Must be provided in training and evaluation modes.
    self.input_file_pattern = None

    # Image format ("jpeg" or "png").
    self.image_format = "jpeg"

    # Approximate number of values per input shard. Used to ensure sufficient
    # mixing between shards in training.
    self.values_per_input_shard = 135
    # Minimum number of shards to keep in the input queue.
    self.input_queue_capacity_factor = 2
    # Number of threads for prefetching SequenceExample protos.
    self.num_input_reader_threads = 1
  
    # Name of the SequenceExample context feature containing set ids.
    self.set_id_name = "set_id"
    
    # Name of the SequenceExample feature list containing captions and images.
    self.image_feature_name = "images"
    self.image_index_name = "image_index"
    self.caption_feature_name = "caption_ids"

    # Number of unique words in the vocab (plus 1, for <UNK>).
    # The default value is larger than the expected actual vocab size to allow
    # for differences between tokenizer versions used in preprocessing. There is
    # no harm in using a value greater than the actual vocab size, but using a
    # value less than the actual vocab size will result in an error.
    self.vocab_size = 2757

    # Number of threads for image preprocessing.
    self.num_preprocess_threads = 1

    # Batch size.
    self.batch_size = 10
    
    # File containing an Inception v3 checkpoint to initialize the variables
    # of the Inception model. Must be provided when starting training for the
    # first time.
    self.inception_checkpoint_file = None

    # Dimensions of Inception v3 input images.
    self.image_height = 299
    self.image_width = 299

    # Scale used to initialize model variables.
    self.initializer_scale = 0.08

    # LSTM input and output dimensionality, respectively. embedding_size is also
    # the embedding size in the visual-semantic joint space.
    self.embedding_size = 512 
    self.num_lstm_units = 512 

    # If < 1.0, the dropout keep probability applied to LSTM variables.
    self.lstm_dropout_keep_prob = 0.7

    # Largest number of images in a fashion set.
    self.number_set_images = 8
    
    # Margin for the embedding loss.
    self.emb_margin = 0.2

    # Balance factor of all losses.
    self.emb_loss_factor = 1.0 # VSE loss
    self.f_rnn_loss_factor = 1.0  # Forward LSTM
    self.b_rnn_loss_factor = 1.0  # Backward LSTM, might give it a lower weight
    # because it is harder to predict backward than forward in our senario.
    
    # RNN type. "lstm", "gru", "rnn"
    self.rnn_type = "lstm"


class TrainingConfig(object):
  """Wrapper class for training hyperparameters."""

  def __init__(self):
    """Sets the default training hyperparameters."""
    # Number of examples per epoch of training data.
    self.num_examples_per_epoch = 17316

    # Optimizer for training the model.
    self.optimizer = "SGD"

    # Learning rate for the initial phase of training.
    # by the FLAGS in train.py
    self.initial_learning_rate = 0.2
    
    self.learning_rate_decay_factor = 0.5
    self.num_epochs_per_decay = 2.0

    # If not None, clip gradients to this value.
    self.clip_gradients = 5.0

    # How many model checkpoints to keep.
    self.max_checkpoints_to_keep = 10
