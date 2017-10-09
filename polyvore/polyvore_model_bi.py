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

"""
Polyvore model used in ACM MM"17 paper
"Learning Fashion Compatibility with Bidirectional LSTMs"
Link: https://arxiv.org/pdf/1707.05691.pdf
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np                                  
import tensorflow as tf

from ops import image_embedding
from ops import image_processing
from ops import inputs as input_ops

class PolyvoreModel(object):
  """ Model for fashion set on Polyvore dataset.
  """

  def __init__(self, config, mode, train_inception=False):
    """Basic setup.

    Args:
      config: Object containing configuration parameters.
      mode: "train", "eval" or "inference".
      train_inception: Whether the inception submodel variables are trainable.
    """
    assert mode in ["train", "eval", "inference"]
    self.config = config
    self.mode = mode
    self.train_inception = train_inception

    # Reader for the input data.
    self.reader = tf.TFRecordReader()

    # To match the "Show and Tell" paper we initialize all variables with a
    # random uniform initializer.
    self.initializer = tf.random_uniform_initializer(
        minval=-self.config.initializer_scale,
        maxval=self.config.initializer_scale)

    # A float32 Tensor with shape
    # [batch_size, num_images, height, width, channels].
    # num_images is the number of images in one outfit, default is 8.
    self.images = None

    # Forward RNN input and target sequences.
    # An int32 Tensor with shape [batch_size, padded_length].
    self.f_input_seqs = None
    # An int32 Tensor with shape [batch_size, padded_length].
    self.f_target_seqs = None
    
    # Backward RNN input and target sequences.
    # An int32 Tensor with shape [batch_size, padded_length].
    self.b_input_seqs = None
    # An int32 Tensor with shape [batch_size, padded_length].
    self.b_target_seqs = None
    
    # An int32 0/1 Tensor with shape [batch_size, padded_length].
    self.input_mask = None
  
    # Image caption sequence and masks.
    # An int32 Tensor with shape [batch_size, num_images, padded_length].
    self.cap_seqs = None
    # An int32 0/1 Tensor with shape [batch_size, padded_length].
    self.cap_mask = None

    # Caption sequence embeddings, we use simple bag of word model.
    # A float32 Tensor with shape [batch_size, num_images, embedding_size].
    self.seq_embeddings = None

    # Image embeddings in the joint visual-semantic space
    # A float32 Tensor with shape [batch_size, num_images, embedding_size].
    self.image_embeddings = None

    # Image embeddings in the RNN output/prediction space.
    self.rnn_image_embeddings = None
    
    # Word embedding map.
    self.embedding_map = None

    # A float32 scalar Tensor; the total loss for the trainer to optimize.
    self.total_loss = None

    # Forward and backward RNN loss.
    # A float32 Tensor with shape [batch_size * padded_length].
    self.forward_losses = None
    # A float32 Tensor with shape [batch_size * padded_length].
    self.backward_losses = None
    # RNN loss, forward + backward.
    self.lstm_losses = None
    
    # Loss mask for lstm loss.
    self.loss_mask = None

    # Visual Semantic Embedding loss.
    # A float32 Tensor with shape [batch_size * padded_length].
    self.emb_losses = None
    
    # A float32 Tensor with shape [batch_size * padded_length].
    self.target_weights = None

    # Collection of variables from the inception submodel.
    self.inception_variables = []

    # Function to restore the inception submodel from checkpoint.
    self.init_fn = None

    # Global step Tensor.
    self.global_step = None
    
    # Some output for debugging purposes .
    self.target_embeddings = None
    self.input_embeddings = None
    self.set_ids = None
    self.f_lstm_state = None
    self.b_lstm_state = None
    self.lstm_output = None
    self.lstm_xent_loss = None


  def is_training(self):
    """Returns true if the model is built for training mode."""
    return self.mode == "train"

  def process_image(self, encoded_image, thread_id=0, image_idx=0):
    """Decodes and processes an image string.

    Args:
      encoded_image: A scalar string Tensor; the encoded image.
      thread_id: Preprocessing thread id used to select the ordering of color
        distortions. Not used in our model.
      image_idx: Index of the image in an outfit. Only used for summaries.
    Returns:
      A float32 Tensor of shape [height, width, 3]; the processed image.
    """
    return image_processing.process_image(encoded_image,
                                          is_training=self.is_training(),
                                          height=self.config.image_height,
                                          width=self.config.image_width,
                                          image_format=self.config.image_format,
                                          image_idx=image_idx)

  def build_inputs(self):
    """Input prefetching, preprocessing and batching.

    Outputs:
      Inputs of the model.
    """
    if self.mode == "inference":
      # In inference mode, images and inputs are fed via placeholders.
      image_feed = tf.placeholder(dtype=tf.string, shape=[], name="image_feed")
      # Process image and insert batch dimensions.
      image_feed = self.process_image(image_feed)
      
      input_feed = tf.placeholder(dtype=tf.int64,
                                  shape=[None],  # batch_size
                                  name="input_feed")

      # Process image and insert batch dimensions.
      image_seqs = tf.expand_dims(image_feed, 0)
      cap_seqs = tf.expand_dims(input_feed, 1)

      # No target sequences or input mask in inference mode.
      input_mask = tf.placeholder(dtype=tf.int64,
                                  shape=[1, 8],  # batch_size
                                  name="input_mask")
      cap_mask = None
      loss_mask = None
      set_ids = None
      
    else:
      # Prefetch serialized SequenceExample protos.
      input_queue = input_ops.prefetch_input_data(
          self.reader,
          self.config.input_file_pattern,
          is_training=self.is_training(),
          batch_size=self.config.batch_size,
          values_per_shard=self.config.values_per_input_shard,
          input_queue_capacity_factor=self.config.input_queue_capacity_factor,
          num_reader_threads=self.config.num_input_reader_threads)

      # Image processing and random distortion. Split across multiple threads
      # with each thread applying a slightly different distortion. But we only
      # use one thread in our Polyvore model. likes are not used.
      images_and_captions = []
      for thread_id in range(self.config.num_preprocess_threads):
        serialized_sequence_example = input_queue.dequeue()
        set_id, encoded_images, image_ids, captions, likes = (
            input_ops.parse_sequence_example(
            serialized_sequence_example,
            set_id =self.config.set_id_name,
            image_feature=self.config.image_feature_name,
            image_index=self.config.image_index_name,
            caption_feature=self.config.caption_feature_name,
            number_set_images=self.config.number_set_images))
        
        images = []
        for i in range(self.config.number_set_images):
          images.append(self.process_image(encoded_images[i],image_idx=i))
        
        images_and_captions.append([set_id, images, image_ids, captions, likes])

      # Batch inputs.
      queue_capacity = (5 * self.config.num_preprocess_threads *
                        self.config.batch_size)

      (set_ids, image_seqs, image_ids, input_mask,
       loss_mask, cap_seqs, cap_mask, likes) = (
       input_ops.batch_with_dynamic_pad(images_and_captions,
                                           batch_size=self.config.batch_size,
                                           queue_capacity=queue_capacity))
    
    self.images = image_seqs
    self.input_mask = input_mask
    self.loss_mask = loss_mask
    self.cap_seqs = cap_seqs
    self.cap_mask = cap_mask
    self.set_ids = set_ids

  def build_image_embeddings(self):
    """Builds the image model subgraph and generates image embeddings
      in visual semantic joint space and RNN prediction space.

    Inputs:
      self.images

    Outputs:
      self.image_embeddings
      self.rnn_image_embeddings
    """
    
    # Reshape 5D image tensor.
    images = tf.reshape(self.images, [-1,
                                 self.config.image_height,
                                 self.config.image_height,
                                 3])
    
    inception_output = image_embedding.inception_v3(
        images,
        trainable=self.train_inception,
        is_training=self.is_training())
    self.inception_variables = tf.get_collection(
        tf.GraphKeys.VARIABLES, scope="InceptionV3")
    
    # Map inception output into embedding space.
    with tf.variable_scope("image_embedding") as scope:
      image_embeddings = tf.contrib.layers.fully_connected(
          inputs=inception_output,
          num_outputs=self.config.embedding_size,
          activation_fn=None,
          weights_initializer=self.initializer,
          biases_initializer=None,
          scope=scope)
    
    with tf.variable_scope("rnn_image_embedding") as scope:
      rnn_image_embeddings = tf.contrib.layers.fully_connected(
          inputs=inception_output,
          num_outputs=self.config.embedding_size,
          activation_fn=None,
          weights_initializer=self.initializer,
          biases_initializer=None,
          scope=scope)

    # Save the embedding size in the graph.
    tf.constant(self.config.embedding_size, name="embedding_size")
    self.image_embeddings = tf.reshape(image_embeddings,
                                       [tf.shape(self.images)[0],
                                        -1,
                                        self.config.embedding_size])

    self.rnn_image_embeddings = tf.reshape(rnn_image_embeddings,
                                           [tf.shape(self.images)[0],
                                            -1,
                                            self.config.embedding_size])

  def build_seq_embeddings(self):
    """Builds the input sequence embeddings.

    Inputs:
      self.input_seqs

    Outputs:
      self.seq_embeddings
      self.embedding_map
    """
    with tf.variable_scope("seq_embedding"), tf.device("/cpu:0"):
      embedding_map = tf.get_variable(
          name="map",
          shape=[self.config.vocab_size, self.config.embedding_size],
          initializer=self.initializer)
      seq_embeddings = tf.nn.embedding_lookup(embedding_map, self.cap_seqs)
      
      # Average pooling the seq_embeddings (bag of words). 
      if self.mode != "inference":
        seq_embeddings = tf.batch_matmul(
                                tf.cast(tf.expand_dims(self.cap_mask, 2),
                                        tf.float32),
                                seq_embeddings)
        seq_embeddings = tf.squeeze(seq_embeddings, [2])
    
    self.embedding_map = embedding_map
    self.seq_embeddings = seq_embeddings

  def build_model(self):
    """Builds the model.
      The original code is written with Tensorflow r0.10
      for Tensorflow > r1.0, many functions can be simplified.
      For example Tensors support slicing now, so no need to use tf.slice()
    """
    norm_image_embeddings = tf.nn.l2_normalize(self.image_embeddings, 2,
                                               name="norm_image_embeddings")
    norm_seq_embeddings = tf.nn.l2_normalize(self.seq_embeddings, 2)
    
    norm_seq_embeddings = (
        tf.pad(norm_seq_embeddings, [[0, 0],
               [0, self.config.number_set_images - tf.shape(norm_seq_embeddings)[1]],
               [0, 0]], name="norm_seq_embeddings"))
    
    if self.mode == "inference":
      pass
    else:
      # Compute losses for joint embedding.
      # Only look at the captions that have length >= 2.
      emb_loss_mask = tf.greater(tf.reduce_sum(self.cap_mask, 2), 1)
      # Image mask is padded it to max length.
      emb_loss_mask = tf.pad(emb_loss_mask,
          [[0,0],
           [0, self.config.number_set_images - tf.shape(emb_loss_mask)[1]]])
      
      # Select the valid image-caption pair.
      emb_loss_mask = tf.reshape(emb_loss_mask, [-1])
      norm_image_embeddings = tf.reshape(norm_image_embeddings,
          [self.config.number_set_images * self.config.batch_size,
           self.config.embedding_size])
      norm_image_embeddings = tf.boolean_mask(norm_image_embeddings,
                                                emb_loss_mask)
      norm_seq_embeddings = tf.reshape(norm_seq_embeddings,
                [self.config.number_set_images * self.config.batch_size,
                 self.config.embedding_size])

      norm_seq_embeddings = tf.boolean_mask(norm_seq_embeddings, emb_loss_mask)

      # The following defines contrastive loss in the joint space.   
      # Reference: https://github.com/ryankiros/visual-semantic-embedding/blob/master/model.py#L39
      scores = tf.matmul(norm_seq_embeddings, norm_image_embeddings,
                         transpose_a=False, transpose_b=True, name="scores")
      
      diagonal = tf.expand_dims(tf.diag_part(scores), 1)
      cost_s = tf.maximum(0.0, self.config.emb_margin - diagonal + scores)
      cost_im = tf.maximum(0.0,
          self.config.emb_margin - tf.transpose(diagonal) + scores)
      cost_s = cost_s - tf.diag(tf.diag_part(cost_s))
      cost_im = cost_im - tf.diag(tf.diag_part(cost_im))
      
      emb_batch_loss = tf.reduce_sum(cost_s) + tf.reduce_sum(cost_im)
      emb_batch_loss = (emb_batch_loss /
              tf.cast(tf.shape(norm_seq_embeddings)[0], tf.float32) ** 2)

      if self.config.emb_loss_factor > 0.0:
        tf.contrib.losses.add_loss(emb_batch_loss * self.config.emb_loss_factor)
      
    # Compute image LSTM loss.
    # Start with one direction.
    tf.logging.info("Rnn_type: %s" % self.config.rnn_type)
    if self.config.rnn_type == "lstm":
      tf.logging.info("----- RNN Type: LSTM ------")
      # Forward LSTM.
      f_lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(
        num_units=self.config.num_lstm_units, state_is_tuple=True)
      # Backward LSTM.
      b_lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(
        num_units=self.config.num_lstm_units, state_is_tuple=True)
    elif self.config.rnn_type == "gru":
      tf.logging.info("----- RNN Type: GRU ------")
      # Forward GRU.
      f_lstm_cell = tf.nn.rnn_cell.GRUCell(num_units=self.config.num_lstm_units)
      # Backward GRU.
      b_lstm_cell = tf.nn.rnn_cell.GRUCell(num_units=self.config.num_lstm_units)
    else:
      tf.logging.info("----- RNN Type: RNN ------")
      # Forward RNN.
      f_lstm_cell = tf.nn.rnn_cell.BasicRNNCell(num_units=self.config.num_lstm_units)
      # Backward RNN.
      b_lstm_cell = tf.nn.rnn_cell.BasicRNNCell(num_units=self.config.num_lstm_units)
   
    if self.mode == "train":
      f_lstm_cell = tf.nn.rnn_cell.DropoutWrapper(
          f_lstm_cell,
          input_keep_prob=self.config.lstm_dropout_keep_prob,
          output_keep_prob=self.config.lstm_dropout_keep_prob)
      b_lstm_cell = tf.nn.rnn_cell.DropoutWrapper(
          b_lstm_cell,
          input_keep_prob=self.config.lstm_dropout_keep_prob,
          output_keep_prob=self.config.lstm_dropout_keep_prob)

    with tf.variable_scope("lstm", initializer=self.initializer) as lstm_scope:
      if self.mode == "inference":
        # Inference for Bi-LSTM.
        pred_feed = tf.placeholder(dtype=tf.float32,
                                   shape=[None, None],
                                   name="pred_feed")
        next_index_feed = tf.placeholder(dtype=tf.int64,
                                   shape=[None],
                                   name="next_index_feed")
        
        self.lstm_xent_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                                    logits=pred_feed,
                                    labels=next_index_feed,
                                    name="lstm_xent")

                    
        if self.config.rnn_type == "lstm":
          # In inference mode, use concatenated states for convenient feeding
          # and fetching.
          # Forward
          # Placeholder for feeding a batch of concatenated states.
          f_state_feed = tf.placeholder(dtype=tf.float32,
                                    shape=[None, sum(f_lstm_cell.state_size)],
                                    name="f_state_feed")
          f_input_feed = tf.placeholder(dtype=tf.float32,
                                    shape=[None, self.config.embedding_size],
                                    name="f_input_feed")
          # Backward:
          # Placeholder for feeding a batch of concatenated states.
          b_state_feed = tf.placeholder(dtype=tf.float32,
                                    shape=[None, sum(b_lstm_cell.state_size)],
                                    name="b_state_feed")
          b_input_feed = tf.placeholder(dtype=tf.float32,
                                    shape=[None, self.config.embedding_size],
                                    name="b_input_feed")
                                        
          f_state_tuple = tf.split(1, 2, f_state_feed)
          # Run a single LSTM step.
          with tf.variable_scope("FW"):
            f_lstm_outputs, f_state_tuple = f_lstm_cell(
                                              inputs=f_input_feed,
                                              state=f_state_tuple)
          # Concatentate the resulting state.
          self.f_lstm_state = tf.concat(1, f_state_tuple, name="f_state")

          b_state_tuple = tf.split(1, 2, b_state_feed)

          # Run a single LSTM step.
          with tf.variable_scope("BW"):
            b_lstm_outputs, b_state_tuple = b_lstm_cell(
                                              inputs=b_input_feed,
                                              state=b_state_tuple)
          # Concatentate the resulting state.
          self.b_lstm_state = tf.concat(1, b_state_tuple, name="b_state")
          
        else:
          # For non-LSTM RNN models, no tuple is used.
          # Forward
          # Placeholder for feeding a batch of concatenated states.
          f_state_feed = tf.placeholder(dtype=tf.float32,
                                    shape=[None, f_lstm_cell.state_size],
                                    name="f_state_feed")
          f_input_feed = tf.placeholder(dtype=tf.float32,
                                    shape=[None, self.config.embedding_size],
                                    name="f_input_feed")
          # Backward:
          # Placeholder for feeding a batch of concatenated states.
          b_state_feed = tf.placeholder(dtype=tf.float32,
                                    shape=[None, b_lstm_cell.state_size],
                                    name="b_state_feed")
          b_input_feed = tf.placeholder(dtype=tf.float32,
                                    shape=[None, self.config.embedding_size],
                                    name="b_input_feed")
          # Run a single RNN step.
          with tf.variable_scope("FW"):
            f_lstm_outputs, f_state_tuple = f_lstm_cell(
                                              inputs=f_input_feed,
                                              state=f_state_feed)
          f_state_tuple = tf.identity(f_state_tuple, name="f_state")
            
          with tf.variable_scope("BW"):
            b_lstm_outputs, b_state_tuple = b_lstm_cell(
                                              inputs=b_input_feed,
                                              state=b_state_feed)
          b_state_tuple = tf.identity(b_state_tuple, name="b_state")
            
        lstm_outputs = (f_lstm_outputs, b_lstm_outputs)
        sequence_length = None
      else:
        # Run the batch of sequence embeddings through the LSTM.
        sequence_length = tf.reduce_sum(self.input_mask, 1)
        lstm_outputs, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw=f_lstm_cell,
                                            cell_bw=b_lstm_cell,
                                            inputs=self.rnn_image_embeddings,
                                            initial_state_fw=None,
                                            initial_state_bw=None,
                                            sequence_length=sequence_length,
                                            dtype=tf.float32,
                                            scope=lstm_scope)

    # Stack batches vertically.
    f_lstm_outputs = tf.reshape(lstm_outputs[0], [-1, f_lstm_cell.output_size])
    if self.mode == "inference":
      b_lstm_outputs = lstm_outputs[1]
    else:
      b_lstm_outputs = tf.reverse_sequence(lstm_outputs[1],
                                           seq_lengths=sequence_length,
                                           seq_dim=1,
                                           batch_dim=0)
    
    b_lstm_outputs = tf.reshape(b_lstm_outputs, [-1, b_lstm_cell.output_size])
    with tf.variable_scope("f_logits") as logits_scope:
      f_input_embeddings = tf.contrib.layers.fully_connected(
          inputs=f_lstm_outputs,
          num_outputs=self.config.embedding_size,
          activation_fn=None,
          weights_initializer=self.initializer,
          scope=logits_scope)
         
    with tf.variable_scope("b_logits") as logits_scope:
      b_input_embeddings = tf.contrib.layers.fully_connected(
          inputs=b_lstm_outputs,
          num_outputs=self.config.embedding_size,
          activation_fn=None,
          weights_initializer=self.initializer,
          scope=logits_scope)
    
    if self.mode == "inference":
      pass
    else:
      # Padding input_mask to match dimension.
      input_mask = tf.pad(self.input_mask,
        [[0,0],
        [0, self.config.number_set_images + 1 - tf.shape(self.input_mask)[1]]])
      input_mask = tf.to_float(
          tf.reshape(tf.slice(input_mask, [0,1], [-1, -1]), [-1,1]))
      loss_mask = tf.pad(self.loss_mask,
        [[0,0],
         [0, self.config.number_set_images - tf.shape(self.loss_mask)[1]]])
      loss_mask = tf.reshape(tf.to_float(loss_mask),
                    [self.config.number_set_images * self.config.batch_size,1])
      
      # Forward rnn.
      f_target_embeddings = tf.slice(tf.pad(self.rnn_image_embeddings,
              [[0,0], [0,1], [0,0]]), [0,1,0], [-1,-1,-1])
      f_target_embeddings = tf.reshape(f_target_embeddings,
              [self.config.number_set_images * self.config.batch_size,
               self.config.embedding_size])
      f_target_embeddings = tf.mul(f_target_embeddings,
                                        input_mask,
                                        name="target_embeddings")
      
      # Softmax loss over all items in this minibatch.
      loss_mask = tf.squeeze(loss_mask)
      f_input_embeddings = tf.boolean_mask(f_input_embeddings,
                                           tf.cast(loss_mask, tf.bool))
      f_target_embeddings = tf.boolean_mask(f_target_embeddings,
                                            tf.cast(loss_mask, tf.bool))
      
      f_lstm_scores = tf.matmul(f_input_embeddings,
                                f_target_embeddings,
                                transpose_a=False,
                                transpose_b=True)
      f_lstm_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                        logits=f_lstm_scores,
                        labels=tf.range(tf.shape(f_lstm_scores)[0]))
      f_lstm_loss = tf.div(tf.reduce_sum(f_lstm_loss),
                           tf.reduce_sum(loss_mask),
                           name="f_lstm_loss")
      
      # Backward rnn.
      # It would be better to put write a function to calcute lstm_loss from
      # loss_mask, inputs, and targets, so the code can be reused, for now
      # just copy and paste the forward to get the backward loss.  
      reverse_embeddings = tf.reverse_sequence(self.rnn_image_embeddings,
                                               seq_lengths=sequence_length,
                                               seq_dim=1,
                                               batch_dim=0)
      b_target_embeddings = tf.slice(tf.pad(reverse_embeddings,
                                            [[0,0], [0,1], [0,0]]),
                                     [0,1,0], [-1,-1,-1])
      b_target_embeddings = tf.reshape(b_target_embeddings,
            [self.config.number_set_images * self.config.batch_size,
             self.config.embedding_size])
      b_target_embeddings = tf.mul(b_target_embeddings,
                                        input_mask,
                                        name="target_embeddings")
      
      # Softmax loss over all items in this minibatch
      b_input_embeddings = tf.boolean_mask(b_input_embeddings,
                                           tf.cast(loss_mask, tf.bool))
      b_target_embeddings = tf.boolean_mask(b_target_embeddings,
                                            tf.cast(loss_mask, tf.bool))
      
      b_lstm_scores = tf.matmul(b_input_embeddings,
                                b_target_embeddings,
                                transpose_a=False,
                                transpose_b=True)
      b_lstm_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
              logits=b_lstm_scores,
              labels=tf.range(tf.shape(b_lstm_scores)[0]))
      b_lstm_loss = tf.div(tf.reduce_sum(b_lstm_loss),
                           tf.reduce_sum(loss_mask),
                           name="b_lstm_loss")
      
      if self.config.f_rnn_loss_factor > 0:
        tf.contrib.losses.add_loss(f_lstm_loss * self.config.f_rnn_loss_factor)
      if self.config.b_rnn_loss_factor > 0:
        tf.contrib.losses.add_loss(b_lstm_loss * self.config.b_rnn_loss_factor)
     
      # Merge all losses and stats.
      total_loss = tf.contrib.losses.get_total_loss()
      
      # Add summaries.
      tf.scalar_summary("emb_batch_loss", emb_batch_loss)
      tf.scalar_summary("f_lstm_loss", f_lstm_loss)
      tf.scalar_summary("b_lstm_loss", b_lstm_loss)
      tf.scalar_summary("lstm_loss",
            (f_lstm_loss * self.config.f_rnn_loss_factor +
             b_lstm_loss * self.config.b_rnn_loss_factor))
      tf.scalar_summary("total_loss", total_loss)
      for var in tf.trainable_variables():
        tf.histogram_summary(var.op.name, var)
      
      weights = tf.to_float(tf.reshape(emb_loss_mask, [-1]))
    
      self.loss_mask = loss_mask
      self.input_mask = input_mask
      self.target_embeddings = (f_target_embeddings, b_target_embeddings)
      self.input_embeddings = (f_input_embeddings, b_input_embeddings)
      self.total_loss = total_loss
      self.emb_losses = emb_batch_loss  # Used in evaluation.
      self.lstm_losses = (f_lstm_loss * self.config.f_rnn_loss_factor +
             b_lstm_loss * self.config.b_rnn_loss_factor) # Used in evaluation.
      self.target_weights = weights  # Used in evaluation.
      
  def setup_inception_initializer(self):
    """Sets up the function to restore inception variables from checkpoint."""
    if self.mode != "inference":
      # Restore inception variables only.
      saver = tf.train.Saver(self.inception_variables)

      def restore_fn(sess):
        tf.logging.info("Restoring Inception variables from checkpoint %s" %
                        self.config.inception_checkpoint_file)
        saver.restore(sess, self.config.inception_checkpoint_file)

      self.init_fn = restore_fn

  def setup_global_step(self):
    """Sets up the global step Tensor."""
    global_step = tf.Variable(
        initial_value=0,
        name="global_step",
        trainable=False,
        collections=[tf.GraphKeys.GLOBAL_STEP, tf.GraphKeys.VARIABLES])

    self.global_step = global_step

  def build(self):
    """Creates all ops for training and evaluation."""
    self.build_inputs()
    self.build_image_embeddings()
    self.build_seq_embeddings()
    self.build_model()
    self.setup_inception_initializer()
    self.setup_global_step()
