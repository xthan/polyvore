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

    # Save the embedding size in the graph.
    tf.constant(self.config.embedding_size, name="embedding_size")
    self.image_embeddings = tf.reshape(image_embeddings,
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

      tf.contrib.losses.add_loss(emb_batch_loss * self.config.emb_loss_factor)
      
      total_loss = tf.contrib.losses.get_total_loss()
      
      # Add summaries.
      tf.scalar_summary("emb_batch_loss", emb_batch_loss)
      tf.scalar_summary("total_loss", total_loss)
      for var in tf.trainable_variables():
        tf.histogram_summary(var.op.name, var)
      
      weights = tf.to_float(tf.reshape(emb_loss_mask, [-1]))
    
      self.loss_mask = loss_mask
      self.input_mask = input_mask
      self.total_loss = total_loss
      self.emb_losses = emb_batch_loss  # Used in evaluation.
      
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
