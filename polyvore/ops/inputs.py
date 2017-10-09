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

"""Input ops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf


def parse_sequence_example(serialized, set_id, image_feature,
                           image_index, caption_feature, number_set_images):
  """Parses a tensorflow.SequenceExample into a set of images and caption.

  Args:
    serialized: A scalar string Tensor; a single serialized SequenceExample.
    set_id: Name of SequenceExample context feature containing the id of
      the outfit.
    image_feature: Name of SequenceExample context feature containing image
      data.
    image_index: Name of SequenceExample feature list containing the index of
      the item in the outfit.
    caption_feature: Name of SequenceExample feature list containing integer
      captions.
    number_set_images: Number of images in an outfit.
  Returns:
    set_id: Set id of the outfit.
    encoded_images: A string Tensor containing all JPEG encoded images
      in the outfit.
    image_ids: Image ids of the items in the outfit.
    captions: A 2-D uint64 Tensor with dynamically specified length.
    likes: Number of likes of the outfit. Hard coded name,
      not used in our model.
  """
  
  context_features = {}
  context_features[set_id] = tf.FixedLenFeature([], dtype=tf.string)
  context_features['likes'] = tf.FixedLenFeature([], dtype=tf.int64,
                                                 default_value=0)
  for i in range(number_set_images):
    context_features[image_feature + '/' + str(i)] = tf.FixedLenFeature([],
                                                         dtype=tf.string,
                                                         default_value = '')
            
  context, sequence = tf.parse_single_sequence_example(
      serialized,
      context_features=context_features,
      sequence_features={
          image_index: tf.FixedLenSequenceFeature([], dtype=tf.int64),
          caption_feature: tf.VarLenFeature(dtype=tf.int64),
      })
      
  set_id = context[set_id]
  likes = context['likes']
  
  encoded_images = []
  for i in range(number_set_images):
    encoded_images.append(context[image_feature + '/' + str(i)])
  
  captions = sequence[caption_feature]
  captions = tf.sparse_tensor_to_dense(captions)
  image_ids = sequence[image_index]
  
  return set_id, encoded_images, image_ids, captions, likes


def prefetch_input_data(reader,
                        file_pattern,
                        is_training,
                        batch_size,
                        values_per_shard,
                        input_queue_capacity_factor=16,
                        num_reader_threads=1,
                        shard_queue_name="filename_queue",
                        value_queue_name="input_queue"):
  """Prefetches string values from disk into an input queue.

  In training the capacity of the queue is important because a larger queue
  means better mixing of training examples between shards. The minimum number of
  values kept in the queue is values_per_shard * input_queue_capacity_factor,
  where input_queue_memory factor should be chosen to trade-off better mixing
  with memory usage.

  Args:
    reader: Instance of tf.ReaderBase.
    file_pattern: Comma-separated list of file patterns (e.g.
        /tmp/train_data-?????-of-00100).
    is_training: Boolean; whether prefetching for training or eval.
    batch_size: Model batch size used to determine queue capacity.
    values_per_shard: Approximate number of values per shard.
    input_queue_capacity_factor: Minimum number of values to keep in the queue
      in multiples of values_per_shard. See comments above.
    num_reader_threads: Number of reader threads to fill the queue.
    shard_queue_name: Name for the shards filename queue.
    value_queue_name: Name for the values input queue.

  Returns:
    A Queue containing prefetched string values.
  """
  data_files = []
  for pattern in file_pattern.split(","):
    data_files.extend(tf.gfile.Glob(pattern))
  if not data_files:
    tf.logging.fatal("Found no input files matching %s", file_pattern)
  else:
    tf.logging.info("Prefetching values from %d files matching %s",
                    len(data_files), file_pattern)

  if is_training:
    filename_queue = tf.train.string_input_producer(
        data_files, shuffle=True, capacity=16, name=shard_queue_name)
    min_queue_examples = values_per_shard * input_queue_capacity_factor
    capacity = min_queue_examples + 100 * batch_size
    values_queue = tf.RandomShuffleQueue(
        capacity=capacity,
        min_after_dequeue=min_queue_examples,
        dtypes=[tf.string],
        name="random_" + value_queue_name)
  else:
    filename_queue = tf.train.string_input_producer(
        data_files, shuffle=False, capacity=1, name=shard_queue_name)
    capacity = values_per_shard + 3 * batch_size
    values_queue = tf.FIFOQueue(
        capacity=capacity, dtypes=[tf.string], name="fifo_" + value_queue_name)

  enqueue_ops = []
  for _ in range(num_reader_threads):
    _, value = reader.read(filename_queue)
    enqueue_ops.append(values_queue.enqueue([value]))
  tf.train.queue_runner.add_queue_runner(tf.train.queue_runner.QueueRunner(
      values_queue, enqueue_ops))
  tf.summary.scalar(
      "queue/%s/fraction_of_%d_full" % (values_queue.name, capacity),
      tf.cast(values_queue.size(), tf.float32) * (1. / capacity))

  return values_queue


def batch_with_dynamic_pad(images_and_captions,
                           batch_size,
                           queue_capacity,
                           add_summaries=True):
  """Batches input images and captions.

  This function splits the caption into an input sequence and a target sequence,
  where the target sequence is the input sequence right-shifted by 1. Input and
  target sequences are batched and padded up to the maximum length of sequences
  in the batch. A mask is created to distinguish real words from padding words.
  Similar sequence processing is used for images in an outfit.
  Example:
    Actual captions in the batch ('-' denotes padded character):
      [
        [ 1 2 5 4 5 ],
        [ 1 2 3 4 - ],
        [ 1 2 3 - - ],
      ]

    input_seqs:
      [
        [ 1 2 3 4 ],
        [ 1 2 3 - ],
        [ 1 2 - - ],
      ]

    target_seqs:
      [
        [ 2 3 4 5 ],
        [ 2 3 4 - ],
        [ 2 3 - - ],
      ]

    mask:
      [
        [ 1 1 1 1 ],
        [ 1 1 1 0 ],
        [ 1 1 0 0 ],
      ]

  Args:
    images_and_captions: A list of image and caption meta data
    batch_size: Batch size.
    queue_capacity: Queue capacity.
    add_summaries: If true, add caption length summaries.

  Returns:
    Padded image, captions, masks, etc.
  """
  enqueue_list = []
  for set_id, images, image_ids, captions, likes in images_and_captions:
    image_seq_length = tf.shape(image_ids)[0]
    input_length = tf.subtract(image_seq_length, 0) # change 1 to 0
    
    cap_indicator = tf.cast(tf.not_equal(captions,
                                         tf.zeros_like(captions)),
                            tf.int32)
    indicator = tf.ones(tf.expand_dims(input_length, 0), dtype=tf.int32)
    loss_indicator = tf.ones(tf.expand_dims(image_seq_length, 0),
                             dtype=tf.int32)
    images = tf.stack(images)
    
    enqueue_list.append([set_id, images, indicator, loss_indicator,
                        image_ids, captions, cap_indicator, likes])

  (set_ids, images, mask, loss_mask, image_ids,
    captions, cap_mask, likes) = tf.train.batch_join(enqueue_list,
                                                     batch_size=batch_size,
                                                     capacity=queue_capacity,
                                                     dynamic_pad=True,
                                                     name="batch_and_pad")

  if add_summaries:
    lengths = tf.add(tf.reduce_sum(mask, 1), 1)
    tf.summary.scalar("caption_length/batch_min", tf.reduce_min(lengths))
    tf.summary.scalar("caption_length/batch_max", tf.reduce_max(lengths))
    tf.summary.scalar("caption_length/batch_mean", tf.reduce_mean(lengths))

  return (set_ids, images, image_ids, mask, loss_mask, captions, cap_mask, likes)
