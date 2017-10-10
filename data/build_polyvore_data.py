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

"""Prepare Polyvore outfit data."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from datetime import datetime
import json
import os
import random
import sys
import threading

import numpy as np
import tensorflow as tf

tf.app.flags.DEFINE_string('train_label', 'data/label/train_no_dup.json',
                           'Training label file')
tf.app.flags.DEFINE_string('test_label', 'data/label/test_no_dup.json',
                           'Testing label file')
tf.app.flags.DEFINE_string('valid_label','data/label/valid_no_dup.json',
                           'Validation label file')
tf.app.flags.DEFINE_string('output_directory', 'data/tf_records/',
                           'Output data directory')
tf.app.flags.DEFINE_string('image_dir', 'data/images/',
                           'Directory of image patches')
tf.app.flags.DEFINE_string('word_dict_file', 'data/final_word_dict.txt',
                           'File containing the word dictionary.')

tf.app.flags.DEFINE_integer('train_shards', 128,
                            'Number of shards in training TFRecord files.')
tf.app.flags.DEFINE_integer('test_shards', 16,
                            'Number of shards in test TFRecord files.')
tf.app.flags.DEFINE_integer('valid_shards', 8,
                            'Number of shards in validation TFRecord files.')
tf.app.flags.DEFINE_integer('num_threads', 8,
                            'Number of threads to preprocess the images.')

FLAGS = tf.flags.FLAGS


class Vocabulary(object):
  """Simple vocabulary wrapper."""

  def __init__(self, vocab, unk_id):
    """Initializes the vocabulary.
    Args:
      vocab: A dictionary of word to word_id.
      unk_id: Id of the special 'unknown' word.
    """
    self._vocab = vocab
    self._unk_id = unk_id

  def word_to_id(self, word):
    """Returns the integer id of a word string."""
    if word in self._vocab:
      return self._vocab[word]
    else:
      print('unknow: ' + word)
      return self._unk_id


def _is_png(filename):
  """Determine if a file contains a PNG format image.
  Args:
    filename: string, path of the image file.
  Returns:
    boolean indicating if the image is a PNG.
  """
  return '.png' in filename


def _int64_feature(value):
  """Wrapper for inserting int64 features into Example proto."""
  if not isinstance(value, list):
    value = [value]
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))
  

def _float_feature(value):
  """Wrapper for inserting float features into Example proto."""
  if not isinstance(value, list):
    value = [value]
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))
  
  
def _bytes_feature(value):
  """Wrapper for inserting bytes features into Example proto."""
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[str(value)]))


def _int64_feature_list(values):
  """Wrapper for inserting an int64 FeatureList into a SequenceExample proto."""
  return tf.train.FeatureList(feature=[_int64_feature(v) for v in values])


def _int64_list_feature_list(values):
  """Wrapper for inserting an int64 list FeatureList into a SequenceExample proto."""
  return tf.train.FeatureList(feature=[_int64_feature(v) for v in values])


def _bytes_feature_list(values):
  """Wrapper for inserting a bytes FeatureList into a SequenceExample proto."""
  return tf.train.FeatureList(feature=[_bytes_feature(v) for v in values])

def _float_feature_list(values):
  """Wrapper for inserting a float FeatureList into a SequenceExample proto."""
  return tf.train.FeatureList(feature=[_float_feature(v) for v in values])


def _to_sequence_example(set_info, decoder, vocab):
  """Builds a SequenceExample proto for an outfit.
  """
  set_id = set_info['set_id']
  image_data = []
  image_ids = []
  caption_data = []
  caption_ids = []
  for image_info in set_info['items']:
    filename = os.path.join(FLAGS.image_dir, set_id,
                            str(image_info['index']) + '.jpg')
    with open(filename, "r") as f:
      encoded_image = f.read()
    try:
      decoded_image = decoder.decode_jpeg(encoded_image)
    except (tf.errors.InvalidArgumentError, AssertionError):
      print("Skipping file with invalid JPEG data: %s" % filename)
      return

    image_data.append(encoded_image)
    image_ids.append(image_info['index'])
    caption = image_info['name'].encode('utf-8')
    caption_data.append(caption)
    caption_id = [vocab.word_to_id(word) + 1 for word in caption.split()]
    caption_ids.append(caption_id)

  feature = {}
  # Only keep 8 images, if outfit has less than 8 items, repeat the last one.
  for index in range(8):
    if index >= len(image_data):
      feature['images/' + str(index)] = _bytes_feature(image_data[-1])
    else:
      feature['images/' + str(index)] = _bytes_feature(image_data[index])
    
  feature["set_id"] = _bytes_feature(set_id)
  feature["set_url"] = _bytes_feature(set_info['set_url'])
  # Likes and Views are not used in our model, but we put it into TFRecords.
  feature["likes"] = _int64_feature(set_info['likes'])
  feature["views"] = _int64_feature(set_info['views'])

  context = tf.train.Features(feature=feature)

  feature_lists = tf.train.FeatureLists(feature_list={
      "caption": _bytes_feature_list(caption_data),
      "caption_ids": _int64_list_feature_list(caption_ids),
      "image_index": _int64_feature_list(image_ids)
  })

  sequence_example = tf.train.SequenceExample(
      context=context, feature_lists=feature_lists)

  return sequence_example


class ImageCoder(object):
  """Helper class that provides TensorFlow image coding utilities."""

  def __init__(self):
    # Create a single Session to run all image coding calls.
    self._sess = tf.Session()

    # Initializes function that converts PNG to JPEG data.
    self._png_data = tf.placeholder(dtype=tf.string)
    image = tf.image.decode_png(self._png_data, channels=3)
    self._png_to_jpeg = tf.image.encode_jpeg(image, format='rgb', quality=100)

    # Initializes function that decodes RGB JPEG data.
    self._decode_jpeg_data = tf.placeholder(dtype=tf.string)
    self._decode_jpeg = tf.image.decode_jpeg(
                                self._decode_jpeg_data, channels=3)

  def png_to_jpeg(self, image_data):
    return self._sess.run(self._png_to_jpeg,
                          feed_dict={self._png_data: image_data})

  def decode_jpeg(self, image_data):
    image = self._sess.run(self._decode_jpeg,
                           feed_dict={self._decode_jpeg_data: image_data})
    assert len(image.shape) == 3
    assert image.shape[2] == 3
    return image


def _process_image_files_batch(coder, thread_index, ranges, name,
                               all_sets, vocab, num_shards):
  """Processes and saves list of images as TFRecord in 1 thread.
  """
  # Each thread produces N shards where N = int(num_shards / num_threads).
  # For instance, if num_shards = 128, and the num_threads = 2, then the first
  # thread would produce shards [0, 64).
  num_threads = len(ranges)
  assert not num_shards % num_threads
  num_shards_per_batch = int(num_shards / num_threads)

  shard_ranges = np.linspace(ranges[thread_index][0],
                             ranges[thread_index][1],
                             num_shards_per_batch + 1).astype(int)
  num_files_in_thread = ranges[thread_index][1] - ranges[thread_index][0]

  counter = 0
  for s in xrange(num_shards_per_batch):
    # Generate a sharded version of the file name, e.g. 'train-00002-of-00010'
    shard = thread_index * num_shards_per_batch + s
    output_filename = '%s-%.5d-of-%.5d' % (name, shard, num_shards)
    output_file = os.path.join(FLAGS.output_directory, output_filename)
    writer = tf.python_io.TFRecordWriter(output_file)

    shard_counter = 0
    files_in_shard = np.arange(shard_ranges[s], shard_ranges[s + 1], dtype=int)
    for i in files_in_shard:
      sequence_example = _to_sequence_example(all_sets[i], coder, vocab)
      if not sequence_example:
        print('fail for set: ' + all_sets[i]['set_id'])
        continue
      writer.write(sequence_example.SerializeToString())
      shard_counter += 1
      counter += 1

      if not counter % 100:
        print('%s [thread %d]: Processed %d of %d images in thread batch.' %
              (datetime.now(), thread_index, counter, num_files_in_thread))
        sys.stdout.flush()

    writer.close()
    print('%s [thread %d]: Wrote %d images to %s' %
          (datetime.now(), thread_index, shard_counter, output_file))
    sys.stdout.flush()
    shard_counter = 0
  print('%s [thread %d]: Wrote %d images to %d shards.' %
        (datetime.now(), thread_index, counter, num_files_in_thread))
  sys.stdout.flush()


def _process_image_files(name, all_sets, vocab, num_shards):
  """Process and save list of images as TFRecord of Example protos.
  """

  # Break all images into batches with a [ranges[i][0], ranges[i][1]].
  spacing = np.linspace(0, len(all_sets), FLAGS.num_threads + 1).astype(np.int)
  ranges = []
  for i in xrange(len(spacing) - 1):
    ranges.append([spacing[i], spacing[i+1]])

  # Launch a thread for each batch.
  print('Launching %d threads for spacings: %s' % (FLAGS.num_threads, ranges))
  sys.stdout.flush()

  # Create a mechanism for monitoring when all threads are finished.
  coord = tf.train.Coordinator()

  # Create a generic TensorFlow-based utility for converting all image codings.
  coder = ImageCoder()

  threads = []
  for thread_index in xrange(len(ranges)):
    args = (coder, thread_index, ranges, name, all_sets, vocab, num_shards)
    t = threading.Thread(target=_process_image_files_batch, args=args)
    t.start()
    threads.append(t)

  # Wait for all the threads to terminate.
  coord.join(threads)
  print('%s: Finished writing all %d fashion sets in data set.' %
        (datetime.now(), len(all_sets)))
  sys.stdout.flush()


def _create_vocab(filename):
  """Creates the vocabulary of word to word_id.
  """
  # Create the vocabulary dictionary.
  word_counts = open(filename).read().splitlines()
  reverse_vocab = [x.split()[0] for x in word_counts]
  unk_id = len(reverse_vocab)
  vocab_dict = dict([(x, y) for (y, x) in enumerate(reverse_vocab)])
  vocab = Vocabulary(vocab_dict, unk_id)

  return vocab


def _find_image_files(labels_file, name):
  """Build a list of all images files and labels in the data set.
  """
  
  # Read image ids
  all_sets = json.load(open(labels_file))
  
  # Shuffle the ordering of all image files in order to guarantee
  # random ordering of the images with respect to label in the
  # saved TFRecord files. Make the randomization repeatable.
  
  shuffled_index = range(len(all_sets))
  random.seed(12345)
  random.shuffle(shuffled_index)

  all_sets = [all_sets[i] for i in shuffled_index]  
  print('Found %d fashion sets.' % (len(all_sets)))
  return all_sets

def _process_dataset(name, label_file, vocab, num_shards):
  """Process a complete data set and save it as a TFRecord.
  Args:
    name: string, unique identifier specifying the data set.
    directory: string, root path to the data set.
    num_shards: integer number of shards for this data set.
    labels_file: string, path to the labels file.
  """
  print(label_file)
  all_sets  = _find_image_files(label_file, name)
  _process_image_files(name, all_sets, vocab, num_shards)


def main(unused_argv):
  assert not FLAGS.train_shards % FLAGS.num_threads, (
      'Please make the FLAGS.num_threads commensurate with FLAGS.train_shards')
  assert not FLAGS.test_shards % FLAGS.num_threads, (
      'Please make the FLAGS.num_threads commensurate with '
      'FLAGS.test_shards')
  assert not FLAGS.valid_shards % FLAGS.num_threads, (
      'Please make the FLAGS.num_threads commensurate with '
      'FLAGS.valid_shards')
  print('Saving results to %s' % FLAGS.output_directory)


  vocab = _create_vocab(FLAGS.word_dict_file)
  # Run it!
  _process_dataset('valid-no-dup', FLAGS.valid_label, vocab, FLAGS.valid_shards)
  _process_dataset('test-no-dup', FLAGS.test_label, vocab, FLAGS.test_shards)
  _process_dataset('train-no-dup', FLAGS.train_label, vocab, FLAGS.train_shards)
  

if __name__ == '__main__':
  tf.app.run()
