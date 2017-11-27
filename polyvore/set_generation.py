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
r"""Given multimodal queries, complete the outfit wiht bi-LSTM and VSE model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import math
import os

import pickle as pkl
import tensorflow as tf
import numpy as np
import configuration
import polyvore_model_bi as polyvore_model


FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string("checkpoint_path", "",
                       "Model checkpoint file or directory containing a "
                       "model checkpoint file.")
tf.flags.DEFINE_string("image_dir", "", "Directory containing images.")
tf.flags.DEFINE_string("feature_file", "", "File which contains the features.")
tf.flags.DEFINE_string("word_dict_file", "", "File containing word list.")

tf.flags.DEFINE_string("query_file", "",
                       "A json file containing the query to generate outfit.")
tf.flags.DEFINE_string("result_dir", "results",
                       "Directory to save the results.")
tf.flags.DEFINE_float("balance_factor", 2.0,
        "Trade off between image and text input."
        "Larger balance_factor encourages higher correlation with text query")


def norm_row(a):
  """L2 normalize each row of a given set."""
  try:
    return a / np.linalg.norm(a, axis=1)[:, np.newaxis]
  except:
    return a / np.linalg.norm(a)

def rnn_one_step(sess, input_feed, lstm_state, direction='f'):
  """Run one step of the RNN."""
  if direction == 'f':
    # Forward
    [lstm_state, lstm_output] = sess.run(
        fetches=['lstm/f_state:0', 'f_logits/f_logits/BiasAdd:0'],
        feed_dict={'lstm/f_input_feed:0': input_feed,
                   'lstm/f_state_feed:0': lstm_state})
  else:
    # Backward
    [lstm_state, lstm_output] = sess.run(
        fetches=['lstm/b_state:0', 'b_logits/b_logits/BiasAdd:0'],
        feed_dict={'lstm/b_input_feed:0': input_feed,
                   'lstm/b_state_feed:0': lstm_state})
    
  return lstm_state, lstm_output


def run_forward_rnn(sess, test_idx, test_feat, num_lstm_units):
  """ Run forward RNN given a query."""
  res_set = []
  lstm_state = np.zeros([1, 2 * num_lstm_units])
  for test_id in test_idx:
    input_feed = np.reshape(test_feat[test_id], [1, -1])
    # Run first step with all zeros initial state.
    [lstm_state, lstm_output] = rnn_one_step(
          sess, input_feed, lstm_state, direction='f')

  # Maximum length of the outfit is set to 10.
  for step in range(10):
    curr_score = np.exp(np.dot(lstm_output, np.transpose(test_feat)))
    curr_score /= np.sum(curr_score)

    next_image = np.argsort(-curr_score)[0][0]
    # 0.00001 is used as a probablity threshold to stop the generation.
    # i.e, if the prob of end-of-set is larger than 0.00001, then stop.
    if next_image == test_feat.shape[0] - 1 or curr_score[0][-1] > 0.00001:
      # print('OVER')
      break
    else:
      input_feed = np.reshape(test_feat[next_image], [1, -1])
      [lstm_state, lstm_output] = rnn_one_step(
            sess, input_feed, lstm_state, direction='f')
      res_set.append(next_image)

  return res_set


def run_backward_rnn(sess, test_idx, test_feat, num_lstm_units):
  """ Run backward RNN given a query."""
  res_set = []
  lstm_state = np.zeros([1, 2 * num_lstm_units])
  for test_id in reversed(test_idx):
    input_feed = np.reshape(test_feat[test_id], [1, -1])
    [lstm_state, lstm_output] = rnn_one_step(
          sess, input_feed, lstm_state, direction='b')
  for step in range(10):
    curr_score = np.exp(np.dot(lstm_output, np.transpose(test_feat)))
    curr_score /= np.sum(curr_score)
    next_image = np.argsort(-curr_score)[0][0]
    # 0.00001 is used as a probablity threshold to stop the generation.
    # i.e, if the prob of end-of-set is larger than 0.00001, then stop.
    if next_image == test_feat.shape[0] - 1 or curr_score[0][-1] > 0.00001:
      # print('OVER')
      break
    else:
      input_feed = np.reshape(test_feat[next_image], [1, -1])
      [lstm_state, lstm_output] = rnn_one_step(
          sess, input_feed, lstm_state, direction='b')
      res_set.append(next_image)

  return res_set


def run_fill_rnn(sess, start_id, end_id, num_blank, test_feat, num_lstm_units):
  """Fill in the blanks between start and end."""
  if num_blank == 0:
    return [start_id, end_id]
  lstm_f_outputs = []
  lstm_state = np.zeros([1, 2 * num_lstm_units])
  input_feed = np.reshape(test_feat[start_id], [1, -1])
  [lstm_state, lstm_output] = rnn_one_step(
        sess, input_feed, lstm_state, direction='f')

  f_outputs = []
  for i in range(num_blank):
    f_outputs.append(lstm_output[0])
    curr_score = np.exp(np.dot(lstm_output, np.transpose(test_feat)))
    curr_score /= np.sum(curr_score)
    next_image = np.argsort(-curr_score)[0][0]
    input_feed = np.reshape(test_feat[next_image], [1, -1])
    [lstm_state, lstm_output] = rnn_one_step(
          sess, input_feed, lstm_state, direction='f')

  lstm_state = np.zeros([1, 2 * num_lstm_units])
  input_feed = np.reshape(test_feat[end_id], [1, -1])
  [lstm_state, lstm_output] = rnn_one_step(
        sess, input_feed, lstm_state, direction='b')

  b_outputs = []
  for i in range(num_blank):
    b_outputs.insert(0, lstm_output[0])
    curr_score = np.exp(np.dot(lstm_output, np.transpose(test_feat)))
    curr_score /= np.sum(curr_score)
    next_image = np.argsort(-curr_score)[0][0]
    input_feed = np.reshape(test_feat[next_image], [1, -1])
    [lstm_state, lstm_output] = rnn_one_step(
          sess, input_feed, lstm_state, direction='b')

  outputs = np.asarray(f_outputs) + np.asarray(b_outputs)
  score = np.exp(np.dot(outputs, np.transpose(test_feat)))
  score /= np.sum(score, axis=1)[:, np.newaxis]
  blank_ids = np.argmax(score, axis=1)
  return [start_id] + list(blank_ids) + [end_id]


def run_set_inference(sess, set_name, test_ids, test_feat, num_lstm_units):
  test_idx = []
  for name in set_name:
    try:
      test_idx.append(test_ids.index(name))
    except:
      print('not found')
      return

  # dynamic search
  # run the whole bi-LSTM on the first item
  first_f_set = run_forward_rnn(sess, test_idx[:1], test_feat, num_lstm_units)
  first_b_set = run_backward_rnn(sess, test_idx[:1], test_feat, num_lstm_units)

  first_posi = len(first_b_set)
  first_set = first_b_set + test_idx[:1] + first_f_set

  image_set = []
  for i in first_set:
    image_set.append(test_ids[i])

  # # Write results into folder.
  # os.system('mkdir %s/%s' % (FLAGS.result_dir, 'first'))
  # for i, image in enumerate(image_set):
  #   name = image.split('_')
  #   os.system('cp %s/%s/%s.jpg %s/%s/%d_%s.jpg' % (FLAGS.image_dir,
  #             name[0], name[1], FLAGS.result_dir, 'first', i, image))

  if len(set_name) >= 2:
    current_set = norm_row(test_feat[first_set, :])
    all_position = [first_posi]
    for test_id in test_idx[1:]:
      # gradually adding items into it
      # findng nn of the next item
      insert_posi = np.argmax(
          np.dot(norm_row(test_feat[test_id, :]), np.transpose(current_set)))
      all_position.append(insert_posi)

    # run bi LSTM to fill items between first item and this item
    start_posi = np.min(all_position)
    end_posi = np.max(all_position)

    sets = run_fill_rnn(sess, test_idx[0], test_idx[1],
                        end_posi - start_posi - 1, test_feat, num_lstm_units)

  else:
    # run bi LSTM again
    sets = test_idx
  f_set = run_forward_rnn(sess, sets, test_feat, num_lstm_units)
  b_set = run_backward_rnn(sess, sets, test_feat, num_lstm_units)

  image_set = []
  for i in b_set[::-1] + sets+f_set:
    image_set.append(test_ids[i])

  # os.system('mkdir %s/%s' % (FLAGS.result_dir, 'final'))
  # for i, image in enumerate(image_set):
  #   name = image.split('_')
  #   os.system('cp %s/%s/%s.jpg %s/%s/%d_%s.jpg' % (FLAGS.image_dir,
  #                   name[0], name[1], FLAGS.result_dir, 'final', i, image))

  return b_set[::-1] + sets + f_set


def nn_search(i, test_emb, word_vec):
  # score = np.dot(test_emb, np.transpose(test_emb[i] + word_vec))
  score = np.dot(test_emb,
        np.transpose(test_emb[i] + FLAGS.balance_factor * word_vec))
  return np.argmax(score)


def main(_):
  # Build the inference graph.
  g = tf.Graph()
  with g.as_default():
    model_config = configuration.ModelConfig()
    model = polyvore_model.PolyvoreModel(model_config, mode="inference")
    model.build()
    saver = tf.train.Saver()

    g.finalize()
    with tf.Session() as sess:
      saver.restore(sess, FLAGS.checkpoint_path)
      with open(FLAGS.feature_file, "rb") as f:
        test_data = pkl.load(f)

      test_ids = test_data.keys()
      test_feat = np.zeros((len(test_ids) + 1,
                            len(test_data[test_ids[0]]["image_rnn_feat"])))
      test_emb = np.zeros((len(test_ids),
                           len(test_data[test_ids[0]]["image_feat"])))

      for i, test_id in enumerate(test_ids):
        # Image feature in the RNN space.
        test_feat[i] = test_data[test_id]["image_rnn_feat"]
        # Image feature in the joint embedding space.
        test_emb[i] = test_data[test_id]["image_feat"]

      test_emb = norm_row(test_emb)

      # load queries from JSON file
      queries = json.load(open(FLAGS.query_file))
      # Get the word embedding.
      [word_emb] = sess.run([model.embedding_map])

      # Read word name
      words = open(FLAGS.word_dict_file).read().splitlines()
      for i, w in enumerate(words):
        words[i] = w.split()[0]

      # Calculate the embedding of the word query
      # only run the first query for demo
      for q in queries[:1]:
        set_name = q['image_query']
        print(set_name)
        # Run Bi-LSTM model using the image query.
        rnn_sets = run_set_inference(sess, set_name, test_ids,
                                     test_feat, model_config.num_lstm_units)
        print(rnn_sets)

        # Reranking the LSTM prediction with similarity with the text query        
        word_query = str(q['text_query'])
        print(word_query)
        if word_query != "":
          # Get the indices of images.
          test_idx = []
          for name in set_name:
            try:
              test_idx.append(test_ids.index(name))
            except:
              print('not found')
              return

          # Calculate the word embedding
          word_query = [i+1 for i in range(len(words))
                            if words[i] in word_query.split()]
          print(word_query)
          query_emb = norm_row(np.sum(word_emb[word_query], axis=0))
          for i, j in enumerate(rnn_sets):
            if j not in test_idx:
              rnn_sets[i] = nn_search(j, test_emb, query_emb)
          print(rnn_sets)

        # write images          
        image_set = []
        for i in rnn_sets:
          image_set.append(test_ids[i])

        # write results
        # os.system('mkdir %s/%s' % (FLAGS.result_dir, 'emb_final'))
        # for i, image in enumerate(image_set):
        #   name = image.split('_')
        #   os.system('cp %s/%s/%s.jpg %s/%s/%d_%s.jpg' % (FLAGS.image_dir,
        #       name[0], name[1], FLAGS.result_dir, 'emb_final', i, image))
  
        for i, image in enumerate(image_set):
          name = image.split('_')
          os.system('cp %s/%s/%s.jpg %s/%d_%s.jpg' % (FLAGS.image_dir,
              name[0], name[1], FLAGS.result_dir, i, image))

if __name__ == "__main__":
  tf.app.run()
