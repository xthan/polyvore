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
"""Predict the fashion compatibility of a given image sequence."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import json

import tensorflow as tf
import numpy as np
import pickle as pkl
from sklearn import metrics

import configuration
import polyvore_model_bi as polyvore_model

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string("checkpoint_path", "",
                       "Model checkpoint file or directory containing a "
                       "model checkpoint file.")
tf.flags.DEFINE_string("label_file", "", "Txt file containing test outfits.")
tf.flags.DEFINE_string("feature_file", "", "Files containing image features")
tf.flags.DEFINE_string("rnn_type", "", "Type of RNN.")
tf.flags.DEFINE_string("result_file", "", "File to store the results.")
tf.flags.DEFINE_integer("direction", 2, "2: bidirectional; 1: forward only;"
                        "-1: backward only.")


def run_compatibility_inference(sess, image_seqs, test_feat,
                                num_lstm_units, model):
  emb_seqs = test_feat[image_seqs,:]
  num_images = float(len(image_seqs))
  if FLAGS.rnn_type == "lstm":
    zero_state = np.zeros([1, 2 * num_lstm_units])
  else:
    zero_state = np.zeros([1, num_lstm_units])
  
  f_score = 0
  b_score = 0
  if FLAGS.direction != -1:
    # Forward RNN.
    outputs = []
    input_feed = np.reshape(emb_seqs[0], [1,-1])
    # Run first step with all zeros initial state.
    [lstm_state, lstm_output] = sess.run(
          fetches=["lstm/f_state:0","f_logits/f_logits/BiasAdd:0"],
          feed_dict={"lstm/f_input_feed:0":input_feed,
                     "lstm/f_state_feed:0":zero_state})
    outputs.append(lstm_output)

    # Run remaining steps.
    for step in range(int(num_images)-1):
      input_feed = np.reshape(emb_seqs[step+1], [1,-1])
      [lstm_state, lstm_output] = sess.run(
                fetches=["lstm/f_state:0","f_logits/f_logits/BiasAdd:0"],
                feed_dict={"lstm/f_input_feed:0":input_feed,
                           "lstm/f_state_feed:0":lstm_state})
      outputs.append(lstm_output)
    
    # Calculate the loss.
    # Different from the training process where the loss is calculated in each
    # mini batch, during testing, we get the loss againist the whole test set.
    # This is pretty slow, maybe a better method could be used.
    s = np.squeeze(np.dot(np.asarray(outputs), np.transpose(test_feat)))
    f_score = sess.run(model.lstm_xent_loss,
         feed_dict={"lstm/pred_feed:0":s,
         "lstm/next_index_feed:0":image_seqs[1:] + [test_feat.shape[0]-1]})
    
    f_score = - np.mean(f_score)
    
  if FLAGS.direction != 1:
    # Backward RNN.
    outputs = []
    input_feed = np.reshape(emb_seqs[-1], [1,-1])
    [lstm_state, lstm_output] = sess.run(
                fetches=["lstm/b_state:0","b_logits/b_logits/BiasAdd:0"],
                feed_dict={"lstm/b_input_feed:0":input_feed,
                           "lstm/b_state_feed:0":zero_state})
    outputs.append(lstm_output)
    for step in range(int(num_images)-1):
      input_feed = np.reshape(emb_seqs[int(num_images)-2-step], [1,-1])
      [lstm_state, lstm_output] = sess.run(
                fetches=["lstm/b_state:0","b_logits/b_logits/BiasAdd:0"],
                feed_dict={"lstm/b_input_feed:0":input_feed,
                           "lstm/b_state_feed:0":lstm_state})
      outputs.append(lstm_output)
    
    # Calculate the loss.
    s = np.squeeze(np.dot(np.asarray(outputs), np.transpose(test_feat)))
    b_score = sess.run(model.lstm_xent_loss,
        feed_dict={"lstm/pred_feed:0":s,
        "lstm/next_index_feed:0": image_seqs[-2::-1] + [test_feat.shape[0]-1]})
    b_score = - np.mean(b_score)
  return [f_score, b_score]


  
def main(_):
  # Build the inference graph.
  g = tf.Graph()
  with g.as_default():
    model_config = configuration.ModelConfig()
    model_config.rnn_type = FLAGS.rnn_type
    model = polyvore_model.PolyvoreModel(model_config, mode="inference")
    model.build()
    saver = tf.train.Saver()
    
    # Load pre-computed image features.
    with open(FLAGS.feature_file, "rb") as f:
      test_data = pkl.load(f)
    test_ids = test_data.keys()
    test_feat = np.zeros((len(test_ids) + 1,
                    len(test_data[test_ids[0]]["image_rnn_feat"])))
    # test_feat has one more zero vector as the representation of END of
    # RNN prediction.
    for i, test_id in enumerate(test_ids):
      # Image feature in the RNN space.
      test_feat[i] = test_data[test_id]["image_rnn_feat"]
    
    g.finalize()
    with tf.Session() as sess:
      saver.restore(sess, FLAGS.checkpoint_path)
      all_f_scores = []
      all_b_scores = []
      all_scores = []
      all_labels = []
      testset = open(FLAGS.label_file).read().splitlines()
      k = 0
      for test_outfit in testset:
        k += 1
        if k % 100 == 0:
          print("Finish %d outfits." % k)
        image_seqs = []
        for test_image in test_outfit.split()[1:]:
          image_seqs.append(test_ids.index(test_image))
          
        [f_score, b_score] = run_compatibility_inference(sess, image_seqs,
                      test_feat, model_config.num_lstm_units, model)
        
        all_f_scores.append(f_score)
        all_b_scores.append(b_score)
        all_scores.append(f_score + b_score)
        all_labels.append(int(test_outfit[0]))
        
      # calculate AUC and AP      
      fpr, tpr, thresholds = metrics.roc_curve(all_labels,
                                               all_scores,
                                               pos_label=1)
      print("Compatibility AUC: %f for %d outfits" %
              (metrics.auc(fpr, tpr), len(all_labels)))

      with open(FLAGS.result_file, "wb") as f:
        pkl.dump({"all_labels": all_labels, "all_f_scores": all_f_scores,
                  "all_b_scores": all_b_scores}, f)

      
if __name__ == "__main__":
  tf.app.run()
