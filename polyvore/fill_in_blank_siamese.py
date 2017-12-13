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

"""Fill in blank evaluation."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json

import tensorflow as tf
import numpy as np
import pickle as pkl

import configuration
import polyvore_model_siamese as polyvore_model

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string("checkpoint_path", "",
                       "Model checkpoint file or directory containing a "
                       "model checkpoint file.")
tf.flags.DEFINE_string("json_file", "",
                       "Json file containing questions and answers.")
tf.flags.DEFINE_string("feature_file", "", "pkl files containing the features")
tf.flags.DEFINE_string("result_file", "", "File to store the results.")

def run_question_inference(sess, question, test_ids, test_feat):
  question_ids = []
  answer_ids = []
  for q in question["question"]:
    try:
      question_ids.append(test_ids.index(q))
    except:
      return [], []
  
  for a in question["answers"]:
    try:
      answer_ids.append(test_ids.index(a))
    except:
      return [], []
      
  blank_posi = question["blank_position"]
  
  # Average pooling of the VSE embeddings
  question_emb = np.reshape(np.mean(test_feat[question_ids], 0), [1,-1])
  q_emb = question_emb / np.linalg.norm(question_emb, axis=1)[:, np.newaxis]
  a_emb = (test_feat[answer_ids] /
             np.linalg.norm(test_feat[answer_ids], axis=1)[:, np.newaxis])
  score = (np.dot(q_emb, np.transpose(a_emb)) + 1) / 2 # scale to [0,1]
  
  predicted_answer = np.argsort(-score)[0]
  return score, predicted_answer
  

  
def main(_):
  # Build the inference graph.
  top_k = 4 # Print the top_k accuracy.
  true_pred = np.zeros(top_k)
  # Load pre-computed image features.
  with open(FLAGS.feature_file, "rb") as f:
    test_data = pkl.load(f)
  test_ids = test_data.keys()
  test_feat = np.zeros((len(test_ids),
                        len(test_data[test_ids[0]]["image_feat"])))
  for i, test_id in enumerate(test_ids):
    # Image feature in visual-semantic embedding space.
    test_feat[i] = test_data[test_id]["image_feat"]

  g = tf.Graph()
  with g.as_default():
    model_config = configuration.ModelConfig()
    model = polyvore_model.PolyvoreModel(model_config, mode="inference")
    model.build()
    saver = tf.train.Saver()
    
    g.finalize()
    with tf.Session() as sess:
      saver.restore(sess, FLAGS.checkpoint_path)
      questions = json.load(open(FLAGS.json_file))
      
      all_pred = []
      set_ids = []
      all_scores = []
      for question in questions:
        score, pred = run_question_inference(sess, question, test_ids,
                                             test_feat)
        if pred != []:
          all_pred.append(pred)
          all_scores.append(score)
          set_ids.append(question["question"][0].split("_")[0])
          # 0 is the correct answer, iterate over top_k.
          for i in range(top_k):
            if 0 in pred[:i+1]:
              true_pred[i] += 1

      # Print all top-k accuracy.
      for i in range(top_k):
        print("Top %d Accuracy: " % (i + 1))
        print("%d correct answers in %d valid questions." %
                  (true_pred[i], len(all_pred)))
        print("Accuracy: %f" % (true_pred[i] / len(all_pred)))
        
      s = np.empty((len(all_scores),), dtype=np.object)
      for i in range(len(all_scores)):
          s[i] = all_scores[i]

      with open(FLAGS.result_file, "wb") as f:
        pkl.dump({"set_ids": set_ids, "pred": all_pred, "score": s}, f)

if __name__ == "__main__":
  tf.app.run()
