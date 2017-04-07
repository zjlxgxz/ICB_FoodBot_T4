# -*- coding: utf-8 -*-
"""
Created on Sun Feb 28 16:23:37 2016

@author: Bing Liu (liubing@cmu.edu)

Modified on Wed Apr 5 2017
@modified by Xingzhi Guo(guoxingzhi@gmail.com) for ICB2017 @ NTU
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import sys
import time

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
from tensorflow.python.platform import gfile

import data_utils
import multi_task_model

import subprocess
import stat

from searchdb import SearchDB


#tf.app.flags.DEFINE_float("learning_rate", 0.1, "Learning rate.")
#tf.app.flags.DEFINE_float("learning_rate_decay_factor", 0.9,
#                          "Learning rate decays by this much.")
tf.app.flags.DEFINE_float("max_gradient_norm", 5.0,
                          "Clip gradients to this norm.")
tf.app.flags.DEFINE_integer("batch_size", 16,
                            "Batch size to use during training.")
tf.app.flags.DEFINE_integer("size", 128, "Size of each model layer.")
tf.app.flags.DEFINE_integer("word_embedding_size", 128, "Size of the word embedding")
tf.app.flags.DEFINE_integer("num_layers", 1, "Number of layers in the model.")
tf.app.flags.DEFINE_integer("in_vocab_size", 10000, "max vocab Size.")
tf.app.flags.DEFINE_integer("out_vocab_size", 10000, "max tag vocab Size.")
tf.app.flags.DEFINE_string("data_dir", "/tmp", "Data directory")
tf.app.flags.DEFINE_string("train_dir", "/tmp", "Training directory.")
tf.app.flags.DEFINE_integer("max_train_data_size", 0,
                            "Limit on the size of training data (0: no limit).")
tf.app.flags.DEFINE_integer("steps_per_checkpoint", 300,
                            "How many training steps to do per checkpoint.")
tf.app.flags.DEFINE_integer("max_training_steps", 10000,
                            "Max training steps.")
tf.app.flags.DEFINE_integer("max_test_data_size", 0,
                            "Max size of test set.")
tf.app.flags.DEFINE_boolean("use_attention", True,
                            "Use attention based RNN")
tf.app.flags.DEFINE_integer("max_sequence_length", 0,
                            "Max sequence length.")
tf.app.flags.DEFINE_float("dropout_keep_prob", 0.5,
                          "dropout keep cell input and output prob.")  
tf.app.flags.DEFINE_boolean("bidirectional_rnn", True,
                            "Use birectional RNN")
tf.app.flags.DEFINE_string("task", None, "Options: joint; intent; tagging")
FLAGS = tf.app.flags.FLAGS
    
if FLAGS.max_sequence_length == 0:
    print ('Please indicate max sequence length. Exit')
    exit()

if FLAGS.task is None:
    print ('Please indicate task to run. Available options: intent; tagging; joint')
    exit()

task = dict({'intent':0, 'tagging':0, 'joint':0})
if FLAGS.task == 'intent':
    task['intent'] = 1
elif FLAGS.task == 'tagging':
    task['tagging'] = 1
elif FLAGS.task == 'joint':
    task['intent'] = 1
    task['tagging'] = 1
    task['joint'] = 1
    
_buckets = [(FLAGS.max_sequence_length, FLAGS.max_sequence_length)]
#_buckets = [(3, 10), (10, 25)]

# metrics function using conlleval.pl
def conlleval(p, g, w, filename):
    '''
    INPUT:
    p :: predictions
    g :: groundtruth
    w :: corresponding words

    OUTPUT:
    filename :: name of the file where the predictions
    are written. it will be the input of conlleval.pl script
    for computing the performance in terms of precision
    recall and f1 score
    '''
    out = ''
    for sl, sp, sw in zip(g, p, w):
        out += 'BOS O O\n'
        for wl, wp, w in zip(sl, sp, sw):
            out += w + ' ' + wl + ' ' + wp + '\n'
        out += 'EOS O O\n\n'

    f = open(filename, 'w')
    f.writelines(out[:-1]) # remove the ending \n on last line
    f.close()

    return get_perf(filename)

def get_perf(filename):
    ''' run conlleval.pl perl script to obtain
    precision/recall and F1 score '''
    _conlleval = os.path.dirname(os.path.realpath(__file__)) + '/conlleval.pl'
    os.chmod(_conlleval, stat.S_IRWXU)  # give the execute permissions

    proc = subprocess.Popen(["perl",
                            _conlleval],
                            stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE)

    stdout, _ = proc.communicate(''.join(open(filename).readlines()))
    for line in stdout.split('\n'):
        if 'accuracy' in line:
            out = line.split()
            break

    precision = float(out[6][:-2])
    recall = float(out[8][:-2])
    f1score = float(out[10])

    return {'p': precision, 'r': recall, 'f1': f1score}


def read_data(source_path, target_path, label_path, max_size=None):
  """Read data from source and target files and put into buckets.

  Args:
    source_path: path to the files with token-ids for the source input - word sequence.
    target_path: path to the file with token-ids for the target output - tag sequence;
      it must be aligned with the source file: n-th line contains the desired
      output for n-th line from the source_path.
    label_path: path to the file with token-ids for the sequence classification label
    max_size: maximum number of lines to read, all other will be ignored;
      if 0 or None, data files will be read completely (no limit).

  Returns:
    data_set: a list of length len(_buckets); data_set[n] contains a list of
      (source, target, label) tuple read from the provided data files that fit
      into the n-th bucket, i.e., such that len(source) < _buckets[n][0] and
      len(target) < _buckets[n][1]; source,  target, and label are lists of token-ids.
  """
  data_set = [[] for _ in _buckets]
  with tf.gfile.GFile(source_path, mode="r") as source_file:
    with tf.gfile.GFile(target_path, mode="r") as target_file:
      with tf.gfile.GFile(label_path, mode="r") as label_file:
        source, target, label = source_file.readline(), target_file.readline(), label_file.readline()
        counter = 0
        while source and target and label and (not max_size or counter < max_size):
          counter += 1
          source_ids = [int(x) for x in source.split()]
          target_ids = [int(x) for x in target.split()]
          label_ids = [int(x) for x in label.split()]
#          target_ids.append(data_utils.EOS_ID)
          for bucket_id, (source_size, target_size) in enumerate(_buckets):
            if len(source_ids) < source_size and len(target_ids) < target_size:
              data_set[bucket_id].append([source_ids, target_ids, label_ids])
              break
          source, target, label = source_file.readline(), target_file.readline(), label_file.readline()
  return data_set # 3 outputs in each unit: source_ids, target_ids, label_ids 

def create_model(session, source_vocab_size, target_vocab_size, label_vocab_size):
  """Create model and initialize or load parameters in session."""
  with tf.variable_scope("model", reuse=None):
    model_train = multi_task_model.MultiTaskModel(
          source_vocab_size, target_vocab_size, label_vocab_size, _buckets,
          FLAGS.word_embedding_size, FLAGS.size, FLAGS.num_layers, FLAGS.max_gradient_norm, FLAGS.batch_size,
          dropout_keep_prob=FLAGS.dropout_keep_prob, use_lstm=True,
          forward_only=False, 
          use_attention=FLAGS.use_attention,
          bidirectional_rnn=FLAGS.bidirectional_rnn,
          task=task)
  with tf.variable_scope("model", reuse=True):
    model_test = multi_task_model.MultiTaskModel(
          source_vocab_size, target_vocab_size, label_vocab_size, _buckets,
          FLAGS.word_embedding_size, FLAGS.size, FLAGS.num_layers, FLAGS.max_gradient_norm, FLAGS.batch_size,
          dropout_keep_prob=FLAGS.dropout_keep_prob, use_lstm=True,
          forward_only=True, 
          use_attention=FLAGS.use_attention,
          bidirectional_rnn=FLAGS.bidirectional_rnn,
          task=task)

    restorationPath = "./model_tmp/model_final.ckpt" # It will change. Somehow we must solve this problem
    if True:
      print("Reading model parameters from %s" % restorationPath)
      model_train.saver.restore(session, restorationPath)
      #model_test.saver.restore(session, restorationPath)
    else:
      print("Created model with fresh parameters.")
      session.run(tf.initialize_all_variables())
    return model_train, model_test
     

def testing():
  print ('Applying Parameters:')
  for k,v in FLAGS.__dict__['__flags'].iteritems():
    print ('%s: %s' % (k, str(v)))
  print("Preparing data in %s" % FLAGS.data_dir)
  vocab_path = ''
  tag_vocab_path = ''
  label_vocab_path = ''
  in_seq_train, out_seq_train, label_train, in_seq_dev, out_seq_dev, label_dev, in_seq_test, out_seq_test, label_test, vocab_path, tag_vocab_path, label_vocab_path = data_utils.prepare_multi_task_data(
    FLAGS.data_dir, FLAGS.in_vocab_size, FLAGS.out_vocab_size)     
     
  result_dir = FLAGS.train_dir + '/test_results'
  if not os.path.isdir(result_dir):
      os.makedirs(result_dir)

  current_taging_valid_out_file = result_dir + '/tagging.valid.hyp.txt'
  current_taging_test_out_file = result_dir + '/tagging.test.hyp.txt'

  vocab, rev_vocab = data_utils.initialize_vocabulary(vocab_path)
  tag_vocab, rev_tag_vocab = data_utils.initialize_vocabulary(tag_vocab_path)
  label_vocab, rev_label_vocab = data_utils.initialize_vocabulary(label_vocab_path)
    
  with tf.Session() as sess:
    # Create model.
    print("Max sequence length: %d." % _buckets[0][0])
    print("Creating %d layers of %d units." % (FLAGS.num_layers, FLAGS.size))
    
    model, model_test = create_model(sess, len(vocab), len(tag_vocab), len(label_vocab))
    print ("Creating model with source_vocab_size=%d, target_vocab_size=%d, and label_vocab_size=%d." % (len(vocab), len(tag_vocab), len(label_vocab)))

    if True:
      def run_valid_test(data_set, mode): # mode: Eval, Test
      # Run evals on development/test set and print the accuracy.
          word_list = list() 
          ref_tag_list = list() 
          hyp_tag_list = list()
          ref_label_list = list()
          hyp_label_list = list()

          for bucket_id in xrange(len(_buckets)):
            for i in xrange(len(data_set[bucket_id])):
              encoder_inputs, tags, tag_weights, sequence_length, labels = model_test.get_one(
                data_set, bucket_id, i)
              tagging_logits = []
              classification_logits = []
              if task['joint'] == 1:
                _, step_loss, tagging_logits, classification_logits = model_test.joint_step(sess, encoder_inputs, tags, tag_weights, labels,
                                            sequence_length, bucket_id, True)
              hyp_label = None
              if task['intent'] == 1:
                ref_label_list.append(rev_label_vocab[labels[0][0]])
                hyp_label = np.argmax(classification_logits[0],0)
                hyp_label_list.append(rev_label_vocab[hyp_label])
              if task['tagging'] == 1:
                word_list.append([rev_vocab[x[0]] for x in encoder_inputs[:sequence_length[0]]])
                ref_tag_list.append([rev_tag_vocab[x[0]] for x in tags[:sequence_length[0]]])
                hyp_tag_list.append([rev_tag_vocab[np.argmax(x)] for x in tagging_logits[:sequence_length[0]]])
          print (hyp_tag_list)
          print (hyp_label_list)
          return hyp_tag_list,hyp_label_list
          
      # run testing
      while True:
        var = raw_input("Enter 'e' to quit, or enter other char to run test ")
        if(str(var) == "e"):
            break
        else:
            userInput = str(var).lower()
            test_path = FLAGS.data_dir + '/test/test'
            in_path = test_path + ".seq.in"
            out_path = test_path + ".seq.out"
            label_path = test_path + ".label"

            # write data to path
            def writeTestingDataToPath(testingString):
              tokens = testingString.split()
              lens = len(tokens)

              with gfile.GFile(in_path, mode="w") as vocab_file:
                vocab_file.write(userInput + "\n")

              with gfile.GFile(out_path, mode="w") as vocab_file:
                for i in range(lens):
                  vocab_file.write('O' + " ")
                vocab_file.write('\n')
                
              with gfile.GFile(label_path, mode="w") as vocab_file:
                vocab_file.write("NONE" + "\n") 
            
            writeTestingDataToPath(userInput)

            in_seq_test, out_seq_test, label_test = data_utils.prepare_multi_task_data_for_testing(FLAGS.data_dir, FLAGS.in_vocab_size, FLAGS.out_vocab_size)     
            test_set = read_data(in_seq_test, out_seq_test, label_test)
            test_tagging_result,test_label_result = run_valid_test(test_set, 'Test')      

            slots = {'CATEGORY':'' ,'RESTAURANTNAME':'' ,'LOCATION':'' ,'TIME':''}
            search = SearchDB('140.112.49.151' ,'foodbot' ,'welovevivian' ,'foodbotDB')

            tokens = userInput.split()
            print tokens
            for index_token in range(len(tokens)):
              if "B-CATEGORY" in test_tagging_result or "I-CATEGORY" in test_tagging_result :
                slots['CATEGORY'] = str(slots['CATEGORY'] +tokens[index_token] + " ")
              if "B-RESTAURANTNAME" in test_tagging_result or "I-RESTAURANTNAME" in test_tagging_result:
                slots['RESTAURANTNAME'] = str(slots['RESTAURANTNAME']+tokens[index_token]+" ")
              if "B-LOCATION" in test_tagging_result or "I-LOCATION" in test_tagging_result:
                slots['LOCATION'] = str(slots['LOCATION'] +tokens[index_token] + " ")
              if "B-TIME" in test_tagging_result or "I-TIME" in test_tagging_result:
                slots['TIME'] = str(slots['TIME'] +tokens[index_token] + " ")
            print (slots)
            search.grabData(test_label_result[0] ,slots)

  

def main(_):
    #train()
    testing()
if __name__ == "__main__":
  tf.app.run()


