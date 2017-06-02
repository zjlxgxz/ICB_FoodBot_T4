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
import json

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
from tensorflow.python.platform import gfile

import data_utils
import multi_task_model

import subprocess
import stat

#GRPC dependencies
sys.path.append('../FoodBot_GRPC_Server/')
import grpc
import FoodBot_pb2
from concurrent import futures
import collections
import random
#GRPC for RL agent.
import FoodBotRLAgent_pb2

from searchdb import SearchDB

#GRPC RL connection
channel = grpc.insecure_channel('localhost:50053')
stub = FoodBotRLAgent_pb2.FoodBotRLRequestStub(channel)


#global vars
model_test =  0
sess = 0
vocab = 0
rev_vocab = 0
tag_vocab = 0
rev_tag_vocab = 0
label_vocab = 0
rev_label_vocab = 0
observation = collections.deque(maxlen=10)
state = {'Get_Restaurant':{'LOCATION':'' ,'CATEGORY':'' ,'TIME':''} ,'Get_location':{'RESTAURANTNAME':''} ,'Get_rating':{'RESTAURANTNAME':''} , 'Get_another_restaurant':'', 'Inform':{'RESTAURANTNAME':'', 'LOCATION':'', 'CATEGORY':'', 'TIME':''}, 'Confirm':'', 'Wrong':''}
stateList = []
changeRestNum = 0
intents = collections.deque(maxlen=2)
waitConfirm = []
dialogNum = 0.0
successNum = 0.0
notfoundNum = 0.0
successRateFileName = 'successRate' + str(time.time()) + '.txt'

textFileName = "record_"+str(time.time())
LUWrongCount = 0 
LURightCount = 0
################## below for state
formerState = [0,0,0,0,0,0,0,0,0,0,0] #To remember the former state

##################below for nlg
#lists needed
content_list = ["category", "time", "location"]
action = -1


# files
f1 = open('restName.txt', 'r')
restaurant_list1 = f1.read().split('\n')
restaurant_list = [item.replace('-', ' ') for item in restaurant_list1]

#patterns of user
f2 = open('sentence_pattern/yes.txt', 'r')
yes_list = f2.read().split('\n')

f3 = open('sentence_pattern/thanks.txt', 'r')
thanks_list = f3.read().split('\n')

f4 = open('sentence_pattern/get_restaurant.txt', 'r')
get_restaurant_pattern = f4.read().split('\n')

f5 = open('sentence_pattern/get_location.txt', 'r')
get_location_pattern = f5.read().split('\n')

f6 = open('sentence_pattern/get_rating.txt', 'r')
get_rating_pattern = f6.read().split('\n')

#f7 = open('get_comment.txt', 'r')
#get_comment_pattern = f7.read().split('\n')

f8 = open('sentence_pattern/recommend.txt', 'r')
recommend_pattern = f8.read().split('\n')

f9 = open('sentence_pattern/inform_location.txt', 'r')
inform_location_pattern = f9.read().split('\n')

f10 = open('sentence_pattern/inform_category.txt', 'r')
inform_category_pattern = f10.read().split('\n')

f11 = open('sentence_pattern/request_location.txt', 'r')
request_location_pattern = f11.read().split('\n')

f12 = open('sentence_pattern/request_category.txt', 'r')
request_category_pattern = f12.read().split('\n')

f13 = open('sentence_pattern/request_time.txt', 'r')
request_time_pattern = f13.read().split('\n')

######################


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
    global model_test
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
     
def run_valid_test(data_set, mode,sess): # mode: Eval, Test
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
        #print (hyp_tag_list)
        #print (hyp_label_list)
        return hyp_tag_list,hyp_label_list


# write test data to path
def writeTestingDataToPath(testingString,in_path,out_path,label_path):
  tokens = testingString.split()
  lens = len(tokens)

  with gfile.GFile(in_path, mode="w") as vocab_file:
    vocab_file.write(testingString + "\n")

  with gfile.GFile(out_path, mode="w") as vocab_file:
    for i in range(lens):
      vocab_file.write('O' + " ")
    vocab_file.write('\n')
    
  with gfile.GFile(label_path, mode="w") as vocab_file:
    vocab_file.write("NONE" + "\n") 

def languageUnderstanding(userInput):
  test_path = FLAGS.data_dir + '/test/test'
  in_path = test_path + ".seq.in"
  out_path = test_path + ".seq.out"
  label_path = test_path + ".label"
  writeTestingDataToPath(userInput,in_path,out_path,label_path)
  in_seq_test, out_seq_test, label_test = data_utils.prepare_multi_task_data_for_testing(FLAGS.data_dir, FLAGS.in_vocab_size, FLAGS.out_vocab_size)     
  test_set = read_data(in_seq_test, out_seq_test, label_test)
  test_tagging_result,test_label_result = run_valid_test(test_set, 'Test', sess) 
  #print (test_tagging_result)
  return test_tagging_result , test_label_result

def DST_reset():
  global state
  for key in state.keys():
    if(type(state[key]).__name__ == 'dict'):
      for slot in state[key].keys():
        state[key][slot] = ''
    else:
       state[key] = ''
  waitConfirm = []
  for x in range(intents.__len__()):
    intents[x] = ''
  #for x in range(observation.__len__()):
  #  observation[x] = []
  global formerState
  formerState = [0,0,0,0,0,0,0,0,0,0,0]

def dialogStateTracking(tokens,test_tagging_result,test_label_result):#semantic frame
  slots = {'CATEGORY':'' ,'RESTAURANTNAME':'' ,'LOCATION':'' ,'TIME':''}
  for index_token in range(len(tokens)):
    if "B-CATEGORY" in test_tagging_result[0][index_token] or "I-CATEGORY" in test_tagging_result[0][index_token] :
      if(slots['CATEGORY'] == ""):
        slots['CATEGORY'] = str(slots['CATEGORY'] +tokens[index_token])
      else:
        slots['CATEGORY'] = str(slots['CATEGORY']+" "+tokens[index_token])
    elif "B-RESTAURANTNAME" in test_tagging_result[0][index_token] or "I-RESTAURANTNAME" in test_tagging_result[0][index_token]:
      if(slots['RESTAURANTNAME'] == ""):
        slots['RESTAURANTNAME'] = str(slots['RESTAURANTNAME']+tokens[index_token])
      else:
        slots['RESTAURANTNAME'] = str(slots['RESTAURANTNAME']+ " " +tokens[index_token])
    elif "B-LOCATION" in test_tagging_result[0][index_token] or "I-LOCATION" in test_tagging_result[0][index_token]:
      if(slots['LOCATION'] == ""):
        slots['LOCATION'] = str(slots['LOCATION'] +tokens[index_token])
      else:
        slots['LOCATION'] = str(slots['LOCATION'] +" "+tokens[index_token])
    elif "B-TIME" in test_tagging_result[0][index_token] or "I-TIME" in test_tagging_result[0][index_token]:
      if(slots['TIME'] == ""):
        slots['TIME'] = str(slots['TIME'] +tokens[index_token])
      else:
        slots['TIME'] = str(slots['TIME'] +" "+ tokens[index_token])

    observation.append([test_label_result[0] ,slots])
  print ("========================================================================")
  print ("\nLU Intent SLOTS:")
  print (slots,test_label_result[0])
  return slots


def dialogPolicy(formerPolicyGoodOrNot,userInput):
  search = SearchDB('140.112.49.151' ,'foodbot' ,'welovevivian' ,'foodbotDB')
  sys_act = {'intent':'' ,'content':'','currentState':''}
  slots = {'CATEGORY':'' ,'RESTAURANTNAME':'' ,'LOCATION':'' ,'TIME':'' ,'TIMES':''}
  needConfirm = False
  needInform = False
  sys_act['content'] = {}
  sys_act['currentstate'] = {}

  global waitConfirm
  global dialogNum
  global changeRestNum
  global notfoundNum
  global successNum
  
  if waitConfirm.__len__() != 0 and waitConfirm[-1][0] == 'confirm' and observation[-1][0] != 'Confirm':
    waitConfirm.pop(-1)

  state['Confirm'] = False
  state['Wrong'] = False
  state['Get_Another_Restaurant'] = False

  if observation[-1][0] == 'Confirm':
    state['Confirm'] = True
    if waitConfirm.__len__() != 0:
      if waitConfirm[-1][0] == 'confirm':
        needInform = True

      else:
        for x in range(1 ,11):
          if waitConfirm[-x][0] == intents[-1]:
            for key in waitConfirm[-x][1].keys():
              if key in state[intents[-1]]:
                state[intents[-1]][key] = waitConfirm[-x][1][key]
            waitConfirm.pop(-x)
            break
  elif observation[-1][0] == 'Wrong':
    #waitConfirm = []
    #Really? should we reset here?
    state['Wrong'] = True
    dialogNum += 1
    DST_reset()
    print ("!!WrongDetected!!")

  elif observation[-1][0] == 'Inform':
    if intents[-1] == 'Get_Restaurant':
      changeRestNum = 0
    for key in observation[-1][1].keys():
      if observation[-1][1][key] != '' and key in state[intents[-1]]:
        state[intents[-1]][key] = observation[-1][1][key]
    '''
    for key in observation[-1][1].keys():
      if observation[-1][1][key] != '' and key in state[intents[-1]]:
        if state[intents[-1]][key] != '':
          needConfirm = True
    
    if needConfirm:
      needConfirm = False
      sys_act['intent'] = 'confirm'
      for key in observation[-1][1].keys():
        if observation[-1][1][key] != '':
          sys_act['content'][key] = observation[-1][1][key]
      waitConfirm.append([intents[-1] ,sys_act['content']])
      #print ('wait confirm : ')
      #print (waitConfirm[-1])
    
    else:
      for key in observation[-1][1].keys():
        if observation[-1][1][key] != '' and key in state[intents[-1]]:
          if state[intents[-1]][key] == '':
            state[intents[-1]][key] = observation[-1][1][key]
    '''
  elif observation[-1][0] == 'Get_Another_Restaurant':
    if state['Get_Restaurant']['CATEGORY'] != '' and state['Get_Restaurant']['LOCATION'] != '':
      state['Get_Another_Restaurant'] = True
      changeRestNum += 1
  else:    
    if observation[-1][0] == 'Get_Restaurant':
      intents.append('Get_Restaurant')
      for key in observation[-1][1].keys():
        if observation[-1][1][key] != '' and key in state[observation[-1][0]]:
          state['Get_Restaurant'][key] = observation[-1][1][key]

    elif observation[-1][0] == 'Get_location':
      intents.append('Get_location')
      for key in observation[-1][1].keys():
        if observation[-1][1][key] != '' and key in state[observation[-1][0]]:
          state['Get_location'][key] = observation[-1][1][key]

    elif observation[-1][0] == 'Get_rating':
      intents.append('Get_rating')
      for key in observation[-1][1].keys():
        if observation[-1][1][key] != '' and key in state[observation[-1][0]]:
          state['Get_rating'][key] = observation[-1][1][key]

  stateList.append(state)
  print ('Now state : ' ,state)
  sys_act['currentstate'] = state

  #translate state to vector
  vector = [[0]*11,['Get_Restaurant','','','','Get_location','','Get_rating','','Get_Another_Restaurant','Confirm','Wrong']]
  if state['Wrong'] == True:
      vector[0][10] = 1
  elif observation[-1][0] == 'Get_Restaurant':
    vector[0][0] = 1
  elif observation[-1][0] == 'Get_location':
    vector[0][4] = 1
  elif observation[-1][0] == 'Get_rating':
    vector[0][6] = 1
  elif observation[-1][0] == 'Get_Another_Restaurant':
    vector[0][8] = 1
  elif observation[-1][0] == 'Confirm':
    vector[0][9] = 1
  #elif observation[-1][0] == 'Wrong':
  #  vector[0][10] = 1
  if state['Get_Restaurant']['LOCATION'] != '':
    vector[0][1] = 1
    vector[1][1] = state['Get_Restaurant']['LOCATION']
  if state['Get_Restaurant']['CATEGORY'] != '':
    vector[0][2] = 1
    vector[1][2] = state['Get_Restaurant']['CATEGORY']
  if state['Get_Restaurant']['TIME'] != '':
    vector[0][3] = 1
    vector[1][3] = state['Get_Restaurant']['TIME']
  if state['Get_location']['RESTAURANTNAME'] != '':
    vector[0][5] = 1
    vector[1][5] = state['Get_location']['RESTAURANTNAME']
  if state['Get_rating']['RESTAURANTNAME'] != '':
    vector[0][7] = 1
    vector[1][7] = state['Get_rating']['RESTAURANTNAME']
  #===============
  # input vector[0] : bits
  #     vector[1] : value
  #      DQN
  #  Update model first
  #  Then, make decisions
  # output action
  #===============
  global action
  global formerState
  feedbackReward  = 0
  currentState = vector[0]
  if userInput == 'end' and formerPolicyGoodOrNot == True:
    currentState = [0,0,0,0,0,0,0,0,0,0,0] # terminate state
  if userInput == 'end' and formerPolicyGoodOrNot == False:
    currentState = [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1]# terminate state
  if formerPolicyGoodOrNot == True:
    feedbackReward  = 1
  if len(set(formerState)) == 1:
    action = -1
  print ("###############################################")
  print("Former State: ", formerState)
  print("former action: ", action)

  #TODO
  # update the model(reward, currentState, formerState )
  # Has connected with the RL agent GRPC at beginning
  request = FoodBotRLAgent_pb2.EnvornmentInfomration(formerState = formerState ,currentState= currentState,rewardForTheFormer = feedbackReward,formerAction = action ,shouldTerminate = False)
  policy = stub.GetRLResponse(request)
  print ("RL agent Policy Choice:",policy.policyNumber)
  action = policy.policyNumber
  formerState = currentState
  print("current State: ", currentState)
  print("RewardForformerAction: ",formerPolicyGoodOrNot)
  print ("###############################################")


  #request location
  if action == 0:
    if intents[-1] == 'Get_Restaurant':
      sys_act['intent'] = 'request'
      sys_act['content'] = {'LOCATION':''}
    else:
      sys_act['intent'] = 'not_a_good_policy'
      sys_act['content'] = ''

  #request category
  elif action == 1:
    if intents[-1] == 'Get_Restaurant':
      sys_act['intent'] = 'request'
      sys_act['content'] = {'CATEGORY':''}
    else:
      sys_act['intent'] = 'not_a_good_policy'
      sys_act['content'] = ''

  #request time
  elif action == 2:
    if intents[-1] == 'Get_Restaurant':
      sys_act['intent'] = 'request'
      sys_act['content'] = {'TIME':''}
    else:
      sys_act['intent'] = 'not_a_good_policy'
      sys_act['content'] = ''

  #request restaurant name
  elif action == 3: 
    if intents[-1] == 'Get_rating' or intents[-1] == 'Get_location':
      sys_act['intent'] = 'request'
      sys_act['content'] = {'RESTAURANTNAME':''}
    else:
      sys_act['intent'] = 'not_a_good_policy'   
      sys_act['content'] = ''

  #inform Get_restaurant
  elif action == 4:
    if slots['RESTAURANTNAME'] != '' and slots['LOCATION'] != '':
      sys_act['intent'] = 'inform'
      for key in state[intents[-1]].keys():
        slots[key] = state[intents[-1]][key]
      sys_act['content'] = search.grabData(intents[-1] ,slots)
      dialogNum += 1
      if sys_act['content'] == '':
        sys_act['intent'] = 'not_found'
        notfoundNum += 1
      else:
        successNum += 1
      for key in state[intents[-1]].keys():
        state[intents[-1]][key] = ''
      waitConfirm.pop(-1)
    else:
      sys_act['intent'] = 'not_a_good_policy'
      sys_act['content'] = ''

  #inform Get_Another_Restaurant
  elif action == 5: 
    if slots['RESTAURANTNAME'] != '' and slots['LOCATION'] != '':
      sys_act['intent'] = 'inform'
      for key in state[intents[-1]].keys():
        slots[key] = state[intents[-1]][key]
      slots['TIMES'] = changeRestNum
      sys_act['content'] = search.grabData(intents[-1] ,slots)
      dialogNum += 1
      if sys_act['content'] == '':
        sys_act['intent'] = 'not_found'
        notfoundNum += 1
      else:
        successNum += 1
      for key in state[intents[-1]].keys():
        state[intents[-1]][key] = ''
      waitConfirm.pop(-1)
    else:
      sys_act['intent'] = 'not_a_good_policy'
      sys_act['content'] = ''

  #inform Get_Rating
  elif action == 6:
    if slots['RESTAURANTNAME'] != '':
      sys_act['intent'] = 'inform'
      for key in state[intents[-1]].keys():
        slots[key] = state[intents[-1]][key]
      sys_act['content'] = search.grabData(intents[-1] ,slots)
      dialogNum += 1
      if sys_act['content'] == '':
        sys_act['intent'] = 'not_found'
        notfoundNum += 1
      else:
        successNum += 1
      for key in state[intents[-1]].keys():
        state[intents[-1]][key] = ''
      waitConfirm.pop(-1)
    else:
      sys_act['intent'] = 'not_a_good_policy'
      sys_act['content'] = ''     

  #inform Get_Location
  elif action == 7:
    if slots['RESTAURANTNAME'] != '':
      sys_act['intent'] = 'inform'
      for key in state[intents[-1]].keys():
        slots[key] = state[intents[-1]][key]
      sys_act['content'] = search.grabData(intents[-1] ,slots)
      dialogNum += 1
      if sys_act['content'] == '':
        sys_act['intent'] = 'not_found'
        notfoundNum += 1
      else:
        successNum += 1
      for key in state[intents[-1]].keys():
        state[intents[-1]][key] = ''
      waitConfirm.pop(-1)
    else:
      sys_act['intent'] = 'not_a_good_policy'
      sys_act['content'] = ''   

  #confirm_restaurant
  elif action == 8:
    if intents[-1] == 'Get_Restaurant' and state['Get_Restaurant']['LOCATION'] != '' and state['Get_Restaurant']['CATEGORY'] != '' and state['Get_Restaurant']['TIME'] != '':
      sys_act['intent'] = 'confirm_restaurant'
      for key in state[intents[-1]].keys():
        sys_act['content'][key] = state[intents[-1]][key]
      waitConfirm.append(['confirm' ,sys_act['content']])
    else:
      sys_act['intent'] = 'not_a_good_policy'
      sys_act['content'] = ''        

  #confirm_info
  elif action == 9:
    if (intents[-1] == 'Get_rating' and state['Get_rating']['RESTAURANTNAME'] != '') or (intents[-1] == 'Get_location' and state['Get_location']['RESTAURANTNAME'] != ''):
      sys_act['intent'] = 'confirm_info'
      for key in state[intents[-1]].keys():
        sys_act['content'][key] = state[intents[-1]][key]
      sys_act['content']['LOCATION'] = ''
      waitConfirm.append(['confirm' ,sys_act['content']])
    else:
      sys_act['intent'] = 'not_a_good_policy'
      sys_act['content'] = ''  
      
  #wrong
  elif action == 10:
    return ''

  '''
  if sys_act['intent'] == 'wrong':
    return ''
  elif sys_act['intent'] != 'confirm':     
    if intents[-1] == 'Get_Restaurant':

      if state[intents[-1]]['LOCATION'] == '':
        sys_act['intent'] = 'request'
        sys_act['content'] = {'LOCATION':''}
      
      elif state[intents[-1]]['CATEGORY'] == '':
        sys_act['intent'] = 'request'
        sys_act['content'] = {'CATEGORY':''}
      
      #elif state[intents[-1]]['TIME'] == '':
      #  sys_act['intent'] = 'request'
      #  sys_act['content'] = {'time':''}

      elif needInform:
        needInform = False
        sys_act['intent'] = 'inform'
        for key in state[intents[-1]].keys():
          slots[key] = state[intents[-1]][key]
        slots['TIMES'] = changeRestNum
        sys_act['content'] = search.grabData(intents[-1] ,slots)
        dialogNum += 1
        if sys_act['content'] == '':
          sys_act['intent'] = 'not_found'
          notfoundNum += 1
        else:
          successNum += 1
        for key in state[intents[-1]].keys():
          state[intents[-1]][key] = ''
        waitConfirm.pop(-1)

      else:
        sys_act['intent'] = 'confirm_restaurant'
        for key in state[intents[-1]].keys():
          sys_act['content'][key] = state[intents[-1]][key]
        waitConfirm.append(['confirm' ,sys_act['content']])
  
    
    elif intents[-1] == 'Get_location':

      if state[intents[-1]]['RESTAURANTNAME'] == '':
        sys_act['intent'] = 'request'
        sys_act['content'] = {'RESTAURANTNAME':''}

      elif needInform:
        needInform = False
        sys_act['intent'] = 'inform'
        for key in state[intents[-1]].keys():
          slots[key] = state[intents[-1]][key]
        sys_act['content'] = search.grabData(intents[-1] ,slots)
        dialogNum += 1
        if sys_act['content'] == '':
          sys_act['intent'] = 'not_found'
          notfoundNum += 1
        else:
          successNum += 1
        for key in state[intents[-1]].keys():
          state[intents[-1]][key] = ''
        waitConfirm.pop(-1)

      else:
        sys_act['intent'] = 'confirm_info'
        for key in state[intents[-1]].keys():
          sys_act['content'][key] = state[intents[-1]][key]
        sys_act['content']['LOCATION'] = ''
        waitConfirm.append(['confirm' ,sys_act['content']])

    elif intents[-1] == 'Get_rating':

      if state[intents[-1]]['RESTAURANTNAME'] == '':
        sys_act['intent'] = 'request'
        sys_act['content'] = {'RESTAURANTNAME':''}

      elif needInform:
        needInform = False
        sys_act['intent'] = 'inform'
        for key in state[intents[-1]].keys():
          slots[key] = state[intents[-1]][key]
        sys_act['content'] = search.grabData(intents[-1] ,slots)
        dialogNum += 1
        if sys_act['content'] == '':
          sys_act['intent'] = 'not_found'
          notfoundNum += 1
        else:
          successNum += 1
        for key in state[intents[-1]].keys():
          state[intents[-1]][key] = ''
        waitConfirm.pop(-1)

      else:
        sys_act['intent'] = 'confirm_info'
        for key in state[intents[-1]].keys():
          sys_act['content'][key] = state[intents[-1]][key]
        sys_act['content']['RATING'] = ''
        waitConfirm.append(['confirm' ,sys_act['content']])

    else:
      print ("I don't know what to say")
  '''
  print ('Policy system action : ' ,sys_act)
  return sys_act

def nlg(sem_frame, bot):
  if bot == 1: #for bot  
    if sem_frame == '':
      return ''
    if sem_frame["intent"] == "request":
      keys = sem_frame["content"].keys()
      sentence = ""
      if "CATEGORY" in keys:
        sentence = random.choice(request_category_pattern) + " "
      if "LOCATION" in keys:
        sentence = sentence + random.choice(request_location_pattern) + " "
      if "TIME" in keys:
        sentence = sentence + random.choice(request_time_pattern)
      if "RESTAURANTNAME" in keys:
        sentence = "Which restaurant again, please?"
  
    if sem_frame["intent"] == "confirm_restaurant":
      keys = sem_frame["content"].keys()
      sentence = "You're looking for a "
      if "CATEGORY" in keys:
        sentence = sentence + sem_frame["content"]["CATEGORY"] + " restaurant"
      else:
        sentence = sentence + "restaurant"
      if "LOCATION" in keys and sem_frame["content"]["LOCATION"]:
        sentence = sentence + " in " + sem_frame["content"]["LOCATION"]
      if "TIME" in keys and sem_frame["content"]["TIME"]:
        sentence = sentence + " for " + sem_frame["content"]["TIME"]
      sentence = sentence + ", right?"

    if sem_frame["intent"] == "confirm_info":
      sentence = "You're looking for "
      if "RATING" in sem_frame["content"].keys():
        sentence = sentence + "the rating of " + sem_frame["content"]["RESTAURANTNAME"]
      if "LOCATION" in sem_frame["content"].keys():
        sentence = sentence + "the location of " + sem_frame["content"]["RESTAURANTNAME"]
      sentence = sentence + ", right?"
    
    if sem_frame["intent"] == "inform":
      #for recommendation
      if "RESTAURANTNAME" in sem_frame["content"].keys():
        sentence = random.choice(recommend_pattern)
        sentence = sentence.replace("RESTAURANT_NAME", sem_frame["content"]["RESTAURANTNAME"])
        sentence = sentence.capitalize() + " It's in " + sem_frame["content"]["LOCATION"] + "."
      
      else:
        #for restaurant info
        if "LOCATION" in sem_frame["content"].keys():
          sentence = "It's here: " + sem_frame["content"]["LOCATION"] + "."
        if "RATING" in sem_frame["content"].keys():
          sentence = "Its rating is " + sem_frame["content"]["RATING"] + "."

    if sem_frame["intent"] == "not_found":
      sentence = "Sorry! I don't have the information you're looking for. Please try another one."
  
    if not sem_frame["intent"]:
      sentence = "Sorry, I don't understand! Please try again..."
    elif sem_frame["intent"] == 'not_a_good_policy':
      sentence = "What?? Please try again..."

  else:
    sentence = "The variable bot should be 1!"
  
  return sentence

class FoodbotRequest(FoodBot_pb2.FoodBotRequestServicer):
  """Provides methods that implement functionality of route guide server."""
  def GetResponse (self, request, context):
    print ("Request from simuser:")
    print (request.response)
    outputFromSim = json.loads(request.response)
    print (outputFromSim)
    if(type(outputFromSim).__name__ == 'unicode'):
      outputFromSim = json.loads(outputFromSim)
    print (type(outputFromSim)) #<type 'unicode'> <type 'dict'>
    realSemanticFrame = outputFromSim["semantic_frame"]
    userInput = outputFromSim["nlg_sentence"].lower()
    #goodpolicy = outputFromSim['goodpolicy']
    if 'goodpolicy' in outputFromSim.keys():        #come from simulated user
      if userInput == 'end': # or sim user said it's not a good policy
        #reset the dialog state.'
        #get the reward for the former action(The current state and the former state should be the same. But it's ok here. The current action won't be executed)
        policyFrame = dialogPolicy(outputFromSim['goodpolicy'],userInput)
        DST_reset()
        return FoodBot_pb2.Sentence(response = "")# for sim-user to initial a new conversation
      else:
        userInput = userInput.replace("?", "")
        userInput = userInput.replace(".", "")
        userInput = userInput.replace(",", "")
        userInput = userInput.replace("!", "")
    else:        #come from web user
      if userInput == 'end': # or sim user said it's not a good policy
        #reset the dialog state.'
        DST_reset()
        return FoodBot_pb2.Sentence(response = "")
      else:
        userInput = userInput.replace("?", "")
        userInput = userInput.replace(".", "")
        userInput = userInput.replace(",", "")
        userInput = userInput.replace("!", "")

    predSlot = []
    policyFrame = []
    nlg_sentence = []

    test_tagging_result,test_label_result = languageUnderstanding(userInput) 
    predSlot = dialogStateTracking(userInput.split(),test_tagging_result,test_label_result)
    if 'goodpolicy' in outputFromSim.keys():
      pass
    else:
      outputFromSim['goodpolicy'] = True # assume users give a comfirmative attitude if they continue to talk OR they will type 'end'
    policyFrame = dialogPolicy(outputFromSim['goodpolicy'],userInput)
    nlg_sentence = nlg(policyFrame,1)

    #Calculate the LU accuracy:
    if realSemanticFrame != "":
      LURight = semanticComparison(realSemanticFrame,test_label_result[0],predSlot)
      global LURightCount
      global LUWrongCount
      if(LURight == True):
        LURightCount = LURightCount+1
      else:
        LUWrongCount = LUWrongCount+1
      TotalTruns = 1.0*(LUWrongCount + LURightCount)

      fp = open(textFileName ,'w')
      fp.write(' LU Accuracy Rate : %f\n Total turns: %f' %(LURightCount/TotalTruns,TotalTruns) )
      fp.close()

    if dialogNum != 0:
      fp = open(successRateFileName ,'w')
      fp.write('Policy Success Rate : %f %f %f\n' %(successNum/dialogNum, successNum, dialogNum))
      fp.write('DB Not Found Rate : %f %f %f\n ' %(notfoundNum/dialogNum, notfoundNum, dialogNum))
      fp.close()

    #dictionary to jsonstring
    policyFrameString = json.dumps(policyFrame)
    return FoodBot_pb2.outSentence(response_nlg = nlg_sentence,response_policy_frame = policyFrameString)

def semanticComparison(realSem,predIntent,predSlots):
  print ("---------")
  print("real seam:",realSem)
  print("pred seam:",predSlots )
  print("pred inte:",predIntent )
  print ("---------")
  
  #conversion
  if predIntent == "Confirm":
    predIntent = "yes"

  if(predIntent.lower() != realSem["intent"].lower()):
    return False
  elif(predSlots["LOCATION"].lower() != realSem["location"].lower()):
    return False
  elif(predSlots["TIME"].lower() != realSem["time"].lower()):
    return False
  elif(predSlots["CATEGORY"].lower() != realSem["category"].lower()):
    return False
  elif(predSlots["RESTAURANTNAME"].lower() != realSem["restaurantname"].lower()):
    return False
  else:
    return True


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
   
  global sess 
  global vocab 
  global rev_vocab
  global tag_vocab 
  global rev_tag_vocab 
  global label_vocab 
  global rev_label_vocab 
  vocab, rev_vocab = data_utils.initialize_vocabulary(vocab_path)
  tag_vocab, rev_tag_vocab = data_utils.initialize_vocabulary(tag_vocab_path)
  label_vocab, rev_label_vocab = data_utils.initialize_vocabulary(label_vocab_path)
    
  global sess
  sess = tf.Session()
  # Create model.
  print("Max sequence length: %d." % _buckets[0][0])
  print("Creating %d layers of %d units." % (FLAGS.num_layers, FLAGS.size))
  global model_test
  model, model_test = create_model(sess, len(vocab), len(tag_vocab), len(label_vocab))
  print ("Creating model with source_vocab_size=%d, target_vocab_size=%d, and label_vocab_size=%d." % (len(vocab), len(tag_vocab), len(label_vocab)))
  
  # The model has been loaded.
  server = grpc.server(futures.ThreadPoolExecutor(max_workers=3))
  #Service_OpenFace_pb2.add_openfaceServicer_to_server(Servicer_openface(), server)
  FoodBot_pb2.add_FoodBotRequestServicer_to_server(FoodbotRequest(),server)
  server.add_insecure_port('[::]:50055')
  server.start()
  print ("GRCP Server is running. Press any key to stop it.")
  try:
    while True:
      # openface_GetXXXXXX will be responsed if any incoming request is received.
      time.sleep(48*60*60)
  except KeyboardInterrupt:
    server.stop(0)

        

  

def main(_):
    #train()
    testing()
if __name__ == "__main__":
  tf.app.run()


