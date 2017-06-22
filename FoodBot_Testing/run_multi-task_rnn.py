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
import glob
import random

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


# NLG
'''
import argparse
sys.path.append('RNNLG/')
from generator.net import Model
'''

#global vars
model_test =  0
sess = 0
vocab = 0
rev_vocab = 0
tag_vocab = 0
rev_tag_vocab = 0
label_vocab = 0
rev_label_vocab = 0
state = {
      'user':{
        'request_restaurant':'',
        'inform':'',
        'request_address':'',
        'request_score':'',
        'request_review':'', 
        'request_price':'',
        'request_time':'',
        'request_phone':'',
        'request_smoke':'',
        'request_wifi':'',
        'confirm':'',
        'reject':'',
        'hi':'',
        'thanks':'',
        'goodbye':'',
        'restaurant_name':'', 'area':'', 'category':'', 'score':'', 'price':''
      },
      'agent':{
        'restaurant_name':'',
        'confirm_info':'',
        'confirm_restaurant':''
      }
    }
stateList = []
changeRestNum = 0
intents = collections.deque(maxlen=2)
waitConfirm = []
dialogNum = 0.0
successNum = 0.0
notfoundNum = 0.0

textFileName = "record_"+str(time.time())
################## below for state
formerState = [2,2,2,2,2,2,2,2,2,2,2] #To remember the former state, 222222 is the start state

##################below for nlg
action = -1
NewDialog = True

##############
memory = dict()
good = []
goal = ''
last_intent = ''


#patterns for agent
pattern_dict = dict()
file_list = glob.glob("./sentence_pattern/agent/*.txt")
for txt in file_list:
  f = open(txt, 'r')
  key = txt.split('/')[3].split('.')[0]
  temp = f.read().split('\n')
  temp = [sent for sent in temp if sent != '']
  pattern_dict[key] = temp

with open("./sentence_pattern/agent/pic_dict.json", 'r') as f:  
  pic_dict = json.load(f)


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
  if(test_label_result[0] == 'confirm'):
    if("B-Wrong" in test_tagging_result[0]):
      test_label_result[0] = "reject"
    if("B-HI" in test_tagging_result[0]):
      test_label_result[0] = "hi"
  return test_tagging_result , test_label_result

def DST_reset():
  global state,NewDialog, memory, goal
  goal = ''
  memory = dict()
  for key in state.keys():
    if(type(state[key]).__name__ == 'dict'):
      for slot in state[key].keys():
        state[key][slot] = ''
    else:
       state[key] = ''
  NewDialog = True

def dialogStateTracking(tokens,test_tagging_result,test_label_result,sem_frame_from_sim):#semantic frame
  global state, memory, goal
  slots = {'restaurant_name':'', 'area':'', 'category':'', 'score':'', 'price':''}
  #reset intent
  for key in state['user'].keys():
    if 'request' in key or 'inform' in key or key == 'hi' or key == 'thanks' or key == 'goodbye' or key == 'confirm' or key == 'reject':
      state['user'][key] = ''
  #from sim user
  if sem_frame_from_sim != '':
    if sem_frame_from_sim['intent'] in state['user']:
      state['user'][sem_frame_from_sim['intent']] = 'True'
    if 'category' in sem_frame_from_sim:
      slots['category'] = sem_frame_from_sim['category']
    if 'area' in sem_frame_from_sim:
      slots['area'] = sem_frame_from_sim['area']
    if 'price' in sem_frame_from_sim:
      slots['price'] = sem_frame_from_sim['price']
    if 'score' in sem_frame_from_sim:
      slots['score'] = sem_frame_from_sim['score']
    if 'name' in sem_frame_from_sim:
      slots['restaurant_name'] = sem_frame_from_sim['name']

    for key in slots.keys():
        if slots[key] != '':
          state['user'][key] = slots[key]

  #from web user
  else:
    for index_token in range(len(tokens)):
      
      if "B-CATEGORY" in test_tagging_result[0][index_token] or "I-CATEGORY" in test_tagging_result[0][index_token] :
        if(slots['category'] == ""):
          slots['category'] = str(slots['category'] +tokens[index_token])
        else:
          slots['category'] = str(slots['category']+" "+tokens[index_token])
      
      elif "B-RESTAURANTNAME" in test_tagging_result[0][index_token] or "I-RESTAURANTNAME" in test_tagging_result[0][index_token]:
        if(slots['restaurant_name'] == ""):
          print (slots['restaurant_name'])
          print (tokens[index_token])
          slots['restaurant_name'] = str(slots['restaurant_name']+tokens[index_token])
        else:
          slots['restaurant_name'] = str(slots['restaurant_name']+ " " +tokens[index_token])
      
      elif "B-AREA" in test_tagging_result[0][index_token] or "I-AREA" in test_tagging_result[0][index_token]:
        if(slots['area'] == ""):
          slots['area'] = str(slots['area'] +tokens[index_token])
        else:
          slots['area'] = str(slots['area'] +" "+tokens[index_token])

      elif "B-SCORE" in test_tagging_result[0][index_token] or "I-SCORE" in test_tagging_result[0][index_token]:
        if(slots['score'] == ""):
          slots['score'] = str(slots['score'] +tokens[index_token])
        else:
          slots['score'] = str(slots['score'] +" "+ tokens[index_token])

      elif "B-PRICE" in test_tagging_result[0][index_token] or "I-PRICE" in test_tagging_result[0][index_token]:
        if(slots['price'] == ""):
          slots['price'] = str(slots['price'] +tokens[index_token])
        else:
          slots['price'] = str(slots['price'] +" "+ tokens[index_token])

    for key in slots.keys():
      if slots[key] != '':
        if key == 'restaurant_name':
          memory['name'] = slots[key]
        else:
          memory[key] = slots[key]
        state['user'][key] = slots[key]

    if test_label_result[0] in state['user']:
      state['user'][test_label_result[0]] = 'True'
   # if goal == '' and test_label_result[0] == 'inform':
   #   goal == 'request_restaurant'
   #   test_label_result[0] = 'request_restaurant'
    if goal == '' and test_label_result[0] not in ['hi','confirm','goodbye', 'thanks','reject','inform']:
      goal = test_label_result[0]  
    
    memory['intent'] = test_label_result[0]

  print ("========================================================================")
  print ("\nLU Intent SLOTS:")
  #print (slots,test_label_result[0])
  return slots

def policyChecker():
	#print("This is sys_act",sys_act)
	#print("This is sys_act[content]",sys_act['content'])
	#print("This is sys_act[intent]",sys_act['intent'])
	#print("This is sys_act[currentState]",sys_act['currentstate'])
	global good
	global memory,goal,last_intent

	soso = []
	if memory["intent"] in ["goodbye", "thanks"]:
		good = ["goodbye"]

	
	elif memory["intent"] == "hi":
		good = ["hi"]
	
#######for test#############
	elif memory["intent"] == "confirm":
		good = ["inform_" + goal.split('_')[1]]
############################
	
	elif memory["intent"] == "reject":
		if last_intent == 'reqmore':
			good = ['confirm_restaurant']
		if last_intent in ['confirm_info','confirm_restaurant']:
			good = ['show_table']
  
	else:
		if goal == "request_restaurant":
			if len(memory.keys()) <=3:
				keys = memory.keys()
				keys.remove("intent")
				request_list = ["area", "category"]
				good = ["reqmore"] + ["request_"+item for item in request_list if item not in keys]
			else:
				good = ["confirm_restaurant", "reqmore"]
				soso = ["reqmore"]
		elif "request" in goal:
			if len(memory.keys()) <=1:
        #keys = memory.keys()
        #keys.remove("intent")
        #request_list = ["area", "name"]
        #good = ["request_"+item for item in request_list if item not in keys]
				good = ["request_name"]
			else:
				good = ["confirm_info"]
	'''
	if sys_act["policy"] in good:
		reward = 5
	elif sys_act["policy"] in soso:
		reward = 2
	else:
		reward = 0
  '''
	print ('==========================')
	print ('memory : ', memory)
	print ('sys_act : ', last_intent)
	print ('good : ', good)
	return good


def simul_user(sys_act):
	global memory	
	global good
	global goodPolicy
	'''
	sys_act: {
			  "policy": "request_category",			  
			  }
	'''
	# initially randomly generated a sentence
	sys_act = json.loads(sys_act)
	print("sys_act from agent:",sys_act)
	sem_frame = dict()
	if sys_act["policy"] == "init":
		memory = dict()
		good = []
		goodPolicy = 0
		sem_frame["intent"] = random.choice(data_dict["intent"])
		if sem_frame["intent"] != "hi":
			keys = slot_dict[sem_frame["intent"]]
			for key in keys:
				dec = round(random.random())
				if "request" in sem_frame["intent"] and sem_frame["intent"] != "request_restaurant":
					dec = 1
				if dec == 1:
					sem_frame[key] = random.choice(data_dict[key])
		memory = sem_frame  #keep the memory

	#in the middle of the dialogue	
	else:
		#print("memory: ", memory)		
		# To see if the policy picked by DQN is reasonable
		goodPolicy = policyChecker(sys_act)
		print (goodPolicy)
		if goodPolicy == 0:
			returnList = dict()
			returnList["semantic_frame"] = dict()
			returnList["semantic_frame"]["intent"] = 'goodbye'
			returnList["goodpolicy"] = goodPolicy
			returnList["user_id"] = 'sim-user'
			returnList["nlg_sentence"] = ''
			#json_list = json.dumps(returnList)
			return returnList

################## hi #########################
		if sys_act["policy"] == "hi":
			intents = data_dict["intent"]
			intents.remove("hi")
			sem_frame["intent"] = random.choice(intents)
			keys = slot_dict[sem_frame["intent"]]
			for key in keys:
				dec = round(random.random())
				if dec == 1:
					sem_frame[key] = random.choice(data_dict[key])
			memory = sem_frame


################## request #########################
		elif "request" in sys_act["policy"]:
			sem_frame["intent"] = "inform"
			#print("content keys:", sys_act["content"].keys())
			key = sys_act["policy"].split('_')[1]
			sem_frame[key] = random.choice(data_dict[key])
			memory[key] = sem_frame[key]

		elif sys_act["policy"] == "reqmore":			
			dec = round(random.random())
			if len(memory.keys()) <= 3:
				sem_frame["intent"] = "inform"
				temp = [item for item in slot_dict[memory["intent"]] if item not in memory.keys()]
				key = random.choice(temp)
				sem_frame[key] = random.choice(data_dict[key])
				memory[key] = sem_frame[key]
			else:
				sem_frame["intent"] = "reject"
				memory["intent"] = "reject"
				good = ["confirm_restaurant"]

################## inform #########################
		elif "inform" in sys_act["policy"]:
			sem_frame["intent"] = random.choice(["goodbye", "thanks"])
					
################## confirm #########################		
		elif "confirm" in sys_act["policy"]:
			sem_frame["intent"] = "goodbye"
			if sys_act["policy"] == "confirm_info":
				if "info_name" not in sys_act.keys() or "name" not in sys_act.keys() or \
				  sys_act["info_name"] != memory["intent"].split('_')[1] or sys_act["name"] != memory["name"]:
					sem_frame["intent"] = "reject"
					good = ["show_table"]
			else:
				keys = memory.keys()
				keys.remove("intent")
				for key in keys:
					if key not in sys_act.keys() or (key in sys_act.keys() and memory[key] != sys_act[key]):
						sem_frame["intent"] = "reject"
						good = ["show_table"]
						break
			memory["intent"] = sem_frame["intent"]
				

		elif sys_act["policy"] == "show_table":
			sem_frame["intent"] = 'goodbye'
		
		else:
			sem_frame["intent"] = 'error'
	#nlg_sentence = nlg(sem_frame)

	returnList = dict()
	#returnList["nlg_sentence"] = nlg_sentence
	returnList["semantic_frame"] = sem_frame
	returnList["goodpolicy"] = goodPolicy
	returnList["user_id"] = 'sim-user'
	returnList["nlg_sentence"] = ''
	
	return returnList
	#return sem_frame, goodPolicy


def dialogPolicy(formerPolicyGoodOrNot):
  search = SearchDB('140.112.49.151' ,'foodbot' ,'welovevivian' ,'foodbotDB')
  slots = {'restaurant_name':'', 'area':'', 'category':'', 'score':'', 'price':''}
  sys_act = {}
  global state,NewDialog

  stateList.append(state)

  for key in state['user'].keys():
    if key in slots:
      slots[key] = state['user'][key]

  if slots['restaurant_name'] == '' and state['agent']['restaurant_name'] != '':
    slots['restaurant_name'] = state['agent']['restaurant_name']

  #translate state to vector
  vector = [[0]*23, ['request_restaurant', 'inform', 'request_address', 'request_score', 'request_review', 'request_price', 'request_time', 'request_phone', 'request_smoke', 'request_wifi', 'confirm', 'reject', 'hi', 'thanks', 'goodbye', '', '', '', '', '', '',  'confirm_info', 'confirm_restaurant']]
  if state['user']['goodbye'] != '':
    vector = [[0]*23, ['request_restaurant', 'inform', 'request_address', 'request_score', 'request_review', 'request_price', 'request_time', 'request_phone', 'request_smoke', 'request_wifi', 'confirm', 'reject', 'hi', 'thanks', 'goodbye', '', '', '', '', '', '', 'confirm_info', 'confirm_restaurant']]
  else:
    if state['user']['request_restaurant'] != '':
      vector[0][0] = 1
    if state['user']['inform'] != '':
      vector[0][1] = 1
    if state['user']['request_address'] != '':
      vector[0][2] = 1
    if state['user']['request_score'] != '':
      vector[0][3] = 1
    if state['user']['request_review'] != '':
      vector[0][4] = 1
    if state['user']['request_price'] != '':
      vector[0][5] = 1
    if state['user']['request_time'] != '':
      vector[0][6] = 1
    if state['user']['request_phone'] != '':
      vector[0][7] = 1
    if state['user']['request_smoke'] != '':
      vector[0][8] = 1
    if state['user']['request_wifi'] != '':
      vector[0][9] = 1
    if state['user']['confirm'] != '':
      vector[0][10] = 1
    if state['user']['reject'] != '':
      vector[0][11] = 1
    if state['user']['hi'] != '':
      vector[0][12] = 1
    if state['user']['thanks'] != '':
      vector[0][13] = 1
    if state['user']['restaurant_name'] != '':
      vector[0][15] = 1
      vector[1][15] = state['user']['restaurant_name']
    if state['user']['area'] != '':
      vector[0][16] = 1
      vector[1][16] = state['user']['area']
    if state['user']['category'] != '':
      vector[0][17] = 1
      vector[1][17] = state['user']['category']
    if state['user']['score'] != '':
      vector[0][18] = 1
      vector[1][18] = state['user']['score']
    if state['user']['price'] != '':
      vector[0][19] = 1
      vector[1][19] = state['user']['price']
    if state['agent']['restaurant_name'] != '':
      vector[0][20] = 1
      vector[1][20] = state['agent']['restaurant_name']
    if state['agent']['confirm_info'] != '':
      vector[0][21] = 1
    if state['agent']['confirm_restaurant'] != '':
      vector[0][22] = 1

  #===============
  # input vector[0] : bits
  #     vector[1] : value
  #      DQN
  #  Update model first
  #  Then, make decisions
  # output action
  #===============
  currentState = vector[0]
 #Notice:
  global formerState, action,last_intent
  if (NewDialog == True):
    action = -1
    NewDialog = False
    formerState = [0]

  #TODO
  # update the model(reward, currentState, formerState )
  # Has connected with the RL agent GRPC at beginning
  request = FoodBotRLAgent_pb2.EnvornmentInfomration(formerState = formerState ,currentState= currentState,rewardForTheFormer = formerPolicyGoodOrNot,formerAction = action ,shouldTerminate = False)
  policy = stub.GetRLResponse(request)
  #print ("RL agent Policy Choice:",policy.policyNumber)
  action = policy.policyNumber
  formerState = currentState

  goodpolicy = policyChecker()
  policy = goodpolicy[random.randint(0,goodpolicy.__len__()-1)]
  print("GoodPolicy:",policy)
  if(state['user']['goodbye']!=''):
    DST_reset()
    #return ''

  #print("current State: ", currentState)
  #print("RewardForformerAction: ",formerPolicyGoodOrNot)
  #print ("###############################################")

  # if end, reset the state.
  #if userInput == 'end':
  #  DST_reset()

  state['agent']['confirm_info'] = ''
  state['agent']['confirm_restaurant'] = ''
  #request area
  if policy == 'request_area':
    sys_act['policy'] = 'request_area'

  #request category
  elif policy == 'request_category':
    sys_act['policy'] = 'request_category'

  #request more
  elif policy == 'reqmore': 
    sys_act['policy'] = 'reqmore'

  #inform address
  elif policy == 'inform_address':
    if slots['restaurant_name'] != '':
      sys_act = search.grabData('inform_address', slots)     
    else:
      sys_act['policy'] = 'not_a_good_policy'

  #inform score
  elif policy == 'inform_score': 
    if slots['restaurant_name'] != '':
      sys_act = search.grabData('inform_score', slots)
    else:
      sys_act['policy'] = 'not_a_good_policy'

  #inform review
  elif policy == 'inform_review':
    if slots['restaurant_name'] != '':
      sys_act = search.grabData('inform_review', slots)
    else:
      sys_act['policy'] = 'not_a_good_policy'

  #inform restaurant
  elif policy == 'inform_restaurant':
    if state['user']['category'] != '' or state['user']['area'] != '' or state['user']['price'] != '' or state['user']['score'] != '':
      sys_act = search.grabData('inform_restaurant', slots)
    else:
      sys_act['policy'] = 'not_a_good_policy'

  #inform smoke
  elif 'inform_smoke' in policy:
    if slots['restaurant_name'] != '':
      sys_act = search.grabData('inform_smoke', slots)
    else:
      sys_act['policy'] = 'not_a_good_policy'

  #inform wifi
  elif 'inform_wifi' in policy:
    if slots['restaurant_name'] != '':
      sys_act = search.grabData('inform_wifi', slots)
    else:
      sys_act['policy'] = 'not_a_good_policy'

  #inform phone
  elif policy == 'inform_phone':
    if slots['restaurant_name'] != '':
      sys_act = search.grabData('inform_phone', slots)
    else:
      sys_act['policy'] = 'not_a_good_policy'

  #inform price
  elif policy == 'inform_price':
    if slots['restaurant_name'] != '':
      sys_act = search.grabData('inform_price', slots)
    else:
      sys_act['policy'] = 'not_a_good_policy'

  #inform business time
  elif policy == 'inform_time':
    if slots['restaurant_name'] != '':
      sys_act = search.grabData('inform_time', slots)
    else:
      sys_act['policy'] = 'not_a_good_policy'


  #confirm_restaurant
  elif policy == 'confirm_restaurant':
    if state['user']['category'] != '' or state['user']['area'] != '' or state['user']['price'] != '' or state['user']['score'] != '':
      state['agent']['confirm_restaurant'] = 'True'
      sys_act['policy'] = 'confirm_restaurant'
      if state['user']['category'] != '':
        sys_act['category'] = state['user']['category']
      if state['user']['area'] != '':
        sys_act['area'] = state['user']['area']
      if state['user']['price'] != '':
        sys_act['price'] = state['user']['price']
      if state['user']['score'] != '':
        sys_act['score'] = state['user']['score']
    else:
      sys_act['policy'] = 'not_a_good_policy'

  #confirm_info
  elif policy == 'confirm_info':
    if state['user']['restaurant_name'] != '':
      state['agent']['confirm_info'] = 'True'
      sys_act['policy'] = 'confirm_info'
      sys_act['name'] = state['user']['restaurant_name']
      for key in state['user'].keys():
        if 'request' in key and state['user'][key] != '':
          sys_act['info_name'] = key.split('_')[1]
    else:
      sys_act['policy'] = 'not_a_good_policy'

  #goodbye
  elif policy == 'goodbye':
    sys_act['policy'] = 'goodbye'

  #hi
  elif policy == 'hi':
    sys_act['policy'] = 'hi'

  elif policy == 'show_table':
    sys_act['policy'] = 'show_table'
    if slots['restaurant_name'] != '':
      sys_act['name'] = slots['restaurant_name']
    if slots['category'] != '':
      sys_act['category'] = slots['category']
    if slots['price'] != '':
      sys_act['price'] = slots['price']
    if slots['score'] != '':
      sys_act['score'] = slots['score']
    if slots['area'] != '':
      sys_act['area'] = slots['area']

  if (sys_act['policy'] == 'not_a_good_policy'):
    DST_reset()

  if("inform" in sys_act['policy']):
    print("alredy inform")
    DST_reset()

  print ('Policy system action : ' ,sys_act)
  last_intent = sys_act['policy']
  return sys_act

def nlg(original_sem_frame): 
  sem_frame = ''
  if original_sem_frame != '':
    sem_frame = original_sem_frame.copy()
  else:
    sem_frame = original_sem_frame

  return_list = dict()
  return_list["pic_url"] = ''
  if sem_frame == '':
    return_list["sentence"] = 'bye'
    return_list["pic_url"] = ''
    return return_list
  
  elif sem_frame["policy"] == "not_a_good_policy":
    sentence = "Sorry, I chose a bad policy. Please start again."

  elif sem_frame["policy"] == "show_table":
    return_list["pic_url"] = sem_frame
    return_list["pic_url"].pop("policy")
    sentence = """Help me! Which of these is/are mistaken?Ex. If category should be japanese and time should be tonight, please reply 'category:japanese,time:tonight' without quotation marks(').Thank you soooo much! If you have any question, please refer to the right side guideline"""

  elif sem_frame["policy"] in ["request_area", "request_category", "request_time", \
                              "request_name", "reqmore", "goodbye", "hi", "inform_smoke_yes", \
                              "inform_smoke_no", "inform_wifi_yes", "inform_wifi_no", "inform_no_match"]:
    sentence = random.choice(pattern_dict[sem_frame["policy"]]) 
  
  elif sem_frame["policy"] == "confirm_restaurant":
    sub_sent = ''
    keys = sem_frame.keys()
    keys.remove("policy")
    sentence = random.choice(pattern_dict[sem_frame["policy"]])
    if "category" in keys:
      sentence = sentence.replace('__', sem_frame["category"] + " restaurant__")
    else:
      sentence = sentence.replace('__', "restaurant__")
    if "area" in keys:
      sub_sent += ' in ' + sem_frame["area"]      
    if "score" in keys:
      sub_sent += ' whose score is higher than ' + sem_frame["score"]
    if "price" in keys:
        sub_sent += ' with the price around ' + sem_frame["price"]
    sentence = sentence.replace('__', sub_sent)

  else:
    keys = sem_frame.keys()
    keys.remove("policy")
    sentence = random.choice(pattern_dict[sem_frame["policy"]])
    for key in keys:
      sentence = sentence.replace("SLOT_"+key.upper(), sem_frame[key])

  
  print (sem_frame)
  if "policy" in sem_frame.keys() and sem_frame["policy"] in pic_dict.keys():
    pic_url = random.choice(pic_dict[sem_frame["policy"]])
    return_list["pic_url"] = pic_url
  
  return_list["sentence"] = sentence.capitalize()
  #json_list = json.dumps(return_list)

  return return_list
  

class FoodbotRequest(FoodBot_pb2.FoodBotRequestServicer):
  """Provides methods that implement functionality of route guide server."""
  def GetResponse (self, request, context):
    #--Reborn 
    #From sim_user:
    good_policy   = request.good_policy #0,1,2,3,4,5...
    nlg_sentence = request.nlg_sentence
    user_id = request.user_id
    sem_frame_from_sim = request.semantic_frame
    sem_frame_from_sim = json.loads(sem_frame_from_sim)
    if user_id != 'sim-user' :
      # from web user
      # LUResult   = LU (nlg_sentence)
      userInput = nlg_sentence.lower()
      userInput = userInput.replace("?", " ")
      userInput = userInput.replace(".", " ")
      userInput = userInput.replace(",", " ")
      userInput = userInput.replace("!", " ")
      test_tagging_result,test_label_result = languageUnderstanding(userInput) 
      print("UserInput:",userInput)
      print("LU Result:",test_label_result[0])
      print("LU Slots: ",test_tagging_result)
      dialogStateTracking(userInput.split(),test_tagging_result,test_label_result,'')#user id
      selectedPolicy =  dialogPolicy(-1)
      # Nlg_result = NLG(Policy, DST_Result_Content)
      nlg_result = nlg(selectedPolicy)
      return FoodBot_pb2.outSentence(response_nlg = nlg_result['sentence'],response_policy_frame = json.dumps(selectedPolicy),url =json.dumps(nlg_result['pic_url']) )

      # Return to the web.
    else:
      # from sim user
      # LUResult = LU (nlg_sentence)
      '''
      userInput = nlg_sentence.lower()
      userInput = userInput.replace("?", " ")
      userInput = userInput.replace(".", " ")
      userInput = userInput.replace(",", " ")
      userInput = userInput.replace("!", " ")
      test_tagging_result,test_label_result = languageUnderstanding(userInput) 
      '''
      print ("Semantic frame from Sim User: ",sem_frame_from_sim )
      print ("===========NLG from Sim User: ",nlg_sentence )
      print ("========Sim user from Policy: ",good_policy )
      print ("==============ID of Sim User: ",user_id )


      dialogStateTracking('','','',sem_frame_from_sim)#user id

      selectedPolicy =  dialogPolicy(good_policy)
      print("before:",selectedPolicy)
      print("nlg:", nlg(selectedPolicy) )
      print("after:",selectedPolicy)

      return FoodBot_pb2.outSentence(response_nlg = '',response_policy_frame = json.dumps(selectedPolicy),url = '')

      # Return to the sim_user with Policy(frame_level), DST(frame_level) 
      
      # in Json String
    

    #==Reborn
    '''
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
    FromeWeb = True
    if 'goodpolicy' in outputFromSim.keys():        #come from simulated user
      FromeWeb = False
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
      FromeWeb = True
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
      outputFromSim['goodpolicy'] = 0 # assume users give a comfirmative attitude if they continue to talk OR they will type 'end'
    policyFrame = dialogPolicy(outputFromSim['goodpolicy'],userInput)
    if(policyFrame == ''):
      DST_reset()
    if FromeWeb == True:
      if(policyFrame == ''):
        nlg_sentence = ''
      elif(policyFrame["intent"] == 'not_a_good_policy'):
        nlg_sentence = "What?? Please try again..."
      else:
        #Run policy converter
        #RNNLGModel = Model(None,None)
        nlg_sentence = nlg(policyFrame)
    else:
      nlg_sentence = nlg(policyFrame)
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
    print("nlg sentence:",nlg_sentence )
    print("frame:",policyFrameString )
    return FoodBot_pb2.outSentence(response_nlg = nlg_sentence,response_policy_frame = policyFrameString)
  '''
def converter(policyframe):
  query = ''
  if policyframe['intent'] == 'request':
    if 'LOCATION' in policyframe['content'].keys():
      query = '?request(location)'
    elif 'TIME' in policyframe['content'].keys():
      query = '?request(time)'
    elif 'RESTAURANTNAME' in policyframe['content'].keys():
      query = '?request(restaurantname)'
    elif 'CATEGORY' in policyframe['content'].keys():
      query = '?request(category)'
  
  elif policyframe['intent'] == 'confirm_restaurant':
    query = 'confirm_restaurant(category='+policyframe['content']['CATEGORY']+';time=sometime;location='+policyframe['content']['LOCATION']+')'
  
  elif policyframe['intent'] == 'confirm_info':
    if intents[-1] == 'Get_rating':
      query = 'confirm_info(info=rating;name='+policyframe['content']['RESTAURANTNAME']+')'
    elif intents[-1] == 'Get_location':
      query = 'confirm_info(info=location;name='+policyframe['content']['RESTAURANTNAME']+')'
  
  elif policyframe['intent'] == 'not_found':
    query = 'inform_no_match(category='+policyframe['content']['CATEGORY']+';time=sometime;location='+policyframe['content']['LOCATION']+')'

  elif policyframe['intent'] == 'inform':
    if 'RESTAURANTNAME' in policyframe['content'].keys():
      query = 'inform(name='+policyframe['content']['RESTAURANTNAME']+';address='+policyframe['content']['LOCATION']+')'
    else:
      if intents[-1] == 'Get_location':
        query = 'inform(address='+policyframe['content']['LOCATION']+')'
      elif intents[-1] == 'Get_rating':
        query = 'inform(address='+policyframe['content']['RATING']+')'

  return query

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
