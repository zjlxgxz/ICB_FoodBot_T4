from __future__ import print_function

import random
import time
import json
import grpc
import sys
sys.path.append('../FoodBot_GRPC_Server/')
import grpc
import FoodBot_pb2
import FoodBotSim_pb2


def AgentOutput(sem_frame,user_id,policyEval):
  channel = grpc.insecure_channel('140.112.49.151:50055')
  stub = FoodBot_pb2.FoodBotRequestStub(channel)

  request = FoodBot_pb2.Sentence(semantic_frame = sem_frame, nlg_sentence = '', user_id=user_id, good_policy =policyEval)
  result = stub.GetResponse(request)

  return result


def SimOutput(rawdata):
  channel = grpc.insecure_channel('140.112.49.151:50054')
  stub = FoodBotSim_pb2.FoodBotSimRequestStub(channel)

  request = FoodBotSim_pb2.Sentence(semantic_frame = rawdata,nlg_sentence = '',user_id='',good_policy =0) #json string containing policy frame and others are empty
  result = stub.GetSimResponse(request)

  return result

if __name__ == '__main__':
	i = 0
	while (i < 10000):
		initDict = dict()
		initDict["policy"] = "init"
		msgToSend = SimOutput( json.dumps(initDict))
		print ("init-sent Turns:",i)
		while(True):
			sim_semantic_frame = msgToSend.semantic_frame
			good_policy = msgToSend.good_policy
			sim_user_id = msgToSend.user_id

			msgToSend = AgentOutput(sim_semantic_frame,sim_user_id,good_policy)

			msgToSend = SimOutput( json.dumps(msgToSend.semantic_frame))

			if(sim_semantic_frame['intent'] == 'goodbye'):
				break
		i = i + 1


