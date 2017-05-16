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


def AgentOutput(rawdata):
  channel = grpc.insecure_channel('140.112.49.151:50055')
  stub = FoodBot_pb2.FoodBotRequestStub(channel)

  request = FoodBot_pb2.Sentence(response = rawdata)
  result = stub.GetResponse(request)

  if not result.response_nlg : # means wrong intent.
	  return ''
  return result.response_policy_frame


def SimOutput(rawdata):
  channel = grpc.insecure_channel('140.112.49.151:50054')
  stub = FoodBotSim_pb2.FoodBotSimRequestStub(channel)

  request = FoodBotSim_pb2.Sentence(response = rawdata)
  result = stub.GetSimResponse(request)

  return result.response

if __name__ == '__main__':
  i = 0;
  while (i < 100):
  	msgToSend = SimOutput("init")
	print ("init-sent")
  	while(True):
  		if json.loads(msgToSend)["nlg_sentence"] == 'END' or json.loads(msgToSend)["nlg_sentence"] == ''or json.loads(msgToSend)["nlg_sentence"] == 'Unknown intent!!!':
  			inputss = json.loads(msgToSend)
  			inputss["nlg_sentence"] = 'end'
  			inputss = json.dumps(inputss)
  			print (inputss)
  			msgToSend = AgentOutput(inputss)
  			break
  		else:	
  			msgToSend = AgentOutput(msgToSend)
			if not msgToSend:
				print ("====== intent wrong  detected!!! ======")
				break
  			msgToSend = SimOutput(msgToSend)
  	i = i + 1


