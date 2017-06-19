import random
import sys
import grpc
import time
import collections
import json
import glob
import string

from concurrent import futures
sys.path.append('../FoodBot_GRPC_Server/')
import FoodBotSim_pb2

#lists needed
with open("./rest_name.txt", "r") as f1:
	name_list = f1.read().split('\n')
	name_list = [sent for sent in name_list if sent != '']
with open("./sentence_pattern/data_dict.json", "r") as f2:
	data_dict = json.load(f2)
	data_dict['name'] = name_list
with open("./sentence_pattern/slot_dict.json", "r") as f3:
	slot_dict = json.load(f3)

f1.close()
f2.close()
f3.close()

#sentence patterns
pattern_dict = dict()
file_list = glob.glob("./sentence_pattern/*.txt")
for txt in file_list:
	f = open(txt, 'r')
	key = txt.split('/')[2].split('.')[0]
	temp = f.read().split('\n')
	temp = [sent for sent in temp if sent != '']
	pattern_dict[key] = temp


memory = dict()

goodPolicy = 0

class FoodbotSimRequest(FoodBotSim_pb2.FoodBotSimRequestServicer):
  """Provides methods that implement functionality of route guide server."""
  def GetSimResponse (self, request, context):
    userInput = request.response.lower()
    #print("Output from LU", userInput)
    return FoodBotSim_pb2.Sentence(response = simul_user(userInput))

def policyChecker(sem_frame ,sys_act):
	#print("This is sys_act",sys_act)
	#print("This is sys_act[content]",sys_act['content'])
	#print("This is sys_act[intent]",sys_act['intent'])
	#print("This is sys_act[currentState]",sys_act['currentstate'])
	
	print '=========================='
	print 'memory : ', memory
	#print 'expect : ', expect
	#print 'confirm : ', confirm
	print 'sys_act : ', sys_act

	soso = []
	if sem_frame["intent"] == "no_more":
		good = ["confirm_restaurant"]
	elif sem_frame["intent"] in ["goodbye", "thanks"]:
		good = ["goodbye"]
	elif sem_frame["intent"] == "reject":
		good = ["show_table"]

	else:
		if memory["intent"] == "hi":
			good = ["hi"]
	
		elif memory["intent"] == "request_restaurant":
			if memory.keys() <=3:
				keys = memory.keys()
				keys.remove("intent")
				request_list = ["area", "category", "time"]
				good = ["reqmore"] + ["request_"+item for item in request_list if item not in keys]
			else:
				good = ["confirm_restaurant"]
				soso = ["reqmore"]
		
		else:
			if memory.keys() <=2:
				keys = memory.keys()
				keys.remove("intent")
				request_list = ["area", "name"]
				good = ["request_"+item for item in request_list if item not in keys]
			else:
				good = ["confirm_info"]
	
	if sys_act["policy"] in good:
		return 5
	elif sys_act["policy"] in soso:
		return 2
	else:
		return 0


def simul_user(sys_act):
	global memory	
	'''
	sys_act: {
			  "policy": "request_category",			  
			  }
	'''
	# initially randomly generated a sentence
	sys_act = json.loads(sys_act)
	sem_frame = dict()
	if sys_act["policy"] in ["init", "hi"]:
		intents = data_dict["intent"]
		if sys_act["policy"] == "hi":
			intents.remove("hi")
		sem_frame["intent"] = random.choice(intents)
		if sem_frame["intent"] != "hi":
			keys = slot_dict[sem_frame["intent"]]
			for key in keys:
				dec = round(random.random())
				if dec == 1:
					sem_frame[key] = random.choice(data_dict[key])
		memory = sem_frame  #keep the memory

	#in the middle of the dialogue	
	else:
		#print("memory: ", memory)		
		global goodPolicy
		# To see if the policy picked by DQN is reasonable
		goodPolicy = policyChecker(sem_frame, sys_act)
		print (goodPolicy)
		if goodPolicy == 0:
			returnList = dict()
			returnList["nlg_sentence"] = 'not a good policy'
			returnList["goodpolicy"] = goodPolicy
			returnList["user_id"] = 'sim-user'
			json_list = json.dumps(returnList)
			return json_list

################## request #########################
		if "request" in sys_act["policy"]:
			sem_frame["intent"] = "inform"
			#print("content keys:", sys_act["content"].keys())
			key = sys_act["policy"].split('_')[1]
			sem_frame[key] = random.choice(data_dict[key])
			memory[key] = sem_frame[key]

		elif sys_act["policy"] == "reqmore":			
			dec = round(random.random())
			if memory.keys() <= 3:
				sem_frame["intent"] = "inform"
				temp = [item for item in slot_dict[sem_frame["intent"]] if item not in memory.keys()]
				key = random.choice(temp)
				sem_frame[key] = random.choice(data_dict[key])
				memory[key] = sem_frame[key]
			else:
				sem_frame["intent"] = "no_more"

################## inform #########################
		elif "inform" in sys_act["policy"]:
			sem_frame["intent"] = random.choice(["goodbye", "thanks"])
					
################## confirm #########################		
		elif "confirm" in sys_act["policy"]:
			keys = sys_act.keys()
			keys.remove("policy")
			sem_frame["intent"] == "confirm"
			for key in keys:
				if key not in memory.keys() or (key in memory.keys() and sys_act[key] != memory[key]):
					sem_frame["intent"] == "reject"
					break

		else:
			sem_frame["intent"] = 'error'

	#nlg_sentence = nlg(sem_frame)

	returnList = dict()
	#returnList["nlg_sentence"] = nlg_sentence
	returnList["semantic_frame"] = sem_frame
	returnList["goodpolicy"] = goodPolicy
	returnList["user_id"] = 'sim-user'
	
	json_list = json.dumps(returnList)
	return json_list


def nlg(sem_frame):
	#print("semantic frame: ", sem_frame)
	if sem_frame["intent"] == "error":
		sentence = "This is an alert: unknown policy intent!"
		return sentence

	sentence = random.choice(pattern_dict[sem_frame["intent"]])
	if sem_frame["intent"]  not in ["confirm", "reject", "hi", "thanks", "goodbye"]:
		keys = sem_frame.keys()
		keys.remove("intent")
		for key in keys:
			sentence = sentence.replace(key.upper(), ' '+sem_frame[key])
		transtab = string.maketrans(string.uppercase, '')
		sentence = sentence.translate(transtab)

	return sentence


if __name__ == "__main__":
	# The model has been loaded.
	server = grpc.server(futures.ThreadPoolExecutor(max_workers=3))
	#Service_OpenFace_pb2.add_openfaceServicer_to_server(Servicer_openface(), server)
	FoodBotSim_pb2.add_FoodBotSimRequestServicer_to_server(FoodbotSimRequest(),server)
	server.add_insecure_port('[::]:50054')
	server.start()
	print ("GRCP Server is running. Press any key to stop it.")
	try:
		while True:
			# openface_GetXXXXXX will be responsed if any incoming request is received.
			time.sleep(48*60*60)
	except KeyboardInterrupt:
		server.stop(0)
