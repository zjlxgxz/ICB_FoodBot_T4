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
good = []
goodPolicy = 0

class FoodbotSimRequest(FoodBotSim_pb2.FoodBotSimRequestServicer):
  """Provides methods that implement functionality of route guide server."""
  def GetSimResponse (self, request, context):
    userInput = request.semantic_frame.lower()
    #print("Output from LU", userInput)
    backdata = simul_user(userInput)
    '''
    string semantic_frame = 1;
	string nlg_sentence = 2;
	string user_id = 3;
	string good_policy = 4;
    '''

    return FoodBotSim_pb2.Sentence(semantic_frame = json.dumps(backdata['semantic_frame']),nlg_sentence = backdata['nlg_sentence'],user_id = backdata['user_id'],good_policy =backdata['goodpolicy'] )


def policyChecker(sys_act):
	#print("This is sys_act",sys_act)
	#print("This is sys_act[content]",sys_act['content'])
	#print("This is sys_act[intent]",sys_act['intent'])
	#print("This is sys_act[currentState]",sys_act['currentstate'])
	global good
	global memory

	soso = []
	if memory["intent"] in ["goodbye", "thanks"]:
		good = ["goodbye"]

	
	elif memory["intent"] == "hi":
		good = ["hi"]
	
#######for test#############
	elif memory["intent"] == "confirm":
		good = ["inform"]
############################
	
	elif memory["intent"] == "request_restaurant":
		if len(memory.keys()) <=3:
			keys = memory.keys()
			keys.remove("intent")
			request_list = ["area", "category"]
			good = ["request_more"] + ["request_"+item for item in request_list if item not in keys]
		else:
			good = ["confirm_restaurant"]
			soso = ["request_more"]
	
	elif memory["intent"] == "reject":
		pass

	else:
		if len(memory.keys()) <=1:
			#keys = memory.keys()
			#keys.remove("intent")
			#request_list = ["area", "name"]
			#good = ["request_"+item for item in request_list if item not in keys]
			good = ["request_name"]
		else:
			good = ["confirm_info"]
	
	if sys_act["policy"] in good:
		reward = 5
	elif sys_act["policy"] in soso:
		reward = 2
	else:
		reward = 0

	print ('==========================')
	print ('memory : ', memory)
	print ('sys_act : ', sys_act)
	print ('good : ', good)
	good = []
	return reward


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

		elif sys_act["policy"] == "request_more":			
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
