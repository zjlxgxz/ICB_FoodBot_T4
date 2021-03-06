import random
import sys
import grpc
import time
import collections
import json

from concurrent import futures
sys.path.append('../FoodBot_GRPC_Server/')
import FoodBotSim_pb2

#lists needed
time_list = ["tonight", "tomorrow", "this morning", "this afternoon", "tomorrow afternoon",
             "tomorrow morning", "tomorrow night", "the day after tomorrow", "now"]

location_list = ["astoria", "bayside", "bronx", "brooklyn", "college point", "corona",
                 "elmhurst", "flushing", "forest hills", "glendale", "howard beach",
								         "jackson heights", "jamaica", "long island city", "maspeth", "new york",
								         "ozone park", "rego park", "ridgewood", "south ozone park", "staten island",
								         "sunnyside", "tompkinsville", "woodside"]

category_list = ["afghan", "african","andalusian","arabian","argentine","armenian","asian fusion","asturian","australian","austrian","baguettes","bangladeshi","basque","bavarian","barbeque","beisl","belgian","bistros","black sea","brasseries","brazilian","breakfast & brunch","british","buffets","bulgarian","burgers","burmese","cafes","cafeteria","cajun","creole","cambodian","canteen","caribbean","catalan","cheesesteaks","chilean","chinese","comfort food","corsican","creperies","cuban","curry sausage","cypriot","czech","slovakian","czech","danish","delis","diners","dumplings","eastern european","parent cafes","ethiopian","filipino","fischbroetchen","fish & chips","flatbread","fondue","freiduria","french","galician","gastropubs","georgian","german","giblets","gluten-free","greek","guamanian","halal","hawaiian","heuriger","nepalese","himalayan","honduran","hot dogs","fast food","hot pot","hungarian","iberian","indonesian","indian","international","irish","israeli","italian","japanese","jewish","kebab","kopitiam","korean","kosher","kurdish","laos","laotian","latin american","lyonnais","malaysian","meatballs","mediterranean","mexican","middle eastern","modern australian","modern european","mongolian","moroccan","american (new)","canadian (new)","new mexican cuisine","new zealand","nicaraguan","night food","nikkei","noodles","norcinerie","traditional norwegian","open sandwiches","oriental","pakistani","pan asian","parma","iranian","persian","peruvian","pf/comercial","pita","pizza","polish","pop-up","portuguese","potatoes","poutineries","pub food","live food","raw food","rice","romanian","rotisserie chicken","rumanian","russian","salad","sandwiches","scandinavian","schnitzel","scottish","seafood","serbo croatian","signature cuisine","singaporean","slovakian","soul food","soup","southern","spanish","sri lankan","steakhouses","french southwest","supper clubs","sushi bars","swabian","swedish","swiss food","syrian","tabernas","taiwanese","small plates","tapas plates","tavola calda","tex-mex","thai","american (traditional)","traditional swedish","trattorie","turkish","ukrainian","uzbek","vegan","vegetarian","venison","vietnamese","waffles","wok","wraps","yugoslav"]

#intent_list = ["get_restaurant", "get_location", "get_rating", "get_comment", "wrong_domain"]
intent_list = ["get_restaurant", "get_location", "get_rating"]

content_list = ["category", "time", "location"]

sent_key = ["time", "location", "category"]
sent_content = [time_list, location_list, category_list]

#files
f1 = open('restName.txt', 'r')
restaurant_list1 = f1.read().split('\n')
restaurant_list = [item.replace('-', ' ') for item in restaurant_list1]

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

f9 = open('sentence_pattern/inform_location.txt', 'r')
inform_location_pattern = f9.read().split('\n')

f10 = open('sentence_pattern/inform_category.txt', 'r')
inform_category_pattern = f10.read().split('\n')

f11 = open('sentence_pattern/wrong_domain_questions.txt', 'r')
wrong_domain_list = f11.read().split('\n')

f12 = open('sentence_pattern/change_list.txt', 'r')
change_list = f12.read().split('\n')


memory = {"intent": "",
				"location": "",
				"time": "",
				"category": "",
				"restaurantname": ""}

#num_of_no = 0
num_of_quest_rest = 0
num_of_quest_info = 0


class FoodbotSimRequest(FoodBotSim_pb2.FoodBotSimRequestServicer):
  """Provides methods that implement functionality of route guide server."""
  def GetSimResponse (self, request, context):
    userInput = request.response.lower()
    print("Output from LU", userInput)
    return FoodBotSim_pb2.Sentence(response = simul_user(userInput))


def simul_user(sys_act):
	global memory
	#global num_of_no
	global num_of_quest_rest
	global num_of_quest_info
	
	'''
	sys_act: {
			  "intent": "request",
			  "content": {"location":""}
			  }

	'''

	# initially randomly generated a sentence
	sem_frame = {"intent": "",
				"location": "",
				"time": "",
				"category": "",
				"restaurantname": ""}
 
	if sys_act == "init":
		#intent_list = ["get_restaurant", "get_location", "get_rating", "get_comment", "wrong_domain"]
		num_of_quest_rest = 0
		num_of_quest_info = 0
		dec = random.randint(0,2) #randomly pick a intent
		sem_frame["intent"] = intent_list[dec]
		
		if dec == 0: #get restaurant
			for i in range(3):
				dec1 = random.randint(0,1)
				if dec1 == 1:
					sem_frame[sent_key[i]] = random.choice(sent_content[i])
		
		if dec in [1, 2]: #get location/rating
			sem_frame["restaurantname"] = random.choice(restaurant_list)

		memory = sem_frame #keep the memory


	#in the middle of the dialogue	
	else:
		#print("memory: ", memory)
		sys_act = json.loads(sys_act)

		if sys_act["intent"] == "request":
			sem_frame["intent"] = "inform"
			#print("content keys:", sys_act["content"].keys())
			if "location" in sys_act["content"].keys():
				sem_frame["location"] = random.choice(location_list)
				memory["location"] = sem_frame["location"]

			if "time" in sys_act["content"].keys():
				sem_frame["time"] = random.choice(time_list)
				memory["time"] = sem_frame["time"]

			if "category" in sys_act["content"].keys():
				sem_frame["category"] = random.choice(category_list)
				memory["category"] = sem_frame["category"]

			if "restaurantname" in sys_act["content"].keys():
				sem_frame["restaurantname"] = memory["restaurantname"]


		elif sys_act["intent"] == "inform":
			if memory["intent"] == "no" or num_of_quest_rest == 2 or num_of_quest_info == 1: #quest for more than twice
				sem_frame["intent"] = "end"

			else:
				sem_frame["intent"] = "end"

				if "restaurantname" in sys_act["content"].keys(): #agent recommended a restaurant
					dec2 = random.randint(0,2)
					#print("dec2: ", dec2)
					if dec2 == 0:
						sem_frame["intent"] = "change"  #to add variety, user may ask for a change of restaurant
						num_of_quest_rest = num_of_quest_rest + 1
					if dec2 == 1:
						sem_frame["intent"] = "get_rating" #change intent to get rating
						sem_frame["restaurantname"] = sys_act["content"]["restaurantname"]
						num_of_quest_rest = num_of_quest_rest + 1
				
				else:
					dec2 = random.randint(0,1)
					#print("dec2: ", dec2)
					if "location" in sys_act["content"].keys() and dec2 == 0:
						sem_frame["intent"] = "get_rating"
						sem_frame["restaurantname"] = memory["restaurantname"]
						num_of_quest_info = num_of_quest_info + 1

					
					if "rating" in sys_act["content"].keys():
						if num_of_quest_rest == 1: #last question is get restaurant
							dec3 = random.randint(0,1)
							#print("dec3: ", dec3)
							if dec3 == 0:
								sem_frame["intent"] = "end"
							else:
								sem_frame["intent"] = "change"
								num_of_quest_rest = num_of_quest_rest + 1
						
						else: # last question is get rating
							if dec2 == 0:
								sem_frame["intent"] = "get_location"
								sem_frame["restaurantname"] = memory["restaurantname"]
								num_of_quest_info = num_of_quest_info + 1
					
		
		elif sys_act["intent"]  == "confirm_restaurant":
			keys = sys_act["content"].keys()
			#print("content keys:", keys)
			for key in keys:
				if sys_act["content"][key] != memory[key.lower()]:
					sem_frame["intent"] = "no"
					memory["intent"] = "no"
					break
				if key == keys[-1]:
					sem_frame["intent"] = "yes"
		
		elif sys_act["intent"]  == "confirm_info":
			sem_frame["intent"] = "no"
			keys = sys_act["content"].keys()
			#print("content keys:", keys)
			
			if "location" in keys: #confirm whether user's intent is get location
				if "location" == memory["intent"][4:] and sys_act["content"]["restaurantname"] == memory["restaurantname"]: #get_location
					sem_frame["intent"] = "yes"
			
			if "rating" in keys: #confirm whether user's intent is get rating
				if "rating" == memory["intent"][4:] and sys_act["content"]["restaurantname"] == memory["restaurantname"]:
					sem_frame["intent"] = "yes"
			
			if sem_frame["intent"] == "no":
				memory["intent"] = "no"

		elif sys_act["intent"] == "not_found":
			sem_frame["intent"] = "end"
		
        
	#print("num_of_quest_rest: ", num_of_quest_rest)
	#print("num_of_quest_info: ", num_of_quest_info)
	nlg_sentence = nlg(sem_frame)

	returnList = dict()
	returnList["nlg_sentence"] = nlg_sentence
	returnList["semantic_frame"] = sem_frame
	
	json_list = json.dumps(returnList)
	return json_list


def nlg(sem_frame):
	#print("semantic frame: ", sem_frame)
	sentence = ""

	if sem_frame["intent"] == "end": #end of the dialogue
		sentence = "END"

	
	elif sem_frame["intent"] == "yes":
		sentence = random.choice(yes_list)
	
	elif sem_frame["intent"] == "no":
		sentence = "disagree"
		#sentence = "No. I was asking for"
		#if memory["intent"] == "get_restaurant":
		#	sentence = sentence + " a " + memory["category"] + " restaurant in " + memory["location"] + " for " + memory["time"] + "."
		#else:
		#	sentence = sentence + " the " + memory["intent"][4:] + " of " + memory["restaurantname"] + "."
		
	elif sem_frame["intent"] == "wrong_domain":
		sentence = random.choice(wrong_domain_list)
		

	elif sem_frame["intent"] == "change": #ask for changing restaurant
		sentence = random.choice(change_list)
	
	elif sem_frame["intent"] == "inform": # category/time/location      
		if sem_frame["category"]:
			sentence = random.choice(inform_category_pattern)
			sentence = sentence.replace("CATEGORY", sem_frame["category"])
			sentence = sentence.capitalize()
		if sem_frame["location"]:
			if sentence:
				pre = " "
			else:
				pre = ""
			sentence = sentence + pre + random.choice(inform_location_pattern)
			sentence = sentence.replace("LOCATION", sem_frame["location"])
		if sem_frame["time"]:
			if sentence:
				pre = " "
			else:
				pre = ""
			sentence = pre + sem_frame["time"].capitalize()
		if sem_frame["restaurantname"]:
			sentence = sem_frame["restaurantname"].capitalize()

	elif sem_frame["intent"] == "get_restaurant":
		# replace category, replace location with "in xxx", time with "for xxx"
		sentence = random.choice(get_restaurant_pattern)
		for item in content_list:
			if sem_frame[item] == "":
				sentence = sentence.replace(item.upper(), "")
			else:
				if item == "category":
					prefix = ' '
				if item == "location":
					prefix = " in "
				if item == "time":
					prefix = " for "
				sentence = sentence.replace(item.upper(), prefix + sem_frame[item])

	
	elif sem_frame["intent"] == "get_location":
		sentence = random.choice(get_location_pattern)
		sentence = sentence.replace("RESTAURANT_NAME", sem_frame["restaurantname"])    
	

	elif sem_frame["intent"] == "get_rating":
		sentence = random.choice(get_rating_pattern)
		sentence = sentence.replace("RESTAURANT_NAME", sem_frame["restaurantname"])

	else:
		sentence = "END"
	
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