import random

#lists needed
sent_key = ["time", "location", "category"]
sent_content = [time_list, location_list, category_list]

time_list = ["tonight","tomorrow","this morning","this afternoon",
"tomorrow afternoon","tomorrow morning","tomorrow night","the day after tomorrow", "now"]

location_list = ["astoria", "bayside", "bronx", "brooklyn", "college point", "corona",
"elmhurst", "flushing", "forest hills", "glendale", "howard beach", "jackson heights",
"jamaica", "long island city", "maspeth", "new york", "ozone park", "rego park", "ridgewood",
"south ozone park", "staten island", "sunnyside", "tompkinsville", "woodside"]

category_list = ["afghan","african","andalusian","arabian","argentine","armenian","asian fusion","asturian","australian","austrian","baguettes","bangladeshi","basque","bavarian","barbeque","beisl","belgian","bistros","black sea","brasseries","brazilian","breakfast & brunch","british","buffets","bulgarian","burgers","burmese","cafes","cafeteria","cajun","creole","cambodian","canteen","caribbean","catalan","cheesesteaks","chilean","chinese","comfort food","corsican","creperies","cuban","curry sausage","cypriot","czech","slovakian","czech","danish","delis","diners","dumplings","eastern european","parent cafes","ethiopian","filipino","fischbroetchen","fish & chips","flatbread","fondue","freiduria","french","galician","gastropubs","georgian","german","giblets","gluten-free","greek","guamanian","halal","hawaiian","heuriger","nepalese","himalayan","honduran","hot dogs","fast food","hot pot","hungarian","iberian","indonesian","indian","international","irish","israeli","italian","japanese","jewish","kebab","kopitiam","korean","kosher","kurdish","laos","laotian","latin american","lyonnais","malaysian","meatballs","mediterranean","mexican","middle eastern","modern australian","modern european","mongolian","moroccan","american (new)","canadian (new)","new mexican cuisine","new zealand","nicaraguan","night food","nikkei","noodles","norcinerie","traditional norwegian","open sandwiches","oriental","pakistani","pan asian","parma","iranian","persian","peruvian","pf/comercial","pita","pizza","polish","pop-up","portuguese","potatoes","poutineries","pub food","live food","raw food","rice","romanian","rotisserie chicken","rumanian","russian","salad","sandwiches","scandinavian","schnitzel","scottish","seafood","serbo croatian","signature cuisine","singaporean","slovakian","soul food","soup","southern","spanish","sri lankan","steakhouses","french southwest","supper clubs","sushi bars","swabian","swedish","swiss food","syrian","tabernas","taiwanese","small plates","tapas plates","tavola calda","tex-mex","thai","american (traditional)","traditional swedish","trattorie","turkish","ukrainian","uzbek","vegan","vegetarian","venison","vietnamese","waffles","wok","wraps","yugoslav"]

intent_list = ["get_restaurant", "get_location", "get_rating"]

content_list = ["category", "time", "location"]


memory = {"intent": "",
				"location": "",
				"time": "",
				"category": "",
				"rest_name": ""}



def simul_user(sys_act, initial):
	'''
	sys_act: {
			  "intent": "request",
			  "content": "location"
			  }

	initial: 1 or 0, indicating whether it's the beginning of a dialogue
	'''

	# initially randomly generated a sentence
	if initial == 1:
		sem_frame = {"intent": "",
				"location": "",
				"time": "",
				"category": "",
				"rest_name": ""}
		
		dec = random.randint(0,2) #randomly pick a intent
		if dec == 0: #get restaurant
			sem_frame["intent"] = intent_list[dec]
			for i in range(3):
				dec1 = random.randint(0,1)
				if dec1 == 1:
					sem_frame[sent_key[i]] = random.choice(sent_content[i])
		
		if dec in [1, 2]: #get location/rating
			sem_frame["intent"] = intent_list[dec]
			sem_frame["rest_name"] = random.choice(restaurant_list)

		memory = sem_frame #keep the memory
		nlg(sem_frame)
		return


	#in the middle of the dialogue	
	else:
		
		if sys_act["intent"] == "request":
			sem_frame["intent"] = "inform"

			if "location" in sys_act["content"]:
				sem_frame["location"] = random.choice(location_list)
				memory["location"] = sem_frame["location"]

			if "time" in sys_act["content"]:
				sem_frame["time"] = random.choice(time_list)
				memory["time"] = sem_frame["time"]

			if "category" in sys_act["content"]:
				sem_frame["category"] = random.choice(category_list)
				memory["category"] = sem_frame["category"]

			#if "rest_name" in sys_act["content"]:
			#	sem_frame["rest_name"] = random.choice(restaurant_list)

		elif sys_act["intent"] == "inform":
			sem_frame["intent"] = "thanks"
		
		elif sys_act["intent"] == "confirm":
			keys = sys_act["content"].keys()
			for key in keys:
				if sys_act["content"][key] != memory[key]:
					sem_frame["intent"] = "no"
					break
				if key == keys[-1]:
					sem_frame["intent"] = "yes"

		nlg(sem_frame)
		return