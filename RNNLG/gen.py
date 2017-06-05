import random
import json
import glob

#read file
f1 = open('data_list/location_list.txt', 'r')
location_list = f1.read().split('\n')
location_list = [item for item in location_list if item != '']

f2 = open('data_list/time_list.txt', 'r')
time_list = f2.read().split('\n')
time_list = [item for item in time_list if item != '']

f3 = open('data_list/category_list.txt', 'r')
category_list = f3.read().split('\n')
category_list = [item for item in category_list if item != '']

f4 = open('data_list/rest_name.txt', 'r')
rest_name_list = f4.read().split('\n')
rest_name_list = [item for item in rest_name_list if item != '']

f5 = open('data_list/address_list.txt', 'r')
address_list = f5.read().split('\n')
address_list = [item for item in address_list if item != '']

f7 = open('data_list/rating_list.txt', 'r')
rating_list = f7.read().split('\n')
rating_list = [item for item in rating_list if item != '']

f6 = open("data_list/little.txt", 'r')
little_list = f6.read().split('\n')
little_list = [item for item in little_list if item != '']

info_list = ["rating", "address"]

#set dictionary
dact = dict()
file_list = glob.glob("./sentence_pattern/agent/*.txt")

for txt in file_list:
    key = txt.split('/')[3]
    key = key.split('.')[0]
    dact[key] = dict()
    #print key
    if key == "confirm_info":       
        dact[key]['frame'] = "confirm_info(info=SLOT_INFO;name=SLOT_NAME)"
        dact[key]['replace_value_list'] = [rest_name_list, info_list]
        dact[key]['replace_word_list'] = ['SLOT_NAME','SLOT_INFO']
    if key in ["confirm_restaurant", "inform_no_match"]:
        dact[key]['frame'] = key + "(category=SLOT_CATEGORY;time=SLOT_TIME;location=SLOT_LOCATION)"
        dact[key]['replace_value_list'] = [category_list, time_list, location_list]
        dact[key]['replace_word_list'] = ['SLOT_CATEGORY','SLOT_TIME', 'SLOT_LOCATION']
    if key == "inform_restaurant":
        dact[key]['frame'] = "inform(name=SLOT_NAME;address=SLOT_ADDRESS)"
        dact[key]['replace_value_list'] = [rest_name_list, address_list]
        dact[key]['replace_word_list'] = ['SLOT_NAME','SLOT_ADDRESS']
    if key == "inform_address":
        dact[key]['frame'] = "inform(address=SLOT_ADDRESS)"
        dact[key]['replace_value_list'] = [address_list]
        dact[key]['replace_word_list'] = ['SLOT_ADDRESS']
    if key == "inform_rating":
        dact[key]['frame'] = "inform(rating=SLOT_RATING)"
        dact[key]['replace_value_list'] = [rating_list]
        dact[key]['replace_word_list'] = ['SLOT_RATING']
    if key in ["request_category", "request_location", "request_time"]:
        dact[key]['frame'] = "?request(" + key.split('_')[1] + ")"
        #f = open(txt, 'r')
        dact[key]['replace_value_list'] = []
        dact[key]['replace_word_list'] = []


with open('dact.json', 'w') as f2:
    json.dump(dact, f2)
f2.close()


def replace_w(sent_list, token, value_list,not_frame):
    re_list = []
    for sent in sent_list:
        for value in value_list:
            if not_frame == True:
                re_sent = sent.replace(token, value.lower())
            else:
                re_sent = sent.replace(token, "'" + value.lower() + "'")
            re_list.append(re_sent)
    return re_list

def recur_replace(seed, token_list, value_list,not_frame):  
    for i, token in enumerate(token_list):
        seed = replace_w(seed, token, value_list[i],not_frame)
    return seed

'''
dact[key]['frame'] = "confirm_info(info = SLOT_INFO; name = SLOT_NAME)"
        dact[key]['replace_value_list'] = [little_list, info_list]
        dact[key]['replace_word_list'] = ['SLOT_NAME','SLOT_INFO']
'''

'''
[
        "inform(name=piperade;goodformeal=dinner;food=basque)",
        "piperade is good for dinner and serves basque",
        "piperade is a nice place , it is good for dinner and it serves basque food"
    ]
'''

train_data = []
valid = []
test = []
s = []
for j, key in enumerate(dact.keys()):
    print(j,len(dact.keys()))
    temp = recur_replace([dact[key]['frame']], dact[key]['replace_word_list'], dact[key]['replace_value_list'], True)
    sents = open('./sentence_pattern/agent/'+key+'.txt', 'r').read().split('\n')
    sents = [sent for sent in sents if sent != '']
    
    temp2 = []
    for sent in sents:
        temp2 += recur_replace([sent], dact[key]['replace_word_list'],dact[key]['replace_value_list'], True)

    count = 76
    #temp -> len(sents) * temp2
    if dact[key]['frame'].startswith('?'):
        for i in range(76):
            if count == 0:
                break
            sample = random.sample(temp2,2)
            train_item = temp + sample
            count -= 1
            if count <= 1:  
                test.append(train_item)                
            elif count <= 25:
                valid.append(train_item)
            else:
                train_data.append(train_item)
            #print(train_data[-1])
    else:
        for i, item in enumerate(temp):
            if count == 0:
                break
            sample = random.sample(temp2[i::len(temp)],2)
            train_item = [item] + sample
            count -= 1
            if count <= 1: 
                test.append(train_item)               
            elif count <= 25:
                valid.append(train_item)
            else:
                train_data.append(train_item)
            
    print train_data[-1]

with open('./data/original/restaurant/little_train.json', 'w') as f3:
    t = json.dumps(train_data, indent = 2)
    f3.write(t)

f3.close()

with open('./data/original/restaurant/little_valid.json', 'w') as f4:
    t = json.dumps(valid, indent = 2)
    f4.write(t)
f4.close()

with open('./data/original/restaurant/little_test.json', 'w') as f5:
    t = json.dumps(test, indent = 2)
    f5.write(t)
f5.close()
'''
print(train_data[:10])
#print(s[:5])
'''










            