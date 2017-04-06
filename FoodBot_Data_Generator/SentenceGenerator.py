# -*- encoding: UTF-8 -*-
import json
from nltk.tokenize import wordpunct_tokenize
import time
import os
import glob
import random 

def main():
    '''
    replacedDB = dict()
    replacedDB['sentence'] = "I like to eat REPLACEDWORD_1 REPLACEDWORD_2"
    replacedDB['REPLACEDWORD_1'] = [[["chinese", "food"],["B-Food","I-Food"]],[["food"],["B-Food"]],[["sushi"],["B-Food"]]]
    replacedDB['REPLACEDWORD_2'] = [ [["tonight"],["B-Time"]],[["this","morning"],["O","B-Time"]] ]
    #tokenizeSentence = wordpunct_tokenize(replacedDB['sentence'])
    
    print replacedDB
    with open("./test.json", 'w') as outfile:
        json_obj = json.dump(replacedDB,outfile)
    '''

    if not os.path.exists("./GeneratedData"):
        os.makedirs("./GeneratedData")

    #get all seed files
    list_of_seed = glob.glob("./seeds/*.json")
    print list_of_seed

    domain_list = list()
    label_list = list()
    tagging_list = list()
    sentence_list = list()

    GlobalGeneratedPairs = list()
    
    for seedFilePath in list_of_seed:
        print seedFilePath
        with open(seedFilePath, 'r') as outfile:
            json_obj = json.load(outfile)

        replacedDB = dict()
        replacedDB['sentence'] = json_obj['sentence']
        replacedDB['domain'] = json_obj['domain']
        replacedDB['intent'] = json_obj['intent']

        tokenizeSentence = wordpunct_tokenize(replacedDB['sentence'])
        for entry in tokenizeSentence:
                if "REPLACEDWORD" in entry:
                    replacedDB[entry] = json_obj[entry]

        sentence_Slot_Tags = list()
        for entry in tokenizeSentence:
            sentence_Slot_Tags.append("O")
        replacedDB['sentence_Slot_Tags'] = sentence_Slot_Tags
        generatedPairs = list()
        generatedPairs.append([tokenizeSentence,sentence_Slot_Tags])

        Generatedsentence_Slot_Tags = []
        GeneratedSentence = []

        while True:
            zippedData = generatedPairs[0]
            replacedIndex = []
            i = 0
            for entry in zippedData[0]:
                if "REPLACEDWORD" in entry:
                    replacedIndex.append((i,entry))
                    break
                i = i+1
            if len(replacedIndex) == 0:
                break
            else:
                generatedPairs.pop(0)
            
            for entry in replacedIndex:
                dict_name = entry[1]
                replacedData = replacedDB[dict_name]  #[["chinese", "food"],["B-Food","I-Food"]],[]..
                textContent = list()
                tagList = list()
                for dataEntry in replacedData:   #[["chinese", "food"],["B-Food","I-Food"]]
                    GeneratedSentence = list(zippedData[0])
                    Generatedsentence_Slot_Tags = list(zippedData[1])
                    
                    textContent = list(dataEntry[0])
                    tagList     = list(dataEntry[1])
                    textContent.reverse()
                    tagList.reverse()

                    #remove the REPLACEWORD_XX
                    GeneratedSentence.remove(entry[1])
                    #remove the original tag
                    replaceIndex = entry[0]
                    Generatedsentence_Slot_Tags.pop(replaceIndex)

                    for index in range(len(tagList)):
                        GeneratedSentence.insert(replaceIndex,textContent[index])
                        Generatedsentence_Slot_Tags.insert(replaceIndex,tagList[index])
                    #print GeneratedSentence
                    #print Generatedsentence_Slot_Tags
                    generatedPairs.append([GeneratedSentence,Generatedsentence_Slot_Tags,replacedDB['domain'],replacedDB['intent']])
        #print len(generatedPairs)
        GlobalGeneratedPairs.extend(generatedPairs)
        #with open("./output.json", 'w') as outfile:
        #    json_obj = json.dump(generatedPairs,outfile)
        '''
        for entry in generatedPairs:
            print '================================================================================'
            print entry[0]
            print entry[1]
            print entry[2]
            print entry[3]
        '''

    timestamp  = str(int(time.time()))
    if not os.path.exists("./GeneratedData/"+ timestamp):
        os.makedirs("./GeneratedData/"+ timestamp)

    if not os.path.exists("./GeneratedData/"+ timestamp +"/test"):
        os.makedirs("./GeneratedData/"+ timestamp + "/test")
    if not os.path.exists("./GeneratedData/"+ timestamp + "/train"):
        os.makedirs("./GeneratedData/"+ timestamp + "/train")
    if not os.path.exists("./GeneratedData/"+ timestamp + "/valid"):
        os.makedirs("./GeneratedData/"+ timestamp + "/valid")
    
    # For training data - random number is less than 0.8
    Train_GlobalGeneratedPairs = []
    Test_GlobalGeneratedPairs = []
    Valid_GlobalGeneratedPairs = []

    def writeTo(GlobalGeneratedPairs,mode):
        # Write the test.seq.in file
        with open("./GeneratedData/"+timestamp +"/"+ mode+"/"+mode+".seq.in" , 'w') as outfile:
            for entry in GlobalGeneratedPairs:
                for word in entry[0]:
                    #print word
                    #print word.lower()
                    outfile.write("%s " % word.lower())
                outfile.write("\n")
        # Write test.seq.out file
        with open("./GeneratedData/"+timestamp +"/"+mode+ "/"+mode+ ".seq.out", 'w') as outfile:
            for entry in GlobalGeneratedPairs:
                for word in entry[1]:
                    outfile.write("%s " % word)
                outfile.write("\n")
        # Write test.label file
        with open("./GeneratedData/"+timestamp +"/"+mode+"/"+ mode+".label", 'w') as outfile:
            for entry in GlobalGeneratedPairs:
                outfile.write("%s " % entry[3])
                outfile.write("\n")

    for index in range(len(GlobalGeneratedPairs)):
        rand = random.random()
        if(rand<=0.8):
            Train_GlobalGeneratedPairs.append(GlobalGeneratedPairs[index])
        else:
            rand2 = random.random()
            if(rand2>0.5):
                Test_GlobalGeneratedPairs.append(GlobalGeneratedPairs[index])
            else:
                Valid_GlobalGeneratedPairs.append(GlobalGeneratedPairs[index])

    writeTo(Train_GlobalGeneratedPairs,"train")
    writeTo(Test_GlobalGeneratedPairs,"test")
    writeTo(Valid_GlobalGeneratedPairs,"valid")

    

if __name__ == "__main__":
    main()