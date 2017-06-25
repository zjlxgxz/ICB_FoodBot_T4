# README
This repo contains code for a chatbot in food domain. [Demo link](http://140.112.49.151:8081/index3.html)

## Functionalities
* **Recommend restaurants** according to requirements, including category, price, area, rating.
* **Provide various info about a certain restaurant**, including address, rating, review, has wifi or not.

## Features
* Support 2 talking styles: gentle & hilarious.
* Support speech API.

## Requirements
* Tensorflow: 0.12.1
`pip install tensorflow==0.12.1`
* Nodejs: [Download](https://nodejs.org/en/download/package-manager/)

## Code Usage
1. Start web server `node client2.js`
2. Brows on borowser `localhost:8081/index3.html`
3. Enjoy

## Implementation
### 1. Ontology
* Data Source: (using the API and crawler)
* Data Size: 1000 restaurants information from New York.
* Number of tables: 3
    * Restaurant Table: (about 1000 rows)
    * Comment Table: (about 300000 rows)
    * Other info of Restaurant Table:(about 1000 rows)

### 2. Language Understanding
* The LU is implmented based on Attention-Based Recurrent Neural Network Models for Joint Intent Detection
and Slot Filling by Bing Liu.
* The training data is composed of 12 intents:
    * confirm
    * goodbye
    * hi
    * inform
    * inform_price
    * inform_score
    * reject
    * request_address
    * request_restaurant
    * request_review
    * request_score
    * request_wifi
    * thanks
* All the artificial generated sentence patterns are included in the folder FoodBot_Data_Generator/seeds/, we use the SentenceGennerator.py to generate artificial training data from those sentence patterns. The generated data is included in the Foodbot_Data_generator/GeneratedData. Each sub-folder,ex: Foodbot_Data_generator/GeneratedData/1497933425, in it is included the traning/validation/teseting data.
* To train the model, user should copy the sub-folder into FoodBot_Training/data/ATIS_samples and then run the shell command ./Foodbot_training/runmodel.sh
* After training, the model will be saved in the path FoodBot_Training/model_tmp_XXXXX. To use it, user should copy it to the path FoodBot_Testing and rename it to become "model_tmp". Then run the shell command ./FoodBot_Testing/runmodel.sh. Then the program will load the model and open the GRPC server.

### 3. Dialogue Management
* Dialogue State Tracking
    * There are two parts for DST to keep the action which user and agent took.
    * user :
        * request_restaurant
        * request_address
        * request_score
	    * request_review
	    * request_price
	    * request_time
	    * request_phone
	    * request_smoke
	    * request_smoke
	    * request_wifi
	    * inform
	    * confirm
	    * reject
	    * restaurant_name
	    * time
	    * area
	    * address
	    * category
	    * score
	    * price
	    * wifi
	    * smoking
    * agent :
	    * restaurant_name
	    * confirm_info
	    * confirm_restaurant

* DQN Reinforcement Learning
    * DQN is a classification consist of three fully connected layer.
    ![DQN](./img/RL.png)
    
* Agent Policy
    * request_area
    * request_category
    * request_name
    * reqmore
    * inform_address
    * inform_score
    * inform_review
    * inform_restaurant
    * inform_smoke
    * inform_wifi
    * inform_phone
    * inform_info
    * confirm_info
    * confirm_restaurant
    * goodbye
    * hi

### 4. Natural Language Generation
(1)NN-based
* An adapted version of a semantically conditioned LSTM: [RNNLG](https://github.com/shawnwun/RNNLG)
* Data collection: sentences with ~10 intents written by human
* Performance: 
    * BLEU score: 0.5518
    * Naturalness: see the example below
    * inform(name = 33 greenwich; address = 7593 kirkland lane rockaway)
    * => “33 greenwich is good good in 7593 kirkland lane rockaway address is in.”

(2)Template-based (using this one!)
* Template: 17 intents, each one with 5~8 templates.

### 5. UI
* Base on nodejs, include: grpc, express, socket, mysql, async
* Workflow:
	* Init: Reset Agent state with magic word "bye" ![init](./img/FoodBot-wf2.jpg)
	* Basic: in this flow when step.4 return, it include two type paremeter (responseMsg, responseUrl), responseMsg is what agent's answer. And responseUrl have two type of response, one is real url represent image of meme, the other is json obj. in this kind of response, will use other workflow(Show Table). ![base flow](./img/FoodBot-wf1.jpg)
	* Show Table:other case of display message ![show table](./img/showTable.png)
	
